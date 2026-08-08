"""LLM 适配器抽象基类。"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

from benchscore.config import ModelInfo


@dataclass
class GenerationResult:
    """单次生成的结果"""
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class BatchProgress:
    """批量生成进度"""
    completed: int
    total: int
    last_latency_ms: float


class BaseAdapter(ABC):
    """LLM 适配器抽象基类。

    所有模型提供商需要实现 generate() 方法。
    generate_batch() 在基类提供默认异步并发实现。
    """

    model_id: str
    provider: str
    model_info: ModelInfo | None

    def __init__(self, model_id: str, model_info: ModelInfo | None = None):
        self.model_id = model_id
        self.model_info = model_info

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> GenerationResult:
        """发送一次生成请求。

        Args:
            prompt: 用户消息
            system_prompt: 系统提示（可选）
            temperature: 温度参数
            max_tokens: 最大输出 token 数

        Returns:
            GenerationResult，包含文本、token 数、费用、延迟
        """
        ...

    async def generate_batch(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
        concurrency: int = 10,
        max_retries: int = 3,
        temperature: float = 0.0,
        on_progress: Callable[[BatchProgress], None] | None = None,
    ) -> list[GenerationResult]:
        """批量生成 — 异步并发 + 指数退避重试。

        Args:
            prompts: 待生成的 prompt 列表
            system_prompt: 系统提示
            concurrency: 最大并发数
            max_retries: 每个请求的最大重试次数
            temperature: 温度参数
            on_progress: 进度回调，接收 BatchProgress

        Returns:
            与 prompts 同长度的 GenerationResult 列表
        """
        semaphore = asyncio.Semaphore(concurrency)
        results: list[GenerationResult] = [None] * len(prompts)
        completed = 0
        total = len(prompts)
        lock = asyncio.Lock()

        async def _one(idx: int, prompt: str) -> None:
            nonlocal completed
            async with semaphore:
                for attempt in range(max_retries):
                    try:
                        result = await self.generate(
                            prompt=prompt,
                            system_prompt=system_prompt,
                            temperature=temperature,
                        )
                        results[idx] = result
                        break
                    except Exception as exc:
                        cls_name = type(exc).__name__
                        # 判断是否可重试（限流/服务端错误/超时）
                        retryable = _is_retryable(exc)
                        if not retryable or attempt == max_retries - 1:
                            results[idx] = GenerationResult(
                                text="",
                                error=f"{cls_name}: {exc}",
                            )
                            break
                        wait = 2 ** attempt  # 指数退避: 1s, 2s, 4s
                        await asyncio.sleep(wait)

                async with lock:
                    completed += 1
                    if on_progress:
                        on_progress(BatchProgress(
                            completed=completed,
                            total=total,
                            last_latency_ms=(
                                results[idx].latency_ms
                                if results[idx] and not results[idx].error
                                else 0
                            ),
                        ))

        tasks = [_one(i, p) for i, p in enumerate(prompts)]
        await asyncio.gather(*tasks, return_exceptions=True)

        return results

    def _calc_cost(self, input_tokens: int, output_tokens: int) -> float:
        """根据预置费率计算费用"""
        if self.model_info is None:
            return 0.0
        return (
            input_tokens / 1000 * self.model_info.cost_per_1k_input
            + output_tokens / 1000 * self.model_info.cost_per_1k_output
        )


def _is_retryable(exc: Exception) -> bool:
    """判断异常是否可重试。"""
    cls_name = type(exc).__name__
    msg = str(exc).lower()

    retryable_names = {
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
        "ServiceUnavailableError",
        "Timeout",
        "ConnectionError",
    }

    if cls_name in retryable_names:
        return True

    # 也用消息内容辅助判断
    retryable_keywords = [
        "rate limit", "too many requests", "429",
        "timeout", "timed out",
        "server error", "internal server error", "503", "502",
        "connection", "network",
        "overloaded",
    ]
    for kw in retryable_keywords:
        if kw in msg:
            return True

    return False
