"""Anthropic 适配器 — 支持 Claude 系列模型。"""

from __future__ import annotations

import time

from anthropic import AsyncAnthropic

from benchscore.adapters.base import BaseAdapter, GenerationResult
from benchscore.config import ModelInfo


class AnthropicAdapter(BaseAdapter):
    """Anthropic Claude 系列适配器"""

    provider = "anthropic"

    def __init__(
        self,
        model_id: str = "claude-sonnet-4-20250514",
        api_key: str = "",
        base_url: str | None = None,
        model_info: ModelInfo | None = None,
    ):
        super().__init__(model_id=model_id, model_info=model_info)
        client_kwargs = {
            "api_key": api_key,
            "timeout": 120.0,
            "max_retries": 0,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**client_kwargs)

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> GenerationResult:
        messages = [{"role": "user", "content": prompt}]

        t0 = time.monotonic()

        # Anthropic 的 system 是顶级参数，不在 messages 里
        kwargs = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        try:
            response = await self._client.messages.create(**kwargs)
        except Exception as api_err:
            raise RuntimeError(
                f"API 调用失败 (base_url={getattr(self._client, 'base_url', '默认')}, "
                f"model={self.model_id}): {api_err}"
            ) from api_err
        latency_ms = (time.monotonic() - t0) * 1000

        # 提取文本内容
        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text

        input_tokens = response.usage.input_tokens if response.usage else 0
        output_tokens = response.usage.output_tokens if response.usage else 0
        cost = self._calc_cost(input_tokens, output_tokens)

        return GenerationResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
        )
