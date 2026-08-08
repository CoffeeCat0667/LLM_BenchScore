"""OpenAI 适配器 — 支持 GPT-4o、GPT-4o-mini、o4-mini 等模型。"""

from __future__ import annotations

import time

from openai import AsyncOpenAI

from benchscore.adapters.base import BaseAdapter, GenerationResult
from benchscore.config import ModelInfo


class OpenAIAdapter(BaseAdapter):
    """OpenAI GPT 系列适配器"""

    provider = "openai"

    def __init__(
        self,
        model_id: str = "gpt-4o",
        api_key: str = "",
        base_url: str | None = None,
        model_info: ModelInfo | None = None,
    ):
        super().__init__(model_id=model_id, model_info=model_info)
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=120.0,
            max_retries=0,  # 我们在 generate_batch 自己管理重试
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> GenerationResult:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        t0 = time.monotonic()
        try:
            response = await self._client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as api_err:
            raise RuntimeError(
                f"API 调用失败 (base_url={self._client.base_url}, "
                f"model={self.model_id}): {api_err}"
            ) from api_err

        latency_ms = (time.monotonic() - t0) * 1000

        # 防护：非标准 API 可能返回非预期格式
        if not hasattr(response, 'choices'):
            hint = ""
            raw = str(response)
            if raw.strip().startswith("<!DOCTYPE") or raw.strip().startswith("<html"):
                hint = (
                    "\n⚠ API 返回了 HTML 页面而非 JSON。"
                    "\n   请检查 Base URL 是否遗漏了 /v1 后缀："
                    f"\n   当前: {self._client.base_url}"
                    f"\n   应为: {self._client.base_url.rstrip('/')}/v1"
                )
            raise RuntimeError(
                f"API 返回了非预期的类型 {type(response).__name__}，"
                f"Base URL 可能不正确。\n"
                f"当前 Base URL: {self._client.base_url}\n"
                f"模型: {self.model_id}\n"
                f"原始响应前 200 字符: {raw[:200]}"
                + hint
            )

        if not response.choices:
            raise RuntimeError(f"API 返回了空的 choices 列表 (base_url={self._client.base_url})")
        choice = response.choices[0]
        text = getattr(getattr(choice, "message", None), "content", None) or ""

        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        cost = self._calc_cost(input_tokens, output_tokens)

        return GenerationResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
        )
