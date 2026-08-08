"""OpenAI 兼容适配器 — 用于 vLLM、Ollama、DeepSeek 等兼容 OpenAI API 的模型。"""

from __future__ import annotations

from benchscore.adapters.openai import OpenAIAdapter
from benchscore.config import ModelInfo


class OpenAICompatibleAdapter(OpenAIAdapter):
    """兼容 OpenAI 协议的第三方模型适配器。

    使用方式：
        adapter = OpenAICompatibleAdapter(
            model_id="deepseek-v3",
            api_key="your-key",
            base_url="https://api.deepseek.com/v1",
        )
    """

    provider = "openai_compatible"

    def __init__(
        self,
        model_id: str = "qwen3",
        api_key: str = "not-needed",
        base_url: str | None = None,
        model_info: ModelInfo | None = None,
    ):
        super().__init__(
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            model_info=model_info,
        )
