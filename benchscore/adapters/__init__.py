"""LLM 适配器层 — 统一接口对接不同模型提供商。"""

from benchscore.adapters.base import BaseAdapter, GenerationResult
from benchscore.adapters.openai import OpenAIAdapter
from benchscore.adapters.anthropic import AnthropicAdapter
from benchscore.adapters.openai_compatible import OpenAICompatibleAdapter

__all__ = [
    "BaseAdapter",
    "GenerationResult",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "OpenAICompatibleAdapter",
    "create_adapter",
]


def create_adapter(
    api_format: str,
    model_id: str,
    api_key: str = "",
    base_url: str = "",
) -> BaseAdapter:
    """工厂函数：根据 API 格式创建对应适配器。

    Args:
        api_format: "openai" 或 "anthropic"
        model_id: 模型 ID（手动填写，如 "gpt-4o", "claude-sonnet-4-20250514"）
        api_key: API Key
        base_url: 自定义 API 地址（留空则用官方默认地址）

    Returns:
        对应的适配器实例

    Raises:
        ValueError: 不支持的 API 格式或缺少 API Key
    """
    if not api_key:
        raise ValueError("API Key 不能为空")

    if api_format in ("openai", "openai_compatible"):
        return OpenAIAdapter(
            model_id=model_id,
            api_key=api_key,
            base_url=base_url or None,
        )

    elif api_format == "anthropic":
        kwargs = {"model_id": model_id, "api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        return AnthropicAdapter(**kwargs)

    else:
        raise ValueError(
            f"不支持的 API 格式: {api_format}，可选: openai, anthropic, openai_compatible"
        )
