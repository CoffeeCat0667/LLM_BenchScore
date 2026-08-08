"""数据集适配器抽象基类。"""

from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkSample:
    """统一的评测样本格式"""
    id: str                                    # 唯一标识
    prompt: str                                # 用户消息（发给 LLM 的内容）
    expected_answer: str | list[str]           # 期望答案（单答案或多可接受答案）
    metadata: dict[str, Any] = field(default_factory=dict)  # 额外信息（学科、难度、选项等）


class BaseDatasetAdapter(ABC):
    """数据集适配器抽象基类。

    每个 benchmark 的数据集有独特格式，Adapter 负责将其统一转换。
    """

    name: str = "base"

    @abstractmethod
    def transform(self, raw: dict[str, Any]) -> BenchmarkSample:
        """将原始 HF 数据集的一行转换为 BenchmarkSample。

        Args:
            raw: HuggingFace datasets 迭代返回的单条记录

        Returns:
            统一的 BenchmarkSample
        """
        ...

    def is_valid(self, sample: BenchmarkSample) -> bool:
        """验证样本是否可用"""
        return bool(sample.prompt and sample.expected_answer)

    def filter_invalid(self, samples: list[BenchmarkSample]) -> list[BenchmarkSample]:
        """过滤无效样本"""
        return [s for s in samples if self.is_valid(s)]


def set_hf_endpoint(endpoint: str) -> None:
    """设置 HuggingFace 镜像端点（国内用户使用 hf-mirror.com）"""
    os.environ["HF_ENDPOINT"] = endpoint


def load_hf_dataset(
    dataset_id: str,
    config: str | None = None,
    split: str = "test",
    streaming: bool = False,
    trust_remote_code: bool = False,
    hf_endpoint: str | None = None,
    hf_token: str | None = None,
    on_status: callable = None,
) -> list[dict[str, Any]]:
    """从 HuggingFace 加载数据集，返回原始字典列表。

    Args:
        dataset_id: HF 数据集 ID，如 "cais/mmlu"
        config: 数据集配置名，如 MMLU 的 "all" 或具体学科
        split: 数据分割，如 "test", "train", "validation"
        streaming: 是否流式加载（不落盘）
        trust_remote_code: 是否允许执行数据集的自定义代码
        hf_endpoint: HF 端点（默认从配置读取）
        hf_token: HF 访问令牌（用于 gated datasets）
        on_status: 状态回调 (message: str) -> None

    Returns:
        原始字典列表
    """
    from datasets import load_dataset

    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    # 检查缓存 — 判断是否需要下载
    try:
        cached = _check_hf_cache(dataset_id, config, split)
    except Exception:
        cached = False

    if not cached:
        size_hint = _dataset_size_hint(dataset_id)
        if on_status:
            on_status(f"首次使用需下载数据集 {dataset_id}"
                      f"{size_hint}（依赖网速，可能需数分钟）...")
    else:
        if on_status:
            on_status(f"从缓存加载 {dataset_id}...")

    ds = load_dataset(
        dataset_id,
        config,
        split=split,
        streaming=streaming,
        trust_remote_code=trust_remote_code,
    )

    if on_status:
        on_status("正在解析数据...")

    # 转换为列表（streaming=False 时直接可迭代）
    samples = []
    for row in ds:
        # 处理 numpy/pandas 类型，转为原生 Python 类型
        sample = {}
        for k, v in row.items():
            if hasattr(v, "tolist"):
                sample[k] = v.tolist()
            elif hasattr(v, "item"):
                sample[k] = v.item()
            else:
                sample[k] = v
        samples.append(sample)

    return samples


def _check_hf_cache(dataset_id: str, config: str | None,
                    split: str) -> bool:
    """粗略检查数据集是否已在本地缓存中。"""
    from pathlib import Path

    try:
        cache_base = Path.home() / ".cache" / "huggingface" / "datasets"
        if not cache_base.exists():
            return False

        # 检查 download 目录下是否有该数据集的文件
        safe_id = dataset_id.replace("/", "___")
        download_dir = cache_base / "downloads" / safe_id
        if download_dir.exists() and any(download_dir.iterdir()):
            return True

        return False
    except Exception:
        return False


def _dataset_size_hint(dataset_id: str) -> str:
    """返回数据集大小的提示文本。"""
    hints = {
        "cais/mmlu": " (~1.2GB)",
        "openai/gsm8k": " (~8MB)",
        "openai/openai_humaneval": " (~1MB)",
    }
    return hints.get(dataset_id, "")


def sample_dataset(
    samples: list[BenchmarkSample],
    n: int | None,
    seed: int = 42,
    stratify_key: str | None = None,
) -> list[BenchmarkSample]:
    """从样本集中随机采样。

    Args:
        samples: 样本列表
        n: 采样数量，None 返回全量
        seed: 随机种子
        stratify_key: metadata 中的分层键（如 MMLU 的 "subject"），
                      用于保证各层均匀分布

    Returns:
        采样后的样本列表
    """
    if n is None or n >= len(samples):
        return samples

    rng = random.Random(seed)

    if stratify_key:
        # 分层采样：按 stratify_key 分组，每组按比例采样
        groups: dict[str, list[BenchmarkSample]] = {}
        for s in samples:
            key = s.metadata.get(stratify_key, "__unknown__")
            groups.setdefault(key, []).append(s)

        sampled = []
        for group_samples in groups.values():
            group_n = max(1, int(n * len(group_samples) / len(samples)))
            sampled.extend(rng.sample(group_samples, min(group_n, len(group_samples))))

        # 如果采样后超了，再随机裁掉多余的
        if len(sampled) > n:
            sampled = rng.sample(sampled, n)
        return sampled
    else:
        return rng.sample(samples, n)
