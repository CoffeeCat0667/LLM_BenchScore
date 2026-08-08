"""数据集适配层 — 将 HuggingFace 异构格式转换为统一内部格式。"""

from benchscore.dataset_adapters.base import BaseDatasetAdapter, BenchmarkSample
from benchscore.dataset_adapters.mmlu_adapter import MMLUDatasetAdapter
from benchscore.dataset_adapters.gsm8k_adapter import GSM8KDatasetAdapter
from benchscore.dataset_adapters.humaneval_adapter import HumanEvalDatasetAdapter

__all__ = [
    "BaseDatasetAdapter",
    "BenchmarkSample",
    "MMLUDatasetAdapter",
    "GSM8KDatasetAdapter",
    "HumanEvalDatasetAdapter",
]
