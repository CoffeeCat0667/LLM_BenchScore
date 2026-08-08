"""Benchmark 抽象基类。"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from benchscore.dataset_adapters.base import BenchmarkSample, load_hf_dataset
from benchscore.dataset_adapters.base import sample_dataset


@dataclass
class BenchmarkScore:
    """单个 Benchmark 的评测结果"""
    name: str                          # benchmark 名称
    dimension: str                     # 所属维度
    overall: float                     # 整体得分 (0-1)
    num_samples: int                   # 实际评测样本数
    total_tokens: int = 0              # 总 token 消耗
    total_cost_usd: float = 0.0        # 总费用
    total_latency_ms: float = 0.0      # 总延迟
    sub_scores: dict[str, float] = field(default_factory=dict)  # 子维度得分（如 MMLU 按学科）
    details: list[dict[str, Any]] = field(default_factory=list)  # 每题详情


class BaseBenchmark(ABC):
    """所有 Benchmark 的抽象基类。

    每个 Benchmark 子类需实现：
    - load_dataset() — 加载并返回样本列表
    - build_prompt() — 构造发给 LLM 的 prompt
    - score() — 对单条回复评分
    - aggregate() — 聚合为总分

    Pipeline: load_dataset → build_prompt → [LLM generate] → score → aggregate
    """

    name: str = "base"
    dimension: str = "general"
    dataset_id: str = ""
    dataset_config: str | None = None  # e.g. MMLU 可指定学科
    dataset_split: str = "test"
    trust_remote_code: bool = False

    few_shot: int = 0                  # few-shot 示例数
    sample_size: int | None = None     # 采样数量（None=全量）
    weight: float = 1.0               # 维度内权重
    seed: int = 42

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    # ── 生命周期钩子 ──────────────────────────────────────────────

    def on_start(self) -> None:
        """Benchmark 开始前的初始化（子类可覆盖）"""
        pass

    def on_finish(self, score: BenchmarkScore) -> None:
        """Benchmark 完成后的清理（子类可覆盖）"""
        pass

    # ── 核心方法 ───────────────────────────────────────────────────

    def load_dataset(self, hf_endpoint: str = "",
                     on_status: callable = None) -> list[BenchmarkSample]:
        """从 HuggingFace 加载数据集。

        子类可覆盖以自定义加载逻辑。
        默认使用 load_hf_dataset + dataset adapter + 采样。

        Args:
            hf_endpoint: HF 端点
            on_status: 状态回调 (message: str) -> None
        """
        raw_samples = load_hf_dataset(
            dataset_id=self.dataset_id,
            config=self.dataset_config,
            split=self.dataset_split,
            trust_remote_code=self.trust_remote_code,
            hf_endpoint=hf_endpoint,
            on_status=on_status,
        )
        samples = [self._transform_sample(s) for s in raw_samples]
        samples = [s for s in samples if self._is_valid_sample(s)]

        if self.sample_size:
            samples = sample_dataset(
                samples, self.sample_size, seed=self.seed,
                stratify_key=self._stratify_key(),
            )

        return samples

    def _transform_sample(self, raw: dict) -> BenchmarkSample:
        """将原始 HF 数据转为 BenchmarkSample（子类必须覆盖）"""
        raise NotImplementedError("子类需实现 _transform_sample")

    def _is_valid_sample(self, sample: BenchmarkSample) -> bool:
        """验证样本是否有效"""
        return bool(sample.prompt and sample.expected_answer)

    def _stratify_key(self) -> str | None:
        """分层采样的键名（None 表示不分组，纯随机采样）"""
        return None

    @abstractmethod
    def build_prompt(self, sample: BenchmarkSample) -> str:
        """构建发给 LLM 的单题 prompt。

        注意：需要包含 few-shot 示例（如果有的话）。
        子类必须实现。
        """
        ...

    @abstractmethod
    def score(self, sample: BenchmarkSample, response: str) -> dict[str, Any]:
        """对单条 LLM 回复进行评分。

        Args:
            sample: 原始样本（含期望答案）
            response: LLM 生成的回复文本

        Returns:
            评分详情字典，至少包含:
                {"correct": bool, ...}
        """
        ...

    def aggregate(self, scores: list[dict[str, Any]]) -> BenchmarkScore:
        """聚合所有单题得分，计算总分。

        默认实现：统计 correct 比例 + 子维度分组。
        子类可覆盖以自定义聚合逻辑。
        """
        if not scores:
            return BenchmarkScore(
                name=self.name,
                dimension=self.dimension,
                overall=0.0,
                num_samples=0,
            )

        total = len(scores)
        correct = sum(1 for s in scores if s.get("correct", False))
        overall = correct / total

        # Token 和费用统计
        total_tokens = sum(s.get("input_tokens", 0) + s.get("output_tokens", 0) for s in scores)
        total_cost = sum(s.get("cost_usd", 0.0) for s in scores)
        total_latency = sum(s.get("latency_ms", 0.0) for s in scores)

        # 子维度得分（按 metadata 中的 subject/tag 分组）
        sub_scores = self._compute_sub_scores(scores)

        return BenchmarkScore(
            name=self.name,
            dimension=self.dimension,
            overall=round(overall, 4),
            num_samples=total,
            total_tokens=total_tokens,
            total_cost_usd=round(total_cost, 6),
            total_latency_ms=round(total_latency, 2),
            sub_scores=sub_scores,
            details=scores,
        )

    def _compute_sub_scores(self, scores: list[dict]) -> dict[str, float]:
        """按子分组（如学科）计算得分"""
        groups: dict[str, list[dict]] = {}
        for s in scores:
            key = s.get("group", "default")
            groups.setdefault(key, []).append(s)

        result = {}
        for key, group_scores in groups.items():
            correct = sum(1 for s in group_scores if s.get("correct", False))
            result[key] = round(correct / len(group_scores), 4) if group_scores else 0.0
        return result

    # ── 辅助方法 ───────────────────────────────────────────────────

    @staticmethod
    def _get_few_shot_samples(raw_samples: list[dict], n: int, seed: int = 42) -> list[dict]:
        """从训练集中随机选取 few-shot 示例"""
        rng = random.Random(seed)
        return rng.sample(raw_samples, min(n, len(raw_samples)))
