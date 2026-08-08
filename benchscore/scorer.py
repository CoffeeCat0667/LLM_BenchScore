"""评分聚合引擎 — 多维度加权总分计算。"""

from __future__ import annotations

from benchscore.benchmarks.base import BenchmarkScore


class Scorer:
    """评分器 — 按维度聚合 benchmark 分数，计算加权总分。

    使用方式:
        scorer = Scorer(weights={"knowledge": 0.3, "reasoning": 0.4, "code": 0.3})
        overall = scorer.aggregate(benchmark_scores)
    """

    def __init__(self, weights: dict[str, float] | None = None):
        """
        Args:
            weights: 维度权重字典，如 {"knowledge": 0.3, "reasoning": 0.4}
                     如果为 None，则使用默认均等权重
        """
        self._weights = weights or {}

    @property
    def weights(self) -> dict[str, float]:
        """当前权重（自动从 config 补全）"""
        if not self._weights:
            from benchscore.config import get_config
            return get_config().dimension_weights
        return self._weights

    def aggregate(self, scores: list[BenchmarkScore]) -> dict[str, float]:
        """按维度聚合分数。

        Returns:
            {dimension: weighted_score, ...} 包含 "overall" 总分
        """
        if not scores:
            return {"overall": 0.0}

        # 按维度分组
        dim_scores: dict[str, list[float]] = {}
        for s in scores:
            dim_scores.setdefault(s.dimension, []).append(s.overall * s.weight)

        # 每维度的加权平均
        aggregated = {}
        for dim, vals in dim_scores.items():
            # 每个维度内部按 weight 加权
            aggregated[dim] = sum(vals) / len(vals) if vals else 0.0

        # 跨维度加权总分
        overall = 0.0
        total_weight = 0.0
        for dim, score in aggregated.items():
            w = self.weights.get(dim, 0.2)
            overall += score * w
            total_weight += w

        aggregated["overall"] = round(overall / total_weight, 4) if total_weight > 0 else 0.0

        return {k: round(v, 4) for k, v in aggregated.items()}
