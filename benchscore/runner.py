"""核心 Runner — 异步并发调度，orchestrate 整个评测流水线。"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable

from benchscore.adapters.base import BaseAdapter, BatchProgress
from benchscore.benchmarks.base import BaseBenchmark, BenchmarkScore


@dataclass
class BenchProgress:
    """单个 Benchmark 的进度信息"""
    bench_name: str
    completed: int
    total: int
    current_score: float
    last_latency_ms: float = 0.0


@dataclass
class RunProgress:
    """整体评测进度"""
    current_bench: str
    bench_total: int
    bench_completed: int
    bench_progress: BenchProgress | None = None


@dataclass
class RunResult:
    """一次完整评测的结果"""
    model_id: str
    provider: str
    timestamp: str
    scores: list[BenchmarkScore] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_duration_seconds: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    aggregated: dict[str, float] = field(default_factory=dict)
    overall: float = 0.0  # 加权总分


class Runner:
    """评测运行器 — 负责调度 Benchmark → 并发调用 LLM → 收集评分。

    使用方式:
        runner = Runner(adapter, [mmlu, gsm8k], concurrency=10)
        result = await runner.run()
    """

    def __init__(
        self,
        adapter: BaseAdapter,
        benchmarks: list[BaseBenchmark],
        concurrency: int = 10,
        hf_endpoint: str = "",
        on_bench_progress: Callable[[BenchProgress], None] | None = None,
        on_run_progress: Callable[[RunProgress], None] | None = None,
        on_log: Callable[[str, str], None] | None = None,  # (level, message)
    ):
        self.adapter = adapter
        self.benchmarks = benchmarks
        self.concurrency = concurrency
        self.hf_endpoint = hf_endpoint

        self._on_bench_progress = on_bench_progress
        self._on_run_progress = on_run_progress
        self._on_log = on_log

        self._cancel_flag = asyncio.Event()

    def cancel(self) -> None:
        """取消当前评测"""
        self._cancel_flag.set()
        self._log("WARN", "收到取消请求，正在停止...")

    async def run(self) -> RunResult:
        """执行全部 Benchmark。

        Benchmark 之间串行执行，每题内部并发调用 LLM。
        """
        self._cancel_flag.clear()
        t_start = time.monotonic()

        all_scores: list[BenchmarkScore] = []

        for i, bench in enumerate(self.benchmarks):
            if self._cancel_flag.is_set():
                self._log("INFO", "评测已取消")
                break

            self._log("INFO", f"[{i+1}/{len(self.benchmarks)}] 开始: {bench.name}")

            bench.on_start()
            bench_score = await self._run_single_benchmark(bench)
            bench.on_finish(bench_score)
            all_scores.append(bench_score)

            self._log(
                "INFO",
                f"[{bench.name}] 完成! 得分: {bench_score.overall:.4f}  "
                f"({bench_score.num_samples} 题, ${bench_score.total_cost_usd:.4f})",
            )

        # 聚合
        duration = time.monotonic() - t_start
        result = self._build_result(all_scores, duration)
        return result

    async def _run_single_benchmark(self, bench: BaseBenchmark) -> BenchmarkScore:
        """执行单个 Benchmark 的全部题目"""
        self._log("INFO", f"[{bench.name}] 加载数据集...")
        samples = bench.load_dataset(
            hf_endpoint=self.hf_endpoint,
            on_status=lambda msg: self._log("INFO", f"[{bench.name}] {msg}"),
        )
        self._log("INFO", f"[{bench.name}] 加载完成，共 {len(samples)} 题")

        total = len(samples)
        partial_scores: list[dict] = []

        # 构建 prompts
        prompts = [bench.build_prompt(s) for s in samples]

        def on_item_progress(bp: BatchProgress) -> None:
            """注意：必须是同步函数！generate_batch 不会 await 回调。"""
            completed = bp.completed
            current_score = 0.0
            if partial_scores:
                correct = sum(1 for s in partial_scores if s.get("correct", False))
                current_score = correct / len(partial_scores) if partial_scores else 0.0

            if self._on_bench_progress:
                self._on_bench_progress(BenchProgress(
                    bench_name=bench.name,
                    completed=completed,
                    total=total,
                    current_score=current_score,
                    last_latency_ms=bp.last_latency_ms,
                ))

        # 批量生成
        results = await self.adapter.generate_batch(
            prompts=prompts,
            concurrency=self.concurrency,
            temperature=0.0,
            on_progress=on_item_progress,
        )

        # 统计错误
        error_count = sum(1 for r in results if r.error)
        if error_count > 0:
            # 收集前几个独特的错误信息
            unique_errors = list(dict.fromkeys(
                r.error for r in results if r.error
            ))[:3]
            self._log("ERROR", f"[{bench.name}] {error_count}/{total} 次 API 调用失败！")
            for err in unique_errors:
                self._log("ERROR", f"  错误示例: {err[:200]}")

        # 评分
        for sample, result in zip(samples, results):
            if result.error:
                score_detail = {
                    "id": sample.id,
                    "correct": False,
                    "error": result.error,
                    "group": "error",
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "cost_usd": result.cost_usd,
                    "latency_ms": result.latency_ms,
                }
            else:
                score_detail = bench.score(sample, result.text)
                score_detail["input_tokens"] = result.input_tokens
                score_detail["output_tokens"] = result.output_tokens
                score_detail["cost_usd"] = result.cost_usd
                score_detail["latency_ms"] = result.latency_ms

            partial_scores.append(score_detail)

            if self._cancel_flag.is_set():
                break

        return bench.aggregate(partial_scores)

    def _build_result(self, scores: list[BenchmarkScore], duration: float) -> RunResult:
        """聚合所有 benchmark 分数，生成最终报告"""
        from datetime import datetime, timezone

        total_cost = sum(s.total_cost_usd for s in scores)
        total_input_tokens = sum(
            d.get("input_tokens", 0) for s in scores for d in s.details
        )
        total_output_tokens = sum(
            d.get("output_tokens", 0) for s in scores for d in s.details
        )

        # 按维度聚合
        dim_scores: dict[str, list[float]] = {}
        for s in scores:
            dim_scores.setdefault(s.dimension, []).append(s.overall)

        aggregated = {
            dim: sum(vals) / len(vals) for dim, vals in dim_scores.items()
        }

        # 加权总分
        from benchscore.config import get_config
        config = get_config()
        weights = config.dimension_weights

        overall = 0.0
        total_weight = 0.0
        for dim, score in aggregated.items():
            w = weights.get(dim, 0.2)
            overall += score * w
            total_weight += w

        if total_weight > 0:
            overall /= total_weight

        return RunResult(
            model_id=self.adapter.model_id,
            provider=self.adapter.provider,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            scores=scores,
            total_cost_usd=round(total_cost, 6),
            total_duration_seconds=round(duration, 2),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            aggregated=aggregated,
            overall=round(overall, 4),
        )

    def _log(self, level: str, message: str) -> None:
        if self._on_log:
            self._on_log(level, message)
