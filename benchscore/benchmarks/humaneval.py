"""HumanEval Benchmark 实现。

HumanEval — 代码生成评测（164 道 Python 编程题）。
评测方式: 0-shot 代码生成 + 子进程执行测试
"""

from __future__ import annotations

from typing import Any

from benchscore.benchmarks.base import BaseBenchmark
from benchscore.dataset_adapters.base import BenchmarkSample, load_hf_dataset
from benchscore.dataset_adapters.humaneval_adapter import HumanEvalDatasetAdapter
from benchscore.metrics.code_executor import check_humaneval


class HumanEvalBenchmark(BaseBenchmark):
    """HumanEval 代码生成评测"""

    name = "humaneval"
    dimension = "code"
    dataset_id = "openai/openai_humaneval"
    dataset_split = "test"
    trust_remote_code = False         # 新版 datasets 已转为 Parquet 格式

    few_shot = 0
    sample_size = None                # 全量 164 题

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._adapter = HumanEvalDatasetAdapter()

    def load_dataset(self, hf_endpoint: str = "",
                     on_status: callable = None) -> list[BenchmarkSample]:
        """从 HumanEval 加载 test set"""
        raw_test = load_hf_dataset(
            dataset_id=self.dataset_id,
            split=self.dataset_split,
            hf_endpoint=hf_endpoint,
            on_status=on_status,
        )
        samples = [self._adapter.transform(s) for s in raw_test]
        samples = [s for s in samples if s.prompt and s.expected_answer]

        if self.sample_size and self.sample_size < len(samples):
            from benchscore.dataset_adapters.base import sample_dataset
            samples = sample_dataset(samples, self.sample_size, seed=self.seed)

        return samples

    def build_prompt(self, sample: BenchmarkSample) -> str:
        """HumanEval prompt 已经是完整的函数签名 + docstring，直接返回"""
        return sample.prompt

    def score(self, sample: BenchmarkSample, response: str) -> dict[str, Any]:
        """执行代码并检查测试用例"""
        test_code = sample.metadata.get("test_code", "")
        entry_point = sample.metadata.get("entry_point", "")

        passes, message = check_humaneval(
            generated_code=response,
            test_code=test_code,
            entry_point=entry_point,
        )

        return {
            "id": sample.id,
            "prompt": sample.prompt,
            "response": response,
            "expected": sample.expected_answer,
            "correct": passes,
            "message": message,
            "group": "humaneval",
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "latency_ms": 0.0,
        }
