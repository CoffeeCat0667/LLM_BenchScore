"""Benchmark 集合 — 各维度评测实现。"""

from benchscore.benchmarks.base import BaseBenchmark, BenchmarkScore
from benchscore.benchmarks.mmlu import MMLUBenchmark
from benchscore.benchmarks.gsm8k import GSM8KBenchmark
from benchscore.benchmarks.humaneval import HumanEvalBenchmark

__all__ = [
    "BaseBenchmark",
    "BenchmarkScore",
    "MMLUBenchmark",
    "GSM8KBenchmark",
    "HumanEvalBenchmark",
    "get_benchmark",
    "list_benchmarks",
]

# 注册所有 benchmark
_BENCHMARK_REGISTRY: dict[str, type[BaseBenchmark]] = {
    "mmlu": MMLUBenchmark,
    "gsm8k": GSM8KBenchmark,
    "humaneval": HumanEvalBenchmark,
}


def get_benchmark(name: str, **kwargs) -> BaseBenchmark:
    """根据名称创建 benchmark 实例。

    Args:
        name: benchmark 名称，如 "mmlu"
        **kwargs: 传递给 benchmark 构造函数的参数

    Returns:
        BaseBenchmark 实例

    Raises:
        ValueError: 未知 benchmark
    """
    cls = _BENCHMARK_REGISTRY.get(name.lower())
    if cls is None:
        raise ValueError(
            f"未知 benchmark: {name}，可用: {list(_BENCHMARK_REGISTRY.keys())}"
        )
    return cls(**kwargs)


def list_benchmarks() -> list[str]:
    """列出所有已注册的 benchmark 名称"""
    return list(_BENCHMARK_REGISTRY.keys())
