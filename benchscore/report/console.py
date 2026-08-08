"""终端控制台报告输出。"""

from __future__ import annotations

from benchscore.runner import RunResult


def print_result(result: RunResult) -> None:
    """在终端打印评测结果。

    Args:
        result: RunResult 实例
    """
    print(f"\n{'='*70}")
    print(f"  评测结果 — {result.model_id}")
    print(f"  耗时: {result.total_duration_seconds:.1f}s  |  "
          f"费用: ${result.total_cost_usd:.4f}  |  "
          f"Token: {result.total_input_tokens:,} in / {result.total_output_tokens:,} out")
    print(f"{'='*70}")

    # 各 benchmark 得分
    print(f"\n{'Benchmark':<15s} {'得分':>8s}  {'题目数':>6s}  {'费用':>10s}")
    print(f"{'-'*45}")
    for s in result.scores:
        print(f"{s.name:<15s} {s.overall:>7.4f}  {s.num_samples:>5d}  "
              f"${s.total_cost_usd:>8.4f}")

    # 维度聚合
    print(f"\n{'─'*70}")
    print(f"  维度得分:")
    for dim, score in result.aggregated.items():
        bar = _make_bar(score)
        print(f"    {dim:<15s} {score:.4f}  {bar}")

    # 总分
    print(f"\n  {'★ 加权总分':<15s} {result.overall:.4f}")
    print(f"{'='*70}\n")


def print_compare(results: list[RunResult]) -> None:
    """终端输出多模型对比表。

    Args:
        results: 多个 RunResult 实例
    """
    if not results:
        print("无对比数据")
        return

    # 收集所有维度
    all_dims = set()
    for r in results:
        all_dims.update(r.aggregated.keys())

    dims = sorted(all_dims - {"overall"})

    # 表头
    header = f"{'模型':<20s}"
    for dim in dims:
        header += f" {dim:>10s}"
    header += f" {'总分':>10s}"
    header += f" {'费用':>10s}"

    print(f"\n{'='*len(header)}")
    print(header)
    print(f"{'='*len(header)}")

    for r in results:
        line = f"{r.model_id:<20s}"
        for dim in dims:
            line += f" {r.aggregated.get(dim, 0):>10.4f}"
        line += f" {r.overall:>10.4f}"
        line += f" ${r.total_cost_usd:>9.4f}"
        print(line)

    print(f"{'='*len(header)}\n")


def _make_bar(score: float, width: int = 20) -> str:
    """生成简易进度条"""
    filled = int(score * width)
    empty = width - filled
    return "█" * filled + "░" * empty
