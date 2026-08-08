"""GUI 弹窗 — 费用确认、详情查看、错误提示。"""

import tkinter as tk
from tkinter import messagebox, ttk


def show_cost_confirm(
    parent,
    bench_name: str,
    model_id: str,
    estimated_samples: int = None,
) -> bool:
    """全量评测费用确认弹窗。

    Args:
        parent: 父窗口
        bench_name: benchmark 名称
        model_id: 模型 ID
        estimated_samples: 预估题目数

    Returns:
        用户确认继续返回 True
    """
    # 粗略成本估算
    cost_map = {
        "mmlu": {"input_per_q": 950, "output_per_q": 2},
        "gsm8k": {"input_per_q": 600, "output_per_q": 200},
        "humaneval": {"input_per_q": 150, "output_per_q": 200},
    }
    info = cost_map.get(bench_name, {"input_per_q": 500, "output_per_q": 100})

    # 尝试从配置获取费率
    from benchscore.config import get_config
    config = get_config()
    model = config.get_model(model_id)

    input_cost = model.cost_per_1k_input if model else 0.0025
    output_cost = model.cost_per_1k_output if model else 0.01

    # 估算
    if estimated_samples:
        samples = estimated_samples
    else:
        defaults_map = {"mmlu": 14042, "gsm8k": 1319, "humaneval": 164}
        samples = defaults_map.get(bench_name, 1000)

    est_input_tokens = info["input_per_q"] * samples
    est_output_tokens = info["output_per_q"] * samples
    est_cost = (
        est_input_tokens / 1000 * input_cost
        + est_output_tokens / 1000 * output_cost
    )

    msg = (
        f"即将对 {bench_name} 进行全量评测\n\n"
        f"题目数: {samples:,}\n"
        f"预估费用: ~${est_cost:.2f}\n\n"
        f"是否继续？"
    )

    return messagebox.askyesno("全量评测确认", msg)


def show_result_detail(parent, result) -> None:
    """显示单次评测的详细结果弹窗"""
    dialog = tk.Toplevel(parent)
    dialog.title(f"评测详情 — {result.model_id}")
    dialog.geometry("700x500")
    dialog.transient(parent)

    text = tk.Text(
        dialog, wrap="word",
        font=("Consolas", 10),
    )
    text.pack(fill="both", expand=True, padx=10, pady=10)

    # 填充内容
    lines = []
    lines.append(f"模型: {result.model_id}")
    lines.append(f"时间: {result.timestamp}")
    lines.append(f"总分: {result.overall:.4f}")
    lines.append(f"费用: ${result.total_cost_usd:.6f}")
    lines.append(f"耗时: {result.total_duration_seconds:.1f}s")
    lines.append(f"Token: {result.total_input_tokens:,} in / "
                 f"{result.total_output_tokens:,} out")
    lines.append("=" * 60)

    for s in result.scores:
        lines.append(f"\n[{s.name}] 得分: {s.overall:.4f}  ({s.num_samples} 题)")
        lines.append(f"  费用: ${s.total_cost_usd:.6f}")
        if s.sub_scores:
            for sub, sc in s.sub_scores.items():
                lines.append(f"    {sub}: {sc:.4f}")

    text.insert("1.0", "\n".join(lines))
    text.configure(state="disabled")

    ttk.Button(dialog, text="关闭", command=dialog.destroy).pack(pady=(0, 10))


def show_error(parent, message: str) -> None:
    """错误弹窗"""
    messagebox.showerror("错误", message)
