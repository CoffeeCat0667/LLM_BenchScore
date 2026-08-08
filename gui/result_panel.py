"""结果展示面板 — 得分表 + 嵌入 matplotlib 图表。"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class ResultPanel(ttk.LabelFrame):
    """评测结果展示"""

    def __init__(self, parent, app, **kwargs):
        kwargs.setdefault("text", "📊 结果概览")
        kwargs.setdefault("padding", (10, 8))
        super().__init__(parent, **kwargs)
        self.app = app

        self._build_ui()

    def _build_ui(self):
        # ── 结果表格 (Treeview) ────────────────────────────────────
        columns = ("benchmark", "score", "samples", "cost", "tokens")
        self._tree = ttk.Treeview(
            self, columns=columns, show="headings", height=4,
        )
        self._tree.heading("benchmark", text="Benchmark")
        self._tree.heading("score", text="得分")
        self._tree.heading("samples", text="题目数")
        self._tree.heading("cost", text="费用")
        self._tree.heading("tokens", text="Token 消耗")

        self._tree.column("benchmark", width=100, anchor="w")
        self._tree.column("score", width=80, anchor="center")
        self._tree.column("samples", width=60, anchor="center")
        self._tree.column("cost", width=80, anchor="center")
        self._tree.column("tokens", width=120, anchor="center")

        self._tree.pack(fill="x", pady=(0, 5))

        # ── 总分行 ──────────────────────────────────────────────────
        summary_frame = ttk.Frame(self)
        summary_frame.pack(fill="x")

        self._overall_var = tk.StringVar(value="加权总分: —")
        self._time_var = tk.StringVar(value="耗时: —")
        self._cost_var = tk.StringVar(value="总费用: —")

        ttk.Label(
            summary_frame, textvariable=self._overall_var,
            font=("", 10, "bold"),
        ).pack(side="left", padx=(0, 20))
        ttk.Label(
            summary_frame, textvariable=self._time_var,
            font=("", 9),
        ).pack(side="left", padx=(0, 20))
        ttk.Label(
            summary_frame, textvariable=self._cost_var,
            font=("", 9),
        ).pack(side="left")

        # ── 图表区域 ────────────────────────────────────────────────
        self._chart_frame = ttk.Frame(self)
        self._chart_frame.pack(fill="both", expand=True, pady=(5, 0))

        # 占位
        self._chart_placeholder = ttk.Label(
            self._chart_frame,
            text="评测完成后将在此显示雷达图/柱状图",
            foreground="gray",
        )
        self._chart_placeholder.pack(expand=True)

    def show_result(self, result):
        """显示评测结果"""
        # 清空表格
        for item in self._tree.get_children():
            self._tree.delete(item)

        # 填充数据
        for s in result.scores:
            tokens_str = f"{s.total_tokens:,}"
            self._tree.insert("", "end", values=(
                s.name,
                f"{s.overall:.4f}",
                s.num_samples,
                f"${s.total_cost_usd:.4f}",
                tokens_str,
            ))

        # 维度聚合
        for dim, score in result.aggregated.items():
            if dim != "overall":
                self._tree.insert("", "end", values=(
                    f"  {dim}",
                    f"{score:.4f}",
                    "—",
                    "—",
                    "—",
                ))

        # 汇总
        self._overall_var.set(f"加权总分: {result.overall:.4f}")
        self._time_var.set(f"耗时: {result.total_duration_seconds:.1f}s")
        self._cost_var.set(f"总费用: ${result.total_cost_usd:.4f}")

        # 绘制图表
        self._draw_chart(result)

    def _draw_chart(self, result):
        """绘制 matplotlib 柱状图"""
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

            # 清除旧内容
            for w in self._chart_frame.winfo_children():
                w.destroy()

            # 数据
            dims = list(result.aggregated.keys())
            scores = list(result.aggregated.values())

            fig = Figure(figsize=(6, 2), dpi=100)
            ax = fig.add_subplot(111)

            colors = ["#4CAF50" if s >= 0.7 else "#FF9800" if s >= 0.5 else "#F44336"
                      for s in scores]

            bars = ax.bar(dims, scores, color=colors, edgecolor="white", linewidth=0.5)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Score")
            ax.set_title(f"{result.model_id} — Benchmark Results")

            # 数值标注
            for bar, score in zip(bars, scores):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{score:.3f}", ha="center", va="bottom", fontsize=9,
                )

            fig.tight_layout()

            canvas = FigureCanvasTkAgg(fig, master=self._chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

        except Exception as e:
            ttk.Label(
                self._chart_frame,
                text=f"图表渲染失败: {e}",
                foreground="red",
            ).pack(expand=True)
