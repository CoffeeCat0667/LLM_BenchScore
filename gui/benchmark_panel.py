"""Benchmark 选择面板 — 多选、采样量滑块、权重调整。"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class BenchmarkRow:
    """单个 Benchmark 的选择行"""

    def __init__(self, parent, name: str, label: str, max_samples: int,
                 default_samples: int):
        self.name = name
        self.max_samples = max_samples
        self.frame = tk.Frame(parent)

        # 复选框 — 使用 tk.Checkbutton 获得清晰 ✓ 标记
        self._enabled = tk.BooleanVar(value=True)
        self._cb = tk.Checkbutton(
            self.frame, text=f" {label}",
            variable=self._enabled,
            anchor="w",
        )
        self._cb.pack(side="left")

        # 采样数滑块
        self._sample_var = tk.IntVar(value=min(default_samples, max_samples))
        self._sample_label = ttk.Label(
            self.frame, text=f"样本数:", font=("", 8),
        )
        self._sample_label.pack(side="left", padx=(10, 2))
        self._scale = ttk.Scale(
            self.frame, from_=1, to=max_samples,
            variable=self._sample_var,
            orient="horizontal", length=120,
            command=self._on_scale_change,
        )
        self._scale.pack(side="left")
        self._count_label = ttk.Label(
            self.frame,
            text=f"{self._sample_var.get()}/{max_samples}",
            width=10, font=("", 8),
        )
        self._count_label.pack(side="left", padx=(5, 0))

        # 全量按钮
        self._full_btn = ttk.Button(
            self.frame, text="全量", width=4,
            command=self._set_full,
        )
        self._full_btn.pack(side="left", padx=(5, 0))

    def _on_scale_change(self, *args):
        self._count_label.configure(
            text=f"{self._sample_var.get()}/{self.max_samples}"
        )

    def _set_full(self):
        self._sample_var.set(self.max_samples)
        self._count_label.configure(
            text=f"{self.max_samples}/{self.max_samples}"
        )

    @property
    def enabled(self) -> bool:
        return self._enabled.get()

    @property
    def sample_size(self) -> int:
        val = self._sample_var.get()
        return val if val < self.max_samples else None  # None = 全量

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)


class BenchmarkPanel(ttk.LabelFrame):
    """Benchmark 选择区域"""

    def __init__(self, parent, app, **kwargs):
        kwargs.setdefault("text", "📊 Benchmark 选择")
        kwargs.setdefault("padding", (10, 8))
        super().__init__(parent, **kwargs)
        self.app = app
        self._rows: dict[str, BenchmarkRow] = {}

        self._build_ui()

    def _build_ui(self):
        # 使用静态已知信息先构建 UI
        self._build_row("mmlu", "MMLU (知识)", 14042, 1000)
        self._build_row("gsm8k", "GSM8K (推理)", 1319, 500)
        self._build_row("humaneval", "HumanEval (代码)", 164, 164)

        # 温度、并发设置
        settings_frame = ttk.Frame(self)
        settings_frame.pack(fill="x", pady=(10, 0))

        ttk.Label(settings_frame, text="温度:").pack(side="left")
        self._temp_var = tk.DoubleVar(value=0.0)
        ttk.Spinbox(
            settings_frame, from_=0.0, to=2.0, increment=0.1,
            textvariable=self._temp_var, width=5,
        ).pack(side="left", padx=(2, 15))

        ttk.Label(settings_frame, text="并发:").pack(side="left")
        self._concurrency_var = tk.IntVar(value=10)
        ttk.Spinbox(
            settings_frame, from_=1, to=50,
            textvariable=self._concurrency_var, width=5,
        ).pack(side="left", padx=(2, 0))

    def _build_row(self, name: str, label: str, max_samples: int,
                   default_samples: int):
        row = BenchmarkRow(
            self, name, label, max_samples, default_samples,
        )
        row.pack(fill="x", pady=2)
        self._rows[name] = row

    def set_config(self, config):
        """加载配置更新默认采样数"""
        for name, row in self._rows.items():
            defaults = config.get_benchmark_defaults(name)
            if defaults.sample_size is not None and defaults.sample_size < row.max_samples:
                row._sample_var.set(defaults.sample_size)
                row._count_label.configure(
                    text=f"{defaults.sample_size}/{row.max_samples}"
                )

    def get_selected(self) -> list[dict]:
        """获取选中的 benchmark 列表"""
        selected = []
        for name, row in self._rows.items():
            if row.enabled:
                selected.append({
                    "name": name,
                    "sample_size": row.sample_size,
                    "weight": 1.0,
                })
        return selected

    def get_temperature(self) -> float:
        return self._temp_var.get()

    def get_concurrency(self) -> int:
        return self._concurrency_var.get()

    def set_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        for row in self._rows.values():
            pass  # Checkbutton 不受 state 影响，保留为可交互
