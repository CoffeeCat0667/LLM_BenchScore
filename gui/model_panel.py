"""模型配置面板 — API 格式选择、API Key、Base URL、Model ID 输入。"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class ModelPanel(ttk.LabelFrame):
    """模型配置区域"""

    def __init__(self, parent, app, **kwargs):
        kwargs.setdefault("text", "📋 模型配置")
        kwargs.setdefault("padding", (10, 8))
        super().__init__(parent, **kwargs)
        self.app = app
        self._api_format_var = tk.StringVar(value="openai")
        self._api_key_var = tk.StringVar()
        self._base_url_var = tk.StringVar()
        self._model_id_var = tk.StringVar()
        self._hf_endpoint_var = tk.StringVar(value="https://hf-mirror.com")
        self._show_key = tk.BooleanVar(value=False)

        self._build_ui()

    def _build_ui(self):
        # Row 0: API 格式
        ttk.Label(self, text="API 格式:").grid(row=0, column=0, sticky="w", pady=2)
        self._format_combo = ttk.Combobox(
            self, textvariable=self._api_format_var,
            values=["openai", "anthropic"],
            state="readonly", width=20,
        )
        self._format_combo.grid(row=0, column=1, sticky="ew", pady=2, padx=(5, 0))

        # Row 1: API Key
        ttk.Label(self, text="API Key:").grid(row=1, column=0, sticky="w", pady=2)
        key_frame = ttk.Frame(self)
        key_frame.grid(row=1, column=1, sticky="ew", pady=2, padx=(5, 0))
        self._key_entry = ttk.Entry(
            key_frame, textvariable=self._api_key_var,
            show="*", width=22,
        )
        self._key_entry.pack(side="left", fill="x", expand=True)
        self._toggle_btn = ttk.Button(
            key_frame, text="👁", width=3,
            command=self._toggle_key_visibility,
        )
        self._toggle_btn.pack(side="left", padx=(3, 0))

        # Row 2: Base URL（始终显示，留空则用官方默认）
        ttk.Label(self, text="Base URL:").grid(row=2, column=0, sticky="w", pady=2)
        self._base_url_entry = ttk.Entry(
            self, textvariable=self._base_url_var, width=22,
        )
        self._base_url_entry.grid(row=2, column=1, sticky="ew", pady=2, padx=(5, 0))
        note = ttk.Label(
            self, text="  需包含 /v1, 留空=使用 OpenAI 官方",
            font=("", 7), foreground="gray",
        )
        note.grid(row=3, column=1, sticky="w", pady=(0, 2), padx=(5, 0))

        # Row 4: Model ID（手动填写）
        ttk.Label(self, text="Model ID:").grid(row=4, column=0, sticky="w", pady=2)
        self._model_entry = ttk.Entry(
            self, textvariable=self._model_id_var, width=22,
        )
        self._model_entry.grid(row=4, column=1, sticky="ew", pady=2, padx=(5, 0))
        hint = ttk.Label(
            self, text="  e.g. gpt-4o / claude-sonnet-4-20250514",
            font=("", 7), foreground="gray",
        )
        hint.grid(row=5, column=1, sticky="w", pady=(0, 2), padx=(5, 0))

        # Row 6: HF 镜像
        ttk.Label(self, text="HF 镜像:").grid(row=6, column=0, sticky="w", pady=2)
        ttk.Entry(
            self, textvariable=self._hf_endpoint_var, width=22,
        ).grid(row=6, column=1, sticky="ew", pady=2, padx=(5, 0))

        self.columnconfigure(1, weight=1)

    def set_config(self, config):
        """加载配置：从环境变量预填 API Key"""
        if config.openai_api_key:
            if not self._api_key_var.get():
                self._api_key_var.set(config.openai_api_key)
        elif config.anthropic_api_key:
            if not self._api_key_var.get():
                self._api_key_var.set(config.anthropic_api_key)
        if config.hf_endpoint:
            self._hf_endpoint_var.set(config.hf_endpoint)

    # ── 公共 getter ────────────────────────────────────────────────

    def get_api_format(self) -> str:
        """返回 "openai" 或 "anthropic" """
        return self._api_format_var.get()

    def get_model_id(self) -> str:
        return self._model_id_var.get().strip()

    def get_api_key(self) -> str:
        return self._api_key_var.get().strip()

    def get_base_url(self) -> str:
        return self._base_url_var.get().strip()

    def get_hf_endpoint(self) -> str:
        return self._hf_endpoint_var.get().strip()

    # ── 状态控制 ───────────────────────────────────────────────────

    def set_enabled(self, enabled: bool):
        """启用/禁用面板（评测中禁用）"""
        state = "normal" if enabled else "disabled"
        self._format_combo.configure(state="readonly" if enabled else "disabled")
        self._key_entry.configure(state=state)
        self._base_url_entry.configure(state=state)
        self._model_entry.configure(state=state)

    def _toggle_key_visibility(self):
        """切换 API Key 显示/隐藏"""
        if self._show_key.get():
            self._key_entry.configure(show="*")
            self._show_key.set(False)
        else:
            self._key_entry.configure(show="")
            self._show_key.set(True)
