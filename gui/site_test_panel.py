"""站点测试面板 — 6 项 API 诊断测试。"""

from __future__ import annotations

import asyncio
import threading
import tkinter as tk
from tkinter import ttk
from datetime import datetime


class SiteTestPanel(ttk.Frame):
    """站点测试界面"""

    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, **kwargs)
        self.app = app
        self._running = False
        self._run_id = 0

        self._build_ui()

    def _build_ui(self):
        # ── 提示：模型配置在"评测"标签页 ──────────────────────────
        hint_frame = ttk.Frame(self, padding=(5, 5))
        hint_frame.pack(fill="x")
        self._model_hint_var = tk.StringVar(value="请在「评测」标签页配置模型，然后回到此处测试")
        ttk.Label(
            hint_frame, textvariable=self._model_hint_var,
            foreground="gray", font=("", 8),
        ).pack(anchor="w")

        # ── 顶部按钮 ──────────────────────────────────────────────
        top = ttk.Frame(self, padding=(5, 5))
        top.pack(fill="x")

        self._start_btn = ttk.Button(
            top, text="🔍 开始测试", command=self._on_start,
        )
        self._start_btn.pack(side="left")

        self._status_var = tk.StringVar(value="就绪")
        ttk.Label(
            top, textvariable=self._status_var,
            foreground="gray",
        ).pack(side="left", padx=(15, 0))

        # ── 结果表格 ──────────────────────────────────────────────
        table_frame = ttk.Frame(self, padding=(5, 0))
        table_frame.pack(fill="both", expand=True)

        columns = ("item", "result", "detail", "latency")
        self._tree = ttk.Treeview(
            table_frame, columns=columns, show="headings", height=7,
        )
        self._tree.heading("item", text="测试项")
        self._tree.heading("result", text="结果")
        self._tree.heading("detail", text="详情")
        self._tree.heading("latency", text="耗时")

        self._tree.column("item", width=140, anchor="w")
        self._tree.column("result", width=60, anchor="center")
        self._tree.column("detail", width=420, anchor="w")
        self._tree.column("latency", width=80, anchor="center")

        # 预填充行
        items = [
            "站点响应延迟",
            "协议一致性",
            "响应结构",
            "型号特征校验",
            "首字响应时间",
            "隐藏提示词检测",
        ]
        for name in items:
            self._tree.insert("", "end", values=(name, "—", "等待中...", "—"))

        scrollbar = ttk.Scrollbar(table_frame, command=self._tree.yview)
        scrollbar.pack(side="right", fill="y")
        self._tree.configure(yscrollcommand=scrollbar.set)
        self._tree.pack(fill="both", expand=True)

        # ── 底部日志 ──────────────────────────────────────────────
        log_frame = ttk.LabelFrame(self, text="日志", padding=(5, 3))
        log_frame.pack(fill="x", padx=5, pady=(3, 5))

        self._log_text = tk.Text(
            log_frame, height=5, wrap="word",
            font=("Consolas", 9), state="disabled",
            bg="#1e1e1e", fg="#d4d4d4",
        )
        self._log_text.pack(fill="x")

    # ── 开始测试 ──────────────────────────────────────────────────

    def _on_start(self):
        if self._running:
            self.log("WARN", "测试正在进行中")
            return

        api_format = self.app._model_panel.get_api_format()
        model_id = self.app._model_panel.get_model_id()
        api_key = self.app._model_panel.get_api_key()
        base_url = self.app._model_panel.get_base_url()

        if not model_id:
            self.log("ERROR", "请在「评测」标签页输入 Model ID")
            return
        if not api_key:
            self.log("ERROR", "请在「评测」标签页填写 API Key")
            return

        # 显示当前测试目标
        self._model_hint_var.set(
            f"测试目标: {model_id}  |  {api_format}  |  {base_url or '官方地址'}"
        )

        self._running = True
        self._run_id += 1
        run_id = self._run_id
        self._start_btn.configure(state="disabled")
        self._status_var.set("测试中...")
        self._reset_table()

        self.log("INFO", f"开始站点测试: {model_id} @ {base_url or '官方'}")

        thread = threading.Thread(
            target=self._run_tests,
            args=(api_format, model_id, api_key, base_url, run_id),
            daemon=True,
        )
        thread.start()

    def _run_tests(self, api_format, model_id, api_key, base_url, run_id):
        """后台线程执行测试"""
        loop = asyncio.new_event_loop()

        async def task():
            from benchscore.adapters import create_adapter
            from benchscore.site_tester import SiteTester

            try:
                adapter = create_adapter(
                    api_format=api_format,
                    model_id=model_id,
                    api_key=api_key,
                    base_url=base_url,
                )

                tester = SiteTester(adapter)

                def on_item(item):
                    if self._run_id == run_id:
                        self.app.after(0, self._update_item, item, run_id)

                result = await tester.run_all(on_item=on_item)

                if self._run_id == run_id:
                    self.app.after(0, self._on_finish, result, run_id)

            except Exception as e:
                if self._run_id == run_id:
                    self.app.after(0, self._on_error, str(e), run_id)

        loop.run_until_complete(task())

    def _update_item(self, item, run_id):
        """更新表格中的一行"""
        if self._run_id != run_id:
            return

        item_names = [
            "站点响应延迟",
            "协议一致性",
            "响应结构",
            "型号特征校验",
            "首字响应时间",
            "隐藏提示词检测",
        ]

        try:
            idx = item_names.index(item.name)
        except ValueError:
            return

        children = self._tree.get_children()
        if idx < len(children):
            self._tree.item(
                children[idx],
                values=(
                    item.name,
                    item.status,
                    item.detail,
                    f"{item.latency_ms:.0f}ms",
                ),
                tags=(item.status,),
            )

    def _on_finish(self, result, run_id):
        """全部测试完成"""
        if self._run_id != run_id:
            return
        self._running = False
        self._start_btn.configure(state="normal")
        self._status_var.set(
            f"完成 — {result.passed_count}/{result.total_count} 项通过"
        )
        self.log("INFO",
                 f"测试完成: {result.passed_count}/{result.total_count} 通过")

    def _on_error(self, error_msg, run_id):
        """测试出错"""
        if self._run_id != run_id:
            return
        self._running = False
        self._start_btn.configure(state="normal")
        self._status_var.set("测试失败")
        self.log("ERROR", error_msg)

    def _reset_table(self):
        """重置表格为初始状态"""
        items = [
            "站点响应延迟",
            "协议一致性",
            "响应结构",
            "型号特征校验",
            "首字响应时间",
            "隐藏提示词检测",
        ]
        for i, name in enumerate(items):
            children = self._tree.get_children()
            if i < len(children):
                self._tree.item(
                    children[i],
                    values=(name, "⏳", "测试中...", "—"),
                    tags=(),
                )

    def log(self, level: str, message: str):
        now = datetime.now().strftime("%H:%M:%S")
        line = f"{now} [{level}] {message}\n"
        self._log_text.configure(state="normal")
        self._log_text.insert("end", line)
        self._log_text.see("end")
        self._log_text.configure(state="disabled")
