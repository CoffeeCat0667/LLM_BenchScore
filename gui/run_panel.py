"""运行控制面板 — 开始/停止按钮、实时进度条、日志输出。"""

from __future__ import annotations

import asyncio
import threading
import tkinter as tk
from tkinter import ttk
from datetime import datetime


class RunPanel(ttk.LabelFrame):
    """运行控制 + 进度 + 日志"""

    def __init__(self, parent, app, **kwargs):
        kwargs.setdefault("text", "📈 进度与控制")
        kwargs.setdefault("padding", (10, 8))
        super().__init__(parent, **kwargs)
        self.app = app
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._runner = None  # 用于取消时通知 Runner
        self._run_id = 0  # 每次运行递增，回调中校验，旧回调忽略

        self._build_ui()

    def _build_ui(self):
        # ── 控制按钮行 ─────────────────────────────────────────────
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(fill="x")

        self._start_btn = ttk.Button(
            ctrl_frame, text="▶ 开始评测", command=self._on_start,
        )
        self._start_btn.pack(side="left")

        self._stop_btn = ttk.Button(
            ctrl_frame, text="⏹ 停止", command=self._on_stop,
            state="disabled",
        )
        self._stop_btn.pack(side="left", padx=(5, 0))

        # ── 进度条区域 ─────────────────────────────────────────────
        self._progress_frame = ttk.Frame(self)
        self._progress_frame.pack(fill="x", pady=(8, 0))

        # 动态创建进度行
        self._progress_bars: dict[str, dict] = {}

        # ── 日志区域 ───────────────────────────────────────────────
        log_label = ttk.Label(self, text="日志:")
        log_label.pack(anchor="w", pady=(8, 2))

        self._log_text = tk.Text(
            self, height=6, wrap="word",
            font=("Consolas", 9),
            state="disabled",
            bg="#1e1e1e", fg="#d4d4d4",
        )
        self._log_text.pack(fill="both", expand=True)

        # 滚动条
        scrollbar = ttk.Scrollbar(self._log_text)
        scrollbar.pack(side="right", fill="y")
        self._log_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.configure(command=self._log_text.yview)

    def _on_start(self):
        """点击开始评测"""
        # 收集配置
        api_format = self.app._model_panel.get_api_format()
        model_id = self.app._model_panel.get_model_id()
        api_key = self.app._model_panel.get_api_key()
        base_url = self.app._model_panel.get_base_url()
        hf_endpoint = self.app._model_panel.get_hf_endpoint()
        benchmarks = self.app._benchmark_panel.get_selected()
        concurrency = self.app._benchmark_panel.get_concurrency()
        temperature = self.app._benchmark_panel.get_temperature()

        # 验证
        if not model_id:
            self.log("ERROR", "请输入 Model ID")
            return
        if not api_key:
            self.log("ERROR", "请填写 API Key")
            return
        if not benchmarks:
            self.log("ERROR", "请至少选择一个 Benchmark")
            return

        # 全量确认
        for b in benchmarks:
            if b["sample_size"] is None:
                from gui.dialogs import show_cost_confirm
                if not show_cost_confirm(self, b["name"], model_id):
                    return

        self.log("INFO", f"开始评测: {model_id} (格式: {api_format})")
        self.log("INFO", f"Benchmarks: {[b['name'] for b in benchmarks]}")

        # 防止重复启动
        if self._running:
            self.log("WARN", "已有评测正在运行，请先停止或等待完成")
            return

        self._run_id += 1
        run_id = self._run_id

        # 异步启动
        self._start_async_task(api_format, model_id, api_key, base_url,
                               hf_endpoint, benchmarks, concurrency, temperature,
                               run_id)

    def _start_async_task(self, api_format, model_id, api_key, base_url,
                          hf_endpoint, benchmarks, concurrency, temperature,
                          run_id: int):
        """在后台线程运行 asyncio 任务"""
        self._running = True
        self._loop = asyncio.new_event_loop()

        async def task():
            from benchscore.adapters import create_adapter
            from benchscore.benchmarks import get_benchmark
            from benchscore.runner import Runner

            try:
                adapter = create_adapter(
                    api_format=api_format,
                    model_id=model_id,
                    api_key=api_key,
                    base_url=base_url,
                )

                self.app.adapter = adapter

                bench_instances = []
                for b in benchmarks:
                    bench = get_benchmark(
                        b["name"],
                        sample_size=b["sample_size"],
                        weight=b["weight"],
                    )
                    bench_instances.append(bench)

                # 创建进度条（捕获 run_id 用于回调过滤）
                self.app.after(0, self._create_progress_bars, bench_instances)
                my_run_id = run_id

                def safe_progress(p):
                    """跳过旧 run 的回调"""
                    if self._run_id == my_run_id:
                        self._on_bench_progress(p)

                def safe_log(level, msg):
                    if self._run_id == my_run_id:
                        self.log(level, msg)

                runner = Runner(
                    adapter=adapter,
                    benchmarks=bench_instances,
                    concurrency=concurrency,
                    hf_endpoint=hf_endpoint,
                    on_bench_progress=lambda p: self.app.after(0, safe_progress, p),
                    on_log=lambda l, m: self.app.after(0, safe_log, l, m),
                )
                self._runner = runner  # 暴露给 _on_stop 取消用

                # 取消检查：如果 run_id 变了，提前退出
                if self._run_id != my_run_id:
                    return

                result = await runner.run()

                # 只有当前 run 的结果才展示
                if self._run_id == my_run_id:
                    self.app.after(0, self.app.on_run_finish, result)

            except Exception as e:
                if self._run_id == my_run_id:
                    self.app.after(0, self.app.on_run_error, str(e))
            finally:
                self._runner = None
                if self._run_id == my_run_id:
                    self._running = False

        def run_in_thread():
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(task())
            finally:
                # 等待 httpx 连接优雅关闭（最多 2 秒）
                try:
                    pending = asyncio.all_tasks(self._loop)
                    for t in pending:
                        t.cancel()
                    if pending:
                        self._loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                except Exception:
                    pass
                try:
                    self._loop.close()
                except Exception:
                    pass

        self._thread = threading.Thread(target=run_in_thread, daemon=True)
        self._thread.start()

        self.app.on_run_start()

    def _on_stop(self):
        """停止评测 — 优雅取消，不暴力关闭事件循环"""
        if not self._running:
            return
        self.log("WARN", "正在停止评测...")
        # 递增 run_id 使所有进行中的回调失效
        self._run_id += 1
        # 通知 Runner 取消（如果在 event loop 线程中）
        if self._runner and self._loop and not self._loop.is_closed():
            try:
                self._loop.call_soon_threadsafe(self._runner.cancel)
            except Exception:
                pass
        self.on_run_finish()

    def _create_progress_bars(self, benchmarks):
        """动态创建每个 benchmark 的进度条"""
        for w in self._progress_frame.winfo_children():
            w.destroy()
        self._progress_bars.clear()

        for bench in benchmarks:
            row_frame = ttk.Frame(self._progress_frame)
            row_frame.pack(fill="x", pady=1)

            label = ttk.Label(row_frame, text=f"{bench.name:10s}", width=12)
            label.pack(side="left")

            bar = ttk.Progressbar(
                row_frame, mode="determinate", length=250,
            )
            bar.pack(side="left", padx=(5, 5))

            score_label = ttk.Label(
                row_frame, text="得分: —", width=12,
            )
            score_label.pack(side="left")

            eta_label = ttk.Label(
                row_frame, text="", width=10,
            )
            eta_label.pack(side="left", padx=(5, 0))

            self._progress_bars[bench.name] = {
                "bar": bar,
                "score": score_label,
                "eta": eta_label,
                "started": datetime.now(),
            }

    def _on_bench_progress(self, progress):
        """更新进度条"""
        info = self._progress_bars.get(progress.bench_name)
        if not info:
            return

        bar = info["bar"]
        score_label = info["score"]
        eta_label = info["eta"]

        bar["maximum"] = progress.total
        bar["value"] = progress.completed

        # 进度回调期间得分不可用（评分在全部调用完成后进行），显示 —
        score_label.configure(text="得分: —")

        # ETA 估算
        elapsed = (datetime.now() - info["started"]).total_seconds()
        if progress.completed > 0:
            rate = elapsed / progress.completed
            remaining = (progress.total - progress.completed) * rate
            eta_label.configure(text=f"ETA: {int(remaining)}s")

    def on_run_start(self):
        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self.app.set_status("评测运行中...")

    def on_run_finish(self):
        self._running = False
        self._start_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self.app.set_status("就绪")

    def log(self, level: str, message: str):
        """添加日志行"""
        now = datetime.now().strftime("%H:%M:%S")
        line = f"{now} [{level}] {message}\n"

        self._log_text.configure(state="normal")
        self._log_text.insert("end", line)
        self._log_text.see("end")
        self._log_text.configure(state="disabled")
