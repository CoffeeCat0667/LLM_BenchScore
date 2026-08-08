"""GUI 主窗口 — tkinter 应用入口。

启动方式:
    python -m gui.app
    或
    from gui import main; main()
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from gui.model_panel import ModelPanel
from gui.benchmark_panel import BenchmarkPanel
from gui.run_panel import RunPanel
from gui.result_panel import ResultPanel
from gui.site_test_panel import SiteTestPanel



class App(tk.Tk):
    """BenchScore 主应用窗口"""

    def __init__(self):
        super().__init__()

        self.title("LLM BenchScore — 大模型能力评测工具 v1.0.0")
        self.geometry("960x780")
        self.minsize(800, 650)

        # 设置 ttk 主题
        style = ttk.Style()
        available = style.theme_names()
        if "clam" in available:
            style.theme_use("clam")

        # ── 全局状态 ──────────────────────────────────────────────
        self._config = None          # Config 实例
        self._adapter = None         # 当前适配器

        # ── 构建 UI ───────────────────────────────────────────────
        self._build_ui()

        # ── 窗口关闭 ──────────────────────────────────────────────
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── 加载配置 ──────────────────────────────────────────────
        self.after(100, self._init_config)

    def _build_ui(self):
        """构建界面布局"""
        # ── 顶部 Notebook（标签页切换） ────────────────────────────
        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Tab 1: 评测
        eval_frame = ttk.Frame(self._notebook)
        self._notebook.add(eval_frame, text="  评测  ")

        # 上半部分：配置区（左右分栏）
        config_frame = ttk.Frame(eval_frame)
        config_frame.pack(fill="x", padx=5, pady=(5, 0))

        # 左侧：模型配置
        self._model_panel = ModelPanel(config_frame, self)
        self._model_panel.pack(side="left", fill="both", expand=True, padx=(0, 3))

        # 右侧：Benchmark 选择
        self._benchmark_panel = BenchmarkPanel(config_frame, self)
        self._benchmark_panel.pack(side="right", fill="both", expand=True, padx=(3, 0))

        # 中部：运行控制 + 进度 + 日志
        self._run_panel = RunPanel(eval_frame, self)
        self._run_panel.pack(fill="both", expand=True, padx=5, pady=5)

        # 底部：结果展示
        self._result_panel = ResultPanel(eval_frame, self)
        self._result_panel.pack(fill="both", expand=True, padx=5, pady=(0, 5))

        # Tab 2: 站点测试
        site_test_frame = ttk.Frame(self._notebook)
        self._notebook.add(site_test_frame, text="  站点测试  ")
        self._site_test_panel = SiteTestPanel(site_test_frame, self)
        self._site_test_panel.pack(fill="both", expand=True, padx=5, pady=5)

        # ── 状态栏 ───────────────────────────────────────────────
        self._status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(
            self, textvariable=self._status_var,
            relief="sunken", anchor="w", padding=(10, 3),
        )
        status_bar.pack(fill="x", side="bottom")

    def _init_config(self):
        """初始化配置"""
        try:
            from benchscore.config import load_config, set_config
            config = load_config()
            set_config(config)
            self._config = config
            self._model_panel.set_config(config)
            self._benchmark_panel.set_config(config)
            self.set_status("就绪 — 请配置模型和 API Key")
        except Exception as e:
            self.set_status(f"配置加载失败: {e}")

    # ── 公共属性 ──────────────────────────────────────────────────

    @property
    def config(self):
        return self._config

    @property
    def adapter(self):
        return self._adapter

    @adapter.setter
    def adapter(self, value):
        self._adapter = value

    # ── 公共方法 ──────────────────────────────────────────────────

    def _on_close(self):
        """窗口关闭时优雅清理"""
        # 取消正在运行的任务
        if hasattr(self, '_run_panel') and self._run_panel._running:
            self._run_panel._on_stop()
        self.destroy()

    def set_status(self, text: str):
        self._status_var.set(text)

    def log(self, level: str, message: str):
        self._run_panel.log(level, message)

    def on_run_start(self):
        """开始评测时的 UI 状态"""
        self._run_panel.on_run_start()
        self._model_panel.set_enabled(False)
        self._benchmark_panel.set_enabled(False)

    def on_run_finish(self, result):
        """评测完成时的 UI 状态"""
        self._run_panel.on_run_finish()
        self._model_panel.set_enabled(True)
        self._benchmark_panel.set_enabled(True)
        self._result_panel.show_result(result)

    def on_run_error(self, error_msg: str):
        """评测出错时的 UI 状态"""
        self._run_panel.on_run_finish()
        self._model_panel.set_enabled(True)
        self._benchmark_panel.set_enabled(True)
        self.log("ERROR", error_msg)
        self.set_status(f"错误: {error_msg}")


def main():
    """GUI 应用入口 — 先显示预加载窗口，再启动主界面。"""
    from gui.splash import show_splash

    should_launch = show_splash()
    if not should_launch:
        return  # 用户关闭了预加载窗口

    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
