"""数据集预加载窗口 — 启动前检查并下载 HuggingFace 数据集。

在进入主界面之前展示，允许用户预先下载所需数据集，
避免评测过程中因首次下载而卡住。
"""

from __future__ import annotations

import os

# ═══════════════════════════════════════════════════════════════════
# 关键：必须在任何 datasets/huggingface_hub import 之前设置镜像！
# 否则 huggingface_hub 会在首次 import 时读取 HF_ENDPOINT，
# 之后再用 os.environ 覆盖就晚了。
# ═══════════════════════════════════════════════════════════════════
_MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_ENDPOINT"] = _MIRROR

import threading
import time
import tkinter as tk
from tkinter import ttk
from pathlib import Path


# ── 预置 Benchmark 信息 ──────────────────────────────────────────

BENCHMARKS_INFO = [
    {
        "name": "mmlu",
        "label": "MMLU",
        "desc": "知识维度 · 57学科多选题",
        "dataset_id": "cais/mmlu",
        "config": "all",
        "split": "test",
        "size_hint": "~1.2 GB",
    },
    {
        "name": "gsm8k",
        "label": "GSM8K",
        "desc": "推理维度 · 小学数学应用题",
        "dataset_id": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "size_hint": "~8 MB",
    },
    {
        "name": "humaneval",
        "label": "HumanEval",
        "desc": "代码维度 · Python编程题",
        "dataset_id": "openai/openai_humaneval",
        "config": None,
        "split": "test",
        "size_hint": "~1 MB",
    },
]


# ── 缓存检查 ──────────────────────────────────────────────────────

def check_dataset_cached(dataset_id: str, config: str | None = None) -> bool:
    """检查数据集是否已在 HuggingFace 缓存中。

    检查策略：查找 ~/.cache/huggingface/datasets/ 下的对应目录，
    以及 parquet 缓存。
    """
    try:
        cache_root = Path.home() / ".cache" / "huggingface"

        # 检查 datasets cache
        ds_cache = cache_root / "datasets"
        safe_id = dataset_id.replace("/", "___")

        # 方式 1: 已下载的 arrow/parquet 文件
        download_dir = ds_cache / "downloads" / safe_id
        if download_dir.exists():
            has_files = any(
                f.suffix in (".parquet", ".arrow", ".jsonl", ".zip")
                for f in download_dir.iterdir() if f.is_file()
            )
            if has_files:
                return True

        # 方式 2: hub cache (新版 huggingface_hub 格式)
        hub_cache = cache_root / "hub"
        if hub_cache.exists():
            # 搜索包含 dataset_id 的目录
            for d in hub_cache.iterdir():
                if d.is_dir() and dataset_id.replace("/", "--") in d.name:
                    if any(d.iterdir()):
                        return True

        return False
    except Exception:
        return False


# ── Splash 窗口 ───────────────────────────────────────────────────

class SplashScreen(tk.Tk):
    """数据集预加载窗口"""

    def __init__(self):
        super().__init__()

        self.title("BenchScore — 数据集准备")
        self.geometry("700x550")
        self.resizable(False, False)

        # 居中显示
        self.eval("tk::PlaceWindow . center")

        # ── 状态 ──────────────────────────────────────────────────
        self._bench_status: list[dict] = []   # 每个 benchmark 的状态
        self._download_thread: threading.Thread | None = None
        self._downloading = False
        self._should_launch_main = True       # 关闭后是否启动主界面
        self._alive = True                    # 窗口存活标记
        self._all_cached_on_start = False     # 启动时是否全部已缓存

        # 从环境变量取 HF 镜像
        self._hf_endpoint = os.environ.get(
            "HF_ENDPOINT", "https://hf-mirror.com"
        )

        # 窗口关闭协议
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # ── 构建 UI ───────────────────────────────────────────────
        self._build_ui()

        # ── 异步检查缓存 ──────────────────────────────────────────
        self.after(100, self._check_all_cache)

    def _build_ui(self):
        # ── 顶部标题 ──────────────────────────────────────────────
        header = ttk.Frame(self, padding=(20, 15))
        header.pack(fill="x")

        ttk.Label(
            header,
            text="📦 数据集准备",
            font=("", 14, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            header,
            text="以下数据集用于 Benchmark 评测，建议提前下载到本地。\n"
                 "已缓存的数据集无需重复下载。",
            font=("", 9),
            foreground="gray",
        ).pack(anchor="w", pady=(5, 0))

        # ── 数据集状态列表 ────────────────────────────────────────
        list_frame = ttk.LabelFrame(
            self, text="数据集状态", padding=(10, 8),
        )
        list_frame.pack(fill="x", padx=20, pady=(5, 10))

        # 表头
        cols_frame = ttk.Frame(list_frame)
        cols_frame.pack(fill="x")
        for col, w in [("数据集", 18), ("说明", 28), ("状态", 14), ("大小", 10)]:
            ttk.Label(
                cols_frame, text=col, font=("", 9, "bold"), width=w,
            ).pack(side="left", padx=2)

        ttk.Separator(list_frame).pack(fill="x", pady=3)

        # 每行
        self._status_widgets: dict[str, dict] = {}
        for info in BENCHMARKS_INFO:
            row = ttk.Frame(list_frame)
            row.pack(fill="x", pady=3)

            ttk.Label(row, text=info["label"], width=18).pack(side="left", padx=2)
            ttk.Label(
                row, text=info["desc"], width=28,
                foreground="gray", font=("", 8),
            ).pack(side="left", padx=2)

            status_var = tk.StringVar(value="检查中...")
            status_label = ttk.Label(
                row, textvariable=status_var, width=14,
                foreground="orange",
            )
            status_label.pack(side="left", padx=2)

            ttk.Label(
                row, text=info["size_hint"], width=10,
                foreground="gray",
            ).pack(side="left", padx=2)

            self._status_widgets[info["name"]] = {
                "var": status_var,
                "label": status_label,
            }

        # ── 下载进度区域 ──────────────────────────────────────────
        progress_frame = ttk.LabelFrame(
            self, text="下载进度", padding=(10, 8),
        )
        progress_frame.pack(fill="x", padx=20, pady=(0, 10))

        self._progress_var = tk.StringVar(value="就绪")
        ttk.Label(
            progress_frame, textvariable=self._progress_var,
            font=("", 9),
        ).pack(anchor="w")

        self._progress_bar = ttk.Progressbar(
            progress_frame, mode="indeterminate", length=600,
        )
        self._progress_bar.pack(fill="x", pady=(5, 0))

        self._elapsed_var = tk.StringVar(value="")
        ttk.Label(
            progress_frame, textvariable=self._elapsed_var,
            font=("", 8), foreground="gray",
        ).pack(anchor="e")

        # ── 底部按钮 ──────────────────────────────────────────────
        btn_frame = ttk.Frame(self, padding=(20, 10))
        btn_frame.pack(fill="x", side="bottom")

        self._skip_btn = ttk.Button(
            btn_frame, text="⏭ 跳过，直接进入",
            command=self._on_skip,
        )
        self._skip_btn.pack(side="right", padx=(10, 0))

        self._download_all_btn = ttk.Button(
            btn_frame, text="📥 下载未缓存的",
            command=self._on_download_all,
        )
        self._download_all_btn.pack(side="right")

        # 底部状态
        self._bottom_status = tk.StringVar(value="")
        ttk.Label(
            self, textvariable=self._bottom_status,
            font=("", 8), foreground="gray",
        ).pack(side="bottom", pady=(0, 10))

    # ── 缓存检查 ──────────────────────────────────────────────────

    def _check_all_cache(self):
        """检查所有数据集缓存状态"""
        all_cached = True
        for info in BENCHMARKS_INFO:
            name = info["name"]
            cached = check_dataset_cached(info["dataset_id"], info.get("config"))

            if cached:
                self._set_status(name, "✓ 已缓存", "green")
            else:
                self._set_status(name, "⬇ 待下载", "orange")
                all_cached = False

        if all_cached:
            self._progress_var.set("全部已缓存，无需下载 ✨")
            self._download_all_btn.configure(state="disabled")
            self._bottom_status.set("所有数据集已就绪，可以直接进入")
        else:
            self._progress_var.set(
                f"共 {sum(1 for w in self._status_widgets.values()
                         if '待下载' in w['var'].get())} 个数据集待下载"
            )
            self._bottom_status.set("建议下载后再开始评测，避免评测中途等待")

    def _set_status(self, name: str, text: str, color: str):
        """更新某个 benchmark 的状态显示"""
        w = self._status_widgets.get(name)
        if w:
            w["var"].set(text)
            w["label"].configure(foreground=color)

    # ── 下载逻辑 ──────────────────────────────────────────────────

    def _on_download_all(self):
        """开始下载所有未缓存的数据集"""
        if self._downloading:
            return

        # 收集需要下载的
        pending = []
        for info in BENCHMARKS_INFO:
            name = info["name"]
            cached = check_dataset_cached(info["dataset_id"], info.get("config"))
            if not cached:
                pending.append(info)

        if not pending:
            self._progress_var.set("全部已缓存，无需下载")
            return

        self._downloading = True
        self._download_all_btn.configure(state="disabled")
        self._skip_btn.configure(state="disabled")

        # 在后台线程执行下载
        self._download_thread = threading.Thread(
            target=self._download_worker,
            args=(pending,),
            daemon=True,
        )
        self._download_thread.start()

        # 启动进度条动画
        self._progress_bar.start(15)
        self._poll_download()

    def _download_worker(self, pending: list[dict]):
        """后台线程：逐个下载数据集"""
        for i, info in enumerate(pending):
            name = info["name"]

            # 更新状态
            self._report_progress(
                f"正在下载 {info['label']} ({i+1}/{len(pending)})"
                f" — {info['size_hint']}...",
                name,
                f"⬇ 下载中... ({info['size_hint']})",
            )

            try:
                t0 = time.time()
                self._do_download(info)
                elapsed = time.time() - t0

                self._set_status(name, "✓ 已缓存", "green")
                self._report_progress(
                    f"{info['label']} 下载完成！耗时 {elapsed:.0f}s",
                    name,
                    None,
                )
            except Exception as e:
                self._set_status(name, f"✗ 失败", "red")
                self._report_progress(
                    f"{info['label']} 下载失败: {e}",
                    name,
                    None,
                )

        self._report_progress("全部下载完成 ✨", None, None)
        self._progress_bar.stop()
        self._elapsed_var.set("")

    def _do_download(self, info: dict):
        """执行实际的数据集下载"""
        from datasets import load_dataset

        os.environ["HF_ENDPOINT"] = self._hf_endpoint

        load_dataset(
            info["dataset_id"],
            info.get("config"),
            split=info.get("split", "test"),
            streaming=False,
            trust_remote_code=(
                info["dataset_id"] == "openai/openai_humaneval"
            ),
        )

    def _safe_after(self, fn, delay_ms=0):
        """安全的 after() 包装——窗口已销毁时跳过"""
        if self._alive:
            self.after(delay_ms, fn)

    def _report_progress(self, text: str, name: str | None,
                         status_text: str | None):
        """从子线程更新 UI"""
        def _update():
            if not self._alive:
                return
            try:
                if text is not None:
                    self._progress_var.set(text)
                if name and status_text:
                    self._set_status(name, status_text, "blue")
            except Exception:
                pass
        self._safe_after(_update)

    def _poll_download(self):
        """轮询下载线程状态，更新耗时显示"""
        if not self._alive:
            return
        if self._download_thread and self._download_thread.is_alive():
            self._elapsed_var.set("下载中...")
            self.after(500, self._poll_download)
        else:
            self._downloading = False
            self._progress_bar.stop()
            self._skip_btn.configure(state="normal")
            all_cached = all(
                "已缓存" in w["var"].get()
                for w in self._status_widgets.values()
            )
            if all_cached:
                self._download_all_btn.configure(state="disabled")
                self._bottom_status.set("全部就绪，2 秒后自动进入主界面...")
                self.after(2000, self._auto_enter)
            else:
                self._download_all_btn.configure(state="normal")
                self._bottom_status.set("有部分数据集下载失败，可手动重试")

    # ── 跳过 ──────────────────────────────────────────────────────

    def _on_skip(self):
        """跳过预下载，直接进入主界面"""
        if self._downloading:
            return
        self._should_launch_main = True
        self._shutdown()

    def _auto_enter(self):
        """下载全部完成后自动进入主界面"""
        if self._alive:
            self._should_launch_main = True
            self._shutdown()

    def on_close(self):
        """窗口关闭事件"""
        if self._downloading:
            return
        self._should_launch_main = False
        self._shutdown()

    def _shutdown(self):
        """安全关闭窗口"""
        self._alive = False
        try:
            self.destroy()
        except Exception:
            pass


# ── 入口 ──────────────────────────────────────────────────────────

def show_splash() -> bool:
    """显示预加载窗口。

    Returns:
        True: 用户选择进入主界面
        False: 用户关闭窗口
    """
    splash = SplashScreen()
    splash.mainloop()
    return splash._should_launch_main
