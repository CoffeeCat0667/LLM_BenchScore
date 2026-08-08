"""GUI 包 — tkinter 桌面界面。

启动方式: python -m gui.app
"""

__all__ = ["main"]


def main():
    """GUI 应用入口（延迟导入，避免 python -m 时的模块冲突）"""
    from gui.app import main as _app_main
    _app_main()

