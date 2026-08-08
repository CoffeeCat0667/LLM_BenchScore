"""JSON 文件存储 — 轻量级结果持久化。

存储路径: results/{model_id}/{timestamp}.json

"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ResultStore:
    """评测结果 JSON 文件存储。

    使用方式:
        store = ResultStore("results")
        store.save(result)
        runs = store.list_runs()
    """

    def __init__(self, base_dir: str = "results"):
        self.base_dir = Path(base_dir)

    def save(self, result: Any) -> Path:
        """保存一次评测结果。

        Args:
            result: RunResult 实例

        Returns:
            保存的文件路径
        """
        # 使用时间戳作为文件名
        ts = result.timestamp.replace(":", "-").replace("T", "_")
        model_dir = self.base_dir / result.model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        path = model_dir / f"{ts}.json"

        data = _to_dict(result)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return path

    def list_runs(self, model_id: str | None = None) -> list[dict[str, Any]]:
        """列出历史评测结果摘要。

        Args:
            model_id: 模型 ID 过滤，None 返回所有模型的汇总

        Returns:
            按时间倒序排列的结果摘要列表
        """
        runs = []

        if model_id:
            dirs = [self.base_dir / model_id]
        else:
            dirs = sorted(self.base_dir.glob("*")) if self.base_dir.exists() else []

        for model_dir in dirs:
            if not model_dir.is_dir():
                continue
            for json_file in sorted(model_dir.glob("*.json"), reverse=True):
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                    runs.append({
                        "file": str(json_file),
                        "model_id": data.get("model_id", ""),
                        "provider": data.get("provider", ""),
                        "timestamp": data.get("timestamp", ""),
                        "overall": data.get("overall", 0.0),
                        "total_cost_usd": data.get("total_cost_usd", 0.0),
                        "total_duration_seconds": data.get("total_duration_seconds", 0.0),
                        "benchmarks": [
                            s.get("name", "") for s in data.get("scores", [])
                        ],
                        "aggregated": data.get("aggregated", {}),
                    })
                except json.JSONDecodeError:
                    continue

        return runs

    def load(self, path: str | Path) -> dict[str, Any]:
        """加载单个结果文件。

        Returns:
            完整的评测结果字典
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"结果文件不存在: {p}")
        return json.loads(p.read_text(encoding="utf-8"))

    def delete(self, path: str | Path) -> None:
        """删除单个结果文件"""
        p = Path(path)
        p.unlink(missing_ok=True)
        # 如果目录为空，也清理掉
        parent = p.parent
        if parent.is_dir() and not list(parent.iterdir()):
            try:
                parent.rmdir()
            except OSError:
                pass


def _to_dict(obj: Any) -> Any:
    """递归将 dataclass 转换为普通字典"""
    if hasattr(obj, "__dataclass_fields__"):
        result = {}
        for f in obj.__dataclass_fields__:
            value = getattr(obj, f)
            result[f] = _to_dict(value)
        return result
    elif isinstance(obj, list):
        return [_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj
