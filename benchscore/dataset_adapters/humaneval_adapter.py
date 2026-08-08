"""HumanEval 数据集适配器。

HumanEval — 代码生成评测（164 道 Python 编程题）。
数据集: openai/openai_humaneval
格式: {task_id, prompt, canonical_solution, test, entry_point}
"""

from __future__ import annotations

from typing import Any

from benchscore.dataset_adapters.base import BaseDatasetAdapter, BenchmarkSample


class HumanEvalDatasetAdapter(BaseDatasetAdapter):
    """HumanEval 数据集格式转换"""

    name = "humaneval"

    def transform(self, raw: dict[str, Any]) -> BenchmarkSample:
        task_id = str(raw.get("task_id", ""))
        prompt = str(raw.get("prompt", "")).strip()
        canonical_solution = str(raw.get("canonical_solution", ""))
        test_code = str(raw.get("test", ""))
        entry_point = str(raw.get("entry_point", ""))

        # HumanEval prompt 已经是完整的函数签名 + docstring
        # 需要在末尾添加代码补全
        full_prompt = prompt + "\n"

        return BenchmarkSample(
            id=f"humaneval-{task_id}",
            prompt=full_prompt,
            expected_answer=canonical_solution,
            metadata={
                "task_id": task_id,
                "test_code": test_code,
                "entry_point": entry_point,
                "canonical_solution": canonical_solution,
                "difficulty": "mixed",
            },
        )
