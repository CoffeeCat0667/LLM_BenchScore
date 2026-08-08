"""MMLU 数据集适配器。

MMLU (Massive Multitask Language Understanding) — 57 个学科的多选题评测。
数据集: cais/mmlu
格式: {question, choices: [A, B, C, D], answer: 0-3, subject}
"""

from __future__ import annotations

import hashlib
from typing import Any

from benchscore.dataset_adapters.base import BaseDatasetAdapter, BenchmarkSample


class MMLUDatasetAdapter(BaseDatasetAdapter):
    """MMLU 数据集格式转换"""

    name = "mmlu"

    def transform(self, raw: dict[str, Any]) -> BenchmarkSample:
        question = str(raw.get("question", "")).strip()
        choices = list(raw.get("choices", []))
        answer_idx = int(raw.get("answer", 0))
        subject = str(raw.get("subject", "unknown"))

        # 构建 prompt：问题 + 选项列表
        labels = ["A", "B", "C", "D", "E", "F", "G", "H"]
        choice_lines = []
        for i, choice_text in enumerate(choices):
            if i < len(labels):
                choice_lines.append(f"{labels[i]}. {choice_text}")

        prompt = f"{question}\n" + "\n".join(choice_lines)

        # 期望答案：字母和文本都存
        expected_letter = labels[answer_idx] if answer_idx < len(labels) else "?"
        expected_text = choices[answer_idx] if answer_idx < len(choices) else ""

        qid = hashlib.md5(question.encode()).hexdigest()[:8]
        return BenchmarkSample(
            id=f"mmlu-{subject}-{qid}",
            prompt=prompt,
            expected_answer=expected_letter,  # 主要比对字母
            metadata={
                "subject": subject,
                "choices": choices,
                "answer_idx": answer_idx,
                "expected_letter": expected_letter,
                "expected_text": expected_text,
                "difficulty": "mixed",  # MMLU 不区分难度
            },
        )
