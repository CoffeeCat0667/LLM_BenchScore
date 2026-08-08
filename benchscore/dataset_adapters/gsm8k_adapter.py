"""GSM8K 数据集适配器。

GSM8K — 小学数学应用题（Grade School Math 8K）。
数据集: openai/gsm8k
格式: {question, answer: "#### 72" → 最终数字答案在 #### 后}
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from benchscore.dataset_adapters.base import BaseDatasetAdapter, BenchmarkSample


class GSM8KDatasetAdapter(BaseDatasetAdapter):
    """GSM8K 数据集格式转换"""

    name = "gsm8k"

    def transform(self, raw: dict[str, Any]) -> BenchmarkSample:
        question = str(raw.get("question", "")).strip()
        raw_answer = str(raw.get("answer", "")).strip()

        # GSM8K 答案格式：逐步推导过程 ... #### 最终数字答案
        # 提取 #### 后面的数字
        final_number = self._extract_final_answer(raw_answer)

        qid = hashlib.md5(question.encode()).hexdigest()[:8]
        return BenchmarkSample(
            id=f"gsm8k-{qid}",
            prompt=question,
            expected_answer=final_number,
            metadata={
                "full_answer": raw_answer,
                "difficulty": "grade-school",
            },
        )

    @staticmethod
    def _extract_final_answer(answer_text: str) -> str:
        """从 GSM8K 答案中提取最终数字。

        GSM8K 格式: "Janet sells 16 - 3 - 4 = 9 duck eggs a day. She makes
        9 * 2 = $18 every day at the farmer's market. #### 18"
        """
        # 匹配 #### 后面的内容
        match = re.search(r"####\s*(.+)", answer_text)
        if match:
            val = match.group(1).strip()
            # 去除逗号（如 1,000 → 1000）
            val = val.replace(",", "")
            return val

        # 兜底：尝试取最后一行
        lines = answer_text.strip().split("\n")
        last_line = lines[-1].strip()
        nums = re.findall(r"[\d,]+\.?\d*", last_line)
        if nums:
            return nums[-1].replace(",", "")

        return answer_text.strip()
