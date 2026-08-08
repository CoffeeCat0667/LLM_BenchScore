"""精确匹配 和 MMLU 答案提取。"""

import re


def exact_match(prediction: str, expected: str | list[str]) -> bool:
    """精确匹配 — 去除首尾空白后逐字符比对。

    支持多可接受答案：expected 为 list 时，匹配任一即算对。
    """
    pred_clean = prediction.strip()

    if isinstance(expected, list):
        return any(pred_clean == e.strip() for e in expected)

    return pred_clean == expected.strip()


def mmlu_answer_extract(response: str) -> str | None:
    """从 LLM 回复中提取 MMLU 的 A/B/C/D 答案。

    支持多种常见输出格式：
        "A"           → "A"
        "B."          → "B"
        "(C)"         → "C"
        "The answer is D" → "D"
        "答案是 A"      → "A"
        "Answer: B"    → "B"

    Returns:
        提取到的字母 (A-H)，失败返回 None
    """
    text = response.strip()

    # 策略 1: 整个回复只有一个字母（最常见格式）
    cleaned = text.strip().rstrip(".").rstrip(")").rstrip("）")
    if len(cleaned) == 1 and cleaned.upper() in "ABCDEFGH":
        return cleaned.upper()

    # 策略 2: "答案是 X" / "answer is X" / "Answer: X"
    patterns = [
        r"(?:答案(?:是|为|：|:)|answer\s*(?:is|:)?)\s*\(?([A-H])\)?",
        r"(?:选|选择)\s*([A-H])",
        r"\(([A-H])\)",
        r"([A-H])[\.\)）]",  # "A." "A)" "A）"
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # 策略 3: 找到最后一个单独的选项字母
    m = re.search(r"\b([A-H])\b\s*$", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return None
