"""GSM8K 数学答案提取与评分。"""

import re


def gsm8k_answer_extract(response: str) -> str | None:
    """从 LLM 回复中提取最终数值答案。

    支持常见输出格式：
        "答案是 72"         → "72"
        "#### 72"           → "72"
        "所以答案是 1,000 元" → "1000"
        "The answer is 42"  → "42"
        "= 3.14"             → "3.14"

    Returns:
        提取到的数值字符串，失败返回 None
    """
    text = response.strip()

    # 策略 1: #### 标记（GSM8K 提示格式）
    m = re.search(r"####\s*(.+)", text)
    if m:
        val = _clean_number(m.group(1))
        if val:
            return val

    # 策略 2: "答案是 X" / "answer is X"
    patterns = [
        r"(?:答案(?:是|为|：|:)|answer\s*(?:is|:)?)\s*(.+?)(?:\n|$)",
        r"(?:所以|因此|最终|结果)(?:是|为|：|:)?\s*(.+?)(?:\n|$)",
        r"=\s*([\d,]+\.?\d*)\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = _clean_number(m.group(1))
            if val:
                return val

    # 策略 3: 最后一行中的数字
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in reversed(lines):
        val = _clean_number(line)
        if val:
            return val

    # 策略 4: 最后一个数字
    nums = re.findall(r"[\d,]+\.?\d*", text)
    if nums:
        return nums[-1].replace(",", "")

    return None


def _clean_number(text: str) -> str | None:
    """从文本中提取干净的数值字符串。"""
    # 移除逗号、中文单位等
    text = text.replace(",", "").replace("，", "")
    text = text.replace("元", "").replace("个", "").replace("只", "")
    text = text.replace("$", "").replace("€", "").replace("¥", "")
    text = text.strip()

    # 匹配数字（整数或小数）
    m = re.search(r"(\d+\.?\d*)", text)
    if m:
        num_str = m.group(1)
        # 去除末尾无意义的小数点
        num_str = num_str.rstrip(".")
        return num_str
    return None


def gsm8k_grade(prediction: str, expected: str) -> bool:
    """GSM8K 评分：提取数值后比对。

    支持数值近似比对（浮点数误差容忍）。
    """
    pred_num = gsm8k_answer_extract(prediction)
    if pred_num is None:
        return False

    # 尝试数值比对
    try:
        pred_val = float(pred_num)
        exp_val = float(expected.replace(",", ""))
        # 容忍 1e-6 的浮点误差（或整数严格相等）
        return abs(pred_val - exp_val) < 1e-6
    except (ValueError, TypeError):
        pass

    # 字符串比对（fallback）
    return pred_num.strip() == expected.strip().replace(",", "")
