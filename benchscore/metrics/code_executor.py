"""代码执行与 HumanEval 评分。

安全警告：代码在子进程中执行，有超时和内存限制。
生产环境中建议使用 Docker 沙箱。
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


# 最大执行时间（秒）
EXEC_TIMEOUT = 10


def execute_code(code: str, timeout: int = EXEC_TIMEOUT) -> tuple[bool, str]:
    """在隔离的 subprocess 中执行 Python 代码。

    Args:
        code: 待执行的 Python 代码
        timeout: 超时秒数

    Returns:
        (success: bool, output: str) — success 为 True 表示正常退出。
        任何异常（包括 assert 失败）都返回 success=False。
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(tmp_path).parent,
        )
        success = result.returncode == 0
        output = result.stdout + result.stderr
        return success, output
    except subprocess.TimeoutExpired:
        return False, f"执行超时 ({timeout}s)"
    except Exception as exc:
        return False, str(exc)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def check_humaneval(
    generated_code: str,
    test_code: str,
    entry_point: str,
    timeout: int = EXEC_TIMEOUT,
) -> tuple[bool, str]:
    """HumanEval 评测：将生成的代码与测试代码拼接后执行。

    Args:
        generated_code: LLM 生成的函数代码
        test_code: HumanEval 的测试用例代码
        entry_point: 函数入口名
        timeout: 执行超时

    Returns:
        (passes_tests: bool, message: str)
    """
    # 清理生成代码：去除 markdown 代码块标记
    code = generated_code.strip()
    code = _strip_markdown_code_block(code)

    # 构建完整可执行代码
    full_code = f"""
{code}

{test_code}

if __name__ == "__main__":
    check({entry_point})
"""
    success, output = execute_code(full_code, timeout=timeout)

    if success:
        return True, "全部测试通过"
    else:
        # 提取关键错误信息
        lines = output.strip().split("\n")
        # 取最后几行作为错误摘要
        key_lines = [l for l in lines if l and not l.startswith("  ")]
        if not key_lines:
            key_lines = lines
        error_msg = "\n".join(key_lines[-5:])
        return False, error_msg


def _strip_markdown_code_block(code: str) -> str:
    """去除 markdown 代码块标记。

    ```python\n...\n``` → ...
    """
    lines = code.strip().split("\n")
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)
