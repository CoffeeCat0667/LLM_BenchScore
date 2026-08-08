"""MMLU Benchmark 实现。

MMLU (Massive Multitask Language Understanding) — 57 个学科的多选题评测。
评测方式: 5-shot + 多选字母答案 + exact match
"""

from __future__ import annotations

from typing import Any

from benchscore.benchmarks.base import BaseBenchmark
from benchscore.dataset_adapters.base import BenchmarkSample, load_hf_dataset
from benchscore.dataset_adapters.mmlu_adapter import MMLUDatasetAdapter
from benchscore.metrics.exact_match import mmlu_answer_extract


# MMLU 的 few-shot prompt 模板（标准格式）
FEW_SHOT_TEMPLATE = """以下是将要回答的单选题示例：

{examples}

请根据以上示例的格式回答下面的问题，只输出正确选项的字母。"""


class MMLUBenchmark(BaseBenchmark):
    """MMLU 评测"""

    name = "mmlu"
    dimension = "knowledge"
    dataset_id = "cais/mmlu"
    dataset_config = "all"        # 加载所有 57 个学科
    dataset_split = "test"
    trust_remote_code = False

    few_shot = 5
    sample_size = 1000

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._adapter = MMLUDatasetAdapter()
        self._few_shot_text: str = ""  # 缓存 few-shot 示例文本

    # ── 数据集加载 ──────────────────────────────────────────────────

    def load_dataset(self, hf_endpoint: str = "",
                     on_status: callable = None) -> list[BenchmarkSample]:
        """从 MMLU 加载 test set。

        MMLU 结构比较复杂：
        - "all" config → 所有学科合并
        - 每条记录: {question, choices, answer, subject}
        - few-shot 从 "auxiliary_train" split 取
        """
        raw_test = load_hf_dataset(
            dataset_id=self.dataset_id,
            config=self.dataset_config,
            split=self.dataset_split,
            hf_endpoint=hf_endpoint,
            on_status=on_status,
        )
        samples = [self._adapter.transform(s) for s in raw_test]
        samples = [s for s in samples if s.prompt and s.expected_answer]

        # 加载 few-shot 示例（从 auxiliary_train / dev）
        if self.few_shot > 0:
            # MMLU 的 few-shot 是每个学科各取几个，这里简化：从 dev split 随机取
            try:
                raw_dev = load_hf_dataset(
                    dataset_id=self.dataset_id,
                    config=self.dataset_config,
                    split="dev",
                    hf_endpoint=hf_endpoint,
                    on_status=on_status,
                )
                few_shot_samples = self._get_few_shot_samples(
                    raw_dev, self.few_shot, seed=self.seed
                )
                fs_adapted = [self._adapter.transform(s) for s in few_shot_samples]
                self._few_shot_text = self._build_few_shot_examples(fs_adapted)
            except Exception as e:
                # dev split 不可用时，跳过 few-shot
                if on_status:
                    on_status(f"few-shot 加载失败 (将跳过): {e}")
                self._few_shot_text = ""

        # 采样
        if self.sample_size and self.sample_size < len(samples):
            from benchscore.dataset_adapters.base import sample_dataset
            samples = sample_dataset(
                samples, self.sample_size, seed=self.seed, stratify_key="subject"
            )

        return samples

    def _build_few_shot_examples(self, samples: list[BenchmarkSample]) -> str:
        """构建 few-shot 示例文本"""
        lines = []
        for i, s in enumerate(samples, 1):
            choices = s.metadata.get("choices", [])
            labels = ["A", "B", "C", "D", "E", "F", "G", "H"]
            choice_lines = []
            for j, c in enumerate(choices):
                if j < len(labels):
                    choice_lines.append(f"{labels[j]}. {c}")
            answer_letter = s.metadata.get("expected_letter", "?")
            lines.append(
                f"问题 {i}: {s.prompt}\n"
                + "\n".join(choice_lines)
                + f"\n答案: {answer_letter}\n"
            )
        return "\n".join(lines)

    # ── Prompt 构建 ─────────────────────────────────────────────────

    def build_prompt(self, sample: BenchmarkSample) -> str:
        """构建带 few-shot 示例的 prompt"""
        if self._few_shot_text:
            return (
                FEW_SHOT_TEMPLATE.format(examples=self._few_shot_text)
                + f"\n\n{sample.prompt}\n答案:"
            )
        else:
            return f"{sample.prompt}\n答案:"

    # ── 评分 ────────────────────────────────────────────────────────

    def score(self, sample: BenchmarkSample, response: str) -> dict[str, Any]:
        """单题评分：从 LLM 回复中提取字母，与正确答案比对"""
        predicted = mmlu_answer_extract(response)
        expected = sample.metadata.get("expected_letter", sample.expected_answer)

        correct = (predicted is not None and predicted == expected)

        return {
            "id": sample.id,
            "prompt": sample.prompt,
            "response": response,
            "predicted": predicted,
            "expected": expected,
            "correct": correct,
            "group": sample.metadata.get("subject", "unknown"),
            "input_tokens": 0,      # 由 Runner 填充
            "output_tokens": 0,
            "cost_usd": 0.0,
            "latency_ms": 0.0,
        }
