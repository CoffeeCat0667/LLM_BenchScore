"""GSM8K Benchmark 实现。

GSM8K — 小学数学应用题评测。
评测方式: 5-shot + Chain-of-Thought + 数值答案提取
"""

from __future__ import annotations

from typing import Any

from benchscore.benchmarks.base import BaseBenchmark
from benchscore.dataset_adapters.base import BenchmarkSample, load_hf_dataset
from benchscore.dataset_adapters.gsm8k_adapter import GSM8KDatasetAdapter
from benchscore.metrics.math_grader import gsm8k_grade


GSM8K_FEW_SHOT_EXAMPLES = """以下是将要回答的数学题示例。请逐步推理，最后以 "#### 数字答案" 的格式给出最终答案。

问题: Janet 的鸭子每天下 16 个蛋。她每天早上吃 3 个当早餐，用 4 个做松饼卖给朋友。剩下的她拿到农贸市场去卖，每个 2 美元。她每天在农贸市场能赚多少钱？
Janet 每天卖 16 - 3 - 4 = 9 个鸭蛋。她在农贸市场每天赚 9 × 2 = 18 美元。
#### 18

问题: 一辆车以每小时 60 英里的速度行驶了 2 小时，然后以每小时 40 英里的速度行驶了 1 小时。总共行驶了多少英里？
前 2 小时行驶了 60 × 2 = 120 英里。后 1 小时行驶了 40 × 1 = 40 英里。总共行驶了 120 + 40 = 160 英里。
#### 160

问题: 小明有 5 个苹果，小红给了他 3 个，他又吃了 2 个。现在小明有几个苹果？
小明开始有 5 个苹果，小红给了他 3 个后是 5 + 3 = 8 个。吃了 2 个后剩下 8 - 2 = 6 个。
#### 6

问题: 一个班级有 30 个学生，其中 18 个是女生。男生占总人数的比例是多少？
男生人数 = 30 - 18 = 12。比例 = 12/30 = 2/5 = 0.4 = 40%。
#### 0.4

问题: 一个长方形的长是 12 厘米，宽是 8 厘米。周长是多少厘米？
周长 = 2 × (长 + 宽) = 2 × (12 + 8) = 2 × 20 = 40 厘米。
#### 40"""


class GSM8KBenchmark(BaseBenchmark):
    """GSM8K 数学推理评测"""

    name = "gsm8k"
    dimension = "reasoning"
    dataset_id = "openai/gsm8k"
    dataset_split = "test"
    trust_remote_code = False

    few_shot = 5
    sample_size = 500

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._adapter = GSM8KDatasetAdapter()

    def load_dataset(self, hf_endpoint: str = "",
                     on_status: callable = None) -> list[BenchmarkSample]:
        """从 GSM8K 加载 test set"""
        raw_test = load_hf_dataset(
            dataset_id=self.dataset_id,
            config="main",
            split=self.dataset_split,
            hf_endpoint=hf_endpoint,
            on_status=on_status,
        )
        samples = [self._adapter.transform(s) for s in raw_test]
        samples = [s for s in samples if s.prompt and s.expected_answer]

        if self.sample_size and self.sample_size < len(samples):
            from benchscore.dataset_adapters.base import sample_dataset
            samples = sample_dataset(samples, self.sample_size, seed=self.seed)

        return samples

    def build_prompt(self, sample: BenchmarkSample) -> str:
        """构建带 5-shot CoT 示例的 prompt"""
        return (
            GSM8K_FEW_SHOT_EXAMPLES
            + f"\n\n问题: {sample.prompt}\n"
        )

    def score(self, sample: BenchmarkSample, response: str) -> dict[str, Any]:
        """单题评分：提取数值答案，与期望答案比对"""
        correct = gsm8k_grade(response, str(sample.expected_answer))

        return {
            "id": sample.id,
            "prompt": sample.prompt,
            "response": response,
            "expected": sample.expected_answer,
            "correct": correct,
            "group": "gsm8k",
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "latency_ms": 0.0,
        }
