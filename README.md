# BenchScore — 轻量化大模型能力评测工具

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

单机运行、tkinter 轻量 GUI、零外部数据库依赖的大模型评测框架。

## 功能

- **多维度评测**: 知识 (MMLU)、推理 (GSM8K)、代码 (HumanEval)
- **多模型对比**: 支持 OpenAI GPT 系列、Anthropic Claude 系列、兼容 OpenAI 协议模型 (DeepSeek/Qwen/vLLM)
- **轻量 GUI**: tkinter 桌面界面，实时进度、得分图表、历史记录
- **CLI 模式**: 命令行一键跑评测，CI/CD 友好
- **灵活采样**: 全量/采样模式，滑块调节样本数
- **费用预估**: 评测前预估 API 费用，全量模式弹窗确认
- **HuggingFace 镜像**: 内置 hf-mirror.com 支持，国内可用

## 快速开始

### 安装

```bash
git clone <repo-url>
cd LLM_BenchScore
pip install -e .
```

### 启动 GUI

```bash
python -m gui.app
```

### CLI 评测

```bash
# 查看可用 Benchmark
benchscore list

# 运行 MMLU 评测 (采样 1000 题)
benchscore run gpt-4o -b mmlu -k sk-your-api-key

# 运行多个 Benchmark
benchscore run gpt-4o -b mmlu,gsm8k -k sk-xxx -n 500

# 查看历史记录
benchscore history
```

### 配置 API Key

三种方式（优先级从高到低）：

1. **GUI 中直接输入**（会话级，不落盘）
2. **环境变量**: `export BENCHSCORE_OPENAI_API_KEY=sk-xxx`
3. **配置文件**: 将 API Key 填入 `configs/default.yaml`

## 项目结构

```
LLM_BenchScore/
├── benchscore/              # 核心 Python 包
│   ├── adapters/            # LLM 接口适配 (OpenAI/Anthropic/兼容)
│   ├── benchmarks/          # Benchmark 实现 (MMLU/GSM8K/HumanEval)
│   ├── dataset_adapters/    # HuggingFace 数据格式转换
│   ├── metrics/             # 评分函数 (精确匹配/数学/代码执行)
│   ├── report/              # 报告输出 (终端)
│   ├── site_tester.py       # 站点测试引擎 (6项诊断)
│   ├── config.py            # 配置管理
│   ├── runner.py            # 异步评测引擎
│   ├── scorer.py            # 评分聚合
│   ├── store.py             # JSON 结果存储
│   └── cli.py               # CLI 入口
├── gui/                     # tkinter GUI
│   ├── app.py               # 主窗口
│   ├── splash.py            # 数据集预加载窗口
│   ├── model_panel.py       # 模型配置面板
│   ├── benchmark_panel.py   # Benchmark 选择面板
│   ├── run_panel.py         # 运行控制+进度+日志
│   ├── result_panel.py      # 结果展示+图表
│   ├── site_test_panel.py   # 站点测试面板
│   └── dialogs.py           # 弹窗
├── configs/                 # 配置文件
│   ├── default.yaml         # 默认配置
│   └── models.yaml          # 预置模型列表
├── results/                 # JSON 结果存储
└── tests/                   # 测试
```

## 扩展新 Benchmark

1. 创建 `benchscore/benchmarks/your_bench.py`，继承 `BaseBenchmark`
2. 创建对应的 `benchscore/dataset_adapters/your_adapter.py`
3. 在 `benchscore/benchmarks/__init__.py` 注册
4. 在 `configs/default.yaml` 添加默认参数

```python
class YourBenchmark(BaseBenchmark):
    name = "your_bench"
    dimension = "knowledge"
    dataset_id = "your/dataset_id"

    def build_prompt(self, sample) -> str:
        return sample.prompt

    def score(self, sample, response: str) -> dict:
        correct = (response.strip() == sample.expected_answer)
        return {"correct": correct, "id": sample.id, ...}
```

## 依赖

```
httpx, openai, anthropic, datasets, pyyaml, matplotlib
```

tkinter 是 Python 内置模块，无需额外安装。

## License

MIT
