"""CLI 命令行入口 — benchscore 命令。

使用方式:
    benchscore run gpt-4o --benchmarks mmlu,gsm8k --api-key sk-xxx
    benchscore list
    benchscore history [model_id]

CLI 是轻量模式入口，GUI 通过 gui/app.py 启动。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

from benchscore import __version__


@click.group()
@click.version_option(__version__, prog_name="benchscore")
def main():
    """BenchScore — 轻量化大模型能力评测工具

    支持 MMLU、GSM8K、HumanEval 等多维度 Benchmark。
    """
    pass


@main.command()
@click.argument("model_id")
@click.option("--benchmarks", "-b", default="mmlu",
              help="评测的 Benchmark，逗号分隔 (默认: mmlu)。可用: mmlu,gsm8k,humaneval")
@click.option("--api-key", "-k", default="", help="LLM API Key")
@click.option("--base-url", default="", help="API Base URL (兼容接口时使用)")
@click.option("--sample-size", "-n", type=int, default=None,
              help="每题采样数 (覆盖配置文件默认值)")
@click.option("--concurrency", "-c", type=int, default=10,
              help="并发数 (默认 10)")
@click.option("--output", "-o", default="", help="结果输出目录 (默认 results/)")
def run(model_id: str, benchmarks: str, api_key: str, base_url: str,
        sample_size: int | None, concurrency: int, output: str):
    """运行评测。

    \b
    示例:
        benchscore run gpt-4o -b mmlu -k sk-xxx
        benchscore run gpt-4o -b mmlu,gsm8k -n 500 -c 20
        benchscore run deepseek-v3 -b humaneval -k xxx --base-url https://api.deepseek.com/v1
    """
    from benchscore.config import load_config, set_config
    from benchscore.adapters import create_adapter
    from benchscore.benchmarks import get_benchmark
    from benchscore.runner import Runner
    from benchscore.store import ResultStore
    from benchscore.report.console import print_result

    config = load_config()
    set_config(config)

    # 解析 benchmark 列表
    bench_names = [b.strip() for b in benchmarks.split(",")]
    bench_instances = []
    for name in bench_names:
        kwargs = {}
        if sample_size is not None:
            kwargs["sample_size"] = sample_size
        bench = get_benchmark(name, **kwargs)
        bench_instances.append(bench)

    # 创建适配器
    try:
        adapter = create_adapter(model_id, config, api_key=api_key)
    except ValueError as e:
        click.echo(f"错误: {e}", err=True)
        sys.exit(1)

    # 如果指定了 base_url（兼容接口），覆盖默认值
    if base_url:
        adapter._client.base_url = base_url

    # 运行
    click.echo(f"\n{'='*60}")
    click.echo(f"  模型: {model_id}")
    click.echo(f"  Benchmark: {', '.join(bench_names)}")
    click.echo(f"  并发: {concurrency}")
    click.echo(f"{'='*60}\n")

    def log_handler(level: str, msg: str):
        click.echo(f"[{level}] {msg}")

    runner = Runner(
        adapter=adapter,
        benchmarks=bench_instances,
        concurrency=concurrency,
        hf_endpoint=config.hf_endpoint,
        on_log=log_handler,
    )

    try:
        result = asyncio.run(runner.run())
    except KeyboardInterrupt:
        runner.cancel()
        click.echo("\n已取消")
        return

    # 保存结果
    store = ResultStore(output or config.results_dir)
    filepath = store.save(result)

    # 输出结果
    print_result(result)
    click.echo(f"\n结果已保存至: {filepath}")


@main.command()
def list():
    """列出可用的 Benchmark"""
    from benchscore.benchmarks import list_benchmarks
    from benchscore.config import load_config

    config = load_config()
    click.echo("\n可用的 Benchmark:\n")

    benchmarks = {
        "mmlu": "知识维度 — 57学科多选题，5-shot",
        "gsm8k": "推理维度 — 小学数学应用题，5-shot CoT",
        "humaneval": "代码维度 — Python代码生成，0-shot",
    }

    for name in list_benchmarks():
        desc = benchmarks.get(name, "")
        defaults = config.get_benchmark_defaults(name)
        click.echo(f"  {name:12s}  {desc}")
        if defaults.sample_size:
            click.echo(f"  {'':12s}  默认采样: {defaults.sample_size} 题, "
                       f"few-shot: {defaults.few_shot}")


@main.command()
@click.argument("model_id", required=False)
def history(model_id: str | None):
    """查看历史评测结果"""
    from benchscore.store import ResultStore

    store = ResultStore()
    runs = store.list_runs(model_id)

    if not runs:
        click.echo("暂无历史评测记录")
        return

    click.echo(f"\n{'='*80}")
    click.echo(f"{'模型':20s} {'时间':22s} {'总分':8s} {'费用':10s} {'Benchmarks'}")
    click.echo(f"{'='*80}")

    for r in runs[:50]:  # 最多显示 50 条
        click.echo(
            f"{r['model_id']:20s} "
            f"{r['timestamp']:22s} "
            f"{r['overall']:.4f}  "
            f"${r['total_cost_usd']:.4f}   "
            f"{', '.join(r['benchmarks'])}"
        )


if __name__ == "__main__":
    main()
