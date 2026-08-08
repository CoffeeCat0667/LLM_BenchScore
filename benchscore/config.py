"""配置管理 — 加载 YAML 配置文件，支持环境变量覆盖。"""

from __future__ import annotations

import os

# ═══════════════════════════════════════════════════════════════════
# 必须在任何 huggingface_hub/datasets import 之前设置镜像，
# 否则 huggingface_hub 会在首次 import 时缓存默认端点
# ═══════════════════════════════════════════════════════════════════
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Windows 不支持 symlink 时 suppress 警告
if "HF_HUB_DISABLE_SYMLINKS_WARNING" not in os.environ:
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 未登录 HF 时的匿名访问警告 suppress
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ── 配置文件查找顺序 ──────────────────────────────────────────────
# 1. BENCHSCORE_CONFIG 环境变量指向的文件
# 2. 当前目录下的 benchscore.yaml
# 3. 包内 configs/default.yaml

def _find_config_dir() -> Path:
    """查找配置目录，按优先级：
    1. 环境变量 BENCHSCORE_CONFIG_DIR
    2. {项目根目录}/configs/ (开发和 pip install -e . 模式)
    3. {包目录}/configs/ (pip install 打包后)
    """
    env_dir = os.environ.get("BENCHSCORE_CONFIG_DIR")
    if env_dir and Path(env_dir).is_dir():
        return Path(env_dir)

    # {项目根目录}/configs/ — benchscore/ 的父目录
    pkg_parent = Path(__file__).resolve().parent.parent
    root_configs = pkg_parent / "configs"
    if root_configs.is_dir():
        return root_configs

    # {包目录}/configs/
    pkg_configs = Path(__file__).resolve().parent / "configs"
    return pkg_configs


def _find_default_config() -> Path:
    config_dir = _find_config_dir()
    # 1. 环境变量指定文件
    env_path = os.environ.get("BENCHSCORE_CONFIG")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    # 2. 当前目录下的 benchscore.yaml
    cwd_path = Path.cwd() / "benchscore.yaml"
    if cwd_path.exists():
        return cwd_path
    # 3. 配置目录下的 default.yaml
    return config_dir / "default.yaml"


def _find_models_config() -> Path:
    config_dir = _find_config_dir()
    env_path = os.environ.get("BENCHSCORE_MODELS_CONFIG")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    cwd_path = Path.cwd() / "models.yaml"
    if cwd_path.exists():
        return cwd_path
    return config_dir / "models.yaml"


@dataclass
class ModelInfo:
    """预置模型信息"""
    id: str
    name: str
    provider: str
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0


@dataclass
class BenchmarkDefaults:
    """单个 Benchmark 的默认配置"""
    sample_size: int | None = None     # None 表示全量
    few_shot: int = 0
    weight: float = 1.0


@dataclass
class Config:
    """全局配置单例"""

    # ── API 密钥 ──
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # ── HuggingFace ──
    hf_endpoint: str = "https://hf-mirror.com"
    hf_token: str = ""

    # ── 运行参数 ──
    concurrency: int = 10
    temperature: float = 0.0
    max_retries: int = 3
    request_timeout: int = 120

    # ── 存储 ──
    results_dir: str = "results"

    # ── 预置模型列表 ──
    models: list[ModelInfo] = field(default_factory=list)

    # ── Benchmark 默认参数 ──
    benchmark_defaults: dict[str, BenchmarkDefaults] = field(default_factory=dict)

    # ── 维度权重 ──
    dimension_weights: dict[str, float] = field(default_factory=lambda: {
        "knowledge": 0.20,
        "reasoning": 0.20,
        "code": 0.20,
        "language": 0.20,
        "safety": 0.20,
    })

    # ── 方法 ──

    def get_benchmark_defaults(self, name: str) -> BenchmarkDefaults:
        """获取某个 benchmark 的默认配置，不存在则返回空默认"""
        return self.benchmark_defaults.get(name, BenchmarkDefaults())

    def get_model(self, model_id: str) -> ModelInfo | None:
        """按 id 查找预置模型"""
        for m in self.models:
            if m.id == model_id:
                return m
        return None

    def get_models_by_provider(self, provider: str) -> list[ModelInfo]:
        """按 provider 筛选模型列表"""
        return [m for m in self.models if m.provider == provider]

    def to_dict(self) -> dict[str, Any]:
        """导出为字典（用于保存到 YAML）"""
        d = {}
        for k, v in self.__dict__.items():
            if k == "models":
                d[k] = [{kk: vv for kk, vv in m.__dict__.items()} for m in v]
            elif k == "benchmark_defaults":
                d[k] = {name: bd.__dict__ for name, bd in v.items()}
            elif k == "dimension_weights":
                d[k] = v
            else:
                d[k] = v
        return d


# ── 工厂函数 ──────────────────────────────────────────────────────

def _parse_benchmark_defaults(raw: dict) -> dict[str, BenchmarkDefaults]:
    return {k: BenchmarkDefaults(**v) for k, v in raw.items()}


def load_config(
    config_path: str | Path | None = None,
    models_path: str | Path | None = None,
) -> Config:
    """从 YAML 文件加载配置，环境变量覆盖敏感字段。

    - config_path: 主配置文件路径，None 则自动查找
    - models_path: 模型列表配置路径，None 则自动查找
    """
    config_file = Path(config_path) if config_path else _find_default_config()
    models_file = Path(models_path) if models_path else _find_models_config()

    cfg = Config()

    # ── 加载主配置 ──
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        cfg.hf_endpoint = data.get("hf_endpoint", cfg.hf_endpoint)
        cfg.concurrency = int(data.get("concurrency", cfg.concurrency))
        cfg.temperature = float(data.get("temperature", cfg.temperature))
        cfg.max_retries = int(data.get("max_retries", cfg.max_retries))
        cfg.request_timeout = int(data.get("request_timeout", cfg.request_timeout))
        cfg.results_dir = data.get("results_dir", cfg.results_dir)
        cfg.hf_token = data.get("hf_token", cfg.hf_token)

        if "benchmark_defaults" in data:
            cfg.benchmark_defaults = _parse_benchmark_defaults(data["benchmark_defaults"])

        if "dimension_weights" in data:
            cfg.dimension_weights = data["dimension_weights"]

    # ── 加载模型列表 ──
    if models_file.exists():
        with open(models_file, "r", encoding="utf-8") as f:
            models_data = yaml.safe_load(f) or {}

        models = []
        for provider, model_list in models_data.get("providers", {}).items():
            for m in model_list:
                m["provider"] = provider
                models.append(ModelInfo(**m))
        cfg.models = models

    # ── 环境变量覆盖 ──
    cfg.openai_api_key = os.environ.get("BENCHSCORE_OPENAI_API_KEY", "")
    cfg.anthropic_api_key = os.environ.get("BENCHSCORE_ANTHROPIC_API_KEY", "")
    if os.environ.get("BENCHSCORE_HF_ENDPOINT"):
        cfg.hf_endpoint = os.environ["BENCHSCORE_HF_ENDPOINT"]
    if os.environ.get("BENCHSCORE_HF_TOKEN"):
        cfg.hf_token = os.environ["BENCHSCORE_HF_TOKEN"]

    return cfg


def save_config(config: Config, path: str | Path) -> None:
    """保存配置到 YAML 文件"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    d = config.to_dict()
    # 不保存敏感信息
    d.pop("openai_api_key", None)
    d.pop("anthropic_api_key", None)
    d.pop("hf_token", None)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


# ── 全局单例 ──────────────────────────────────────────────────────

_config: Config | None = None


def get_config() -> Config:
    """获取全局配置单例"""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: Config) -> None:
    """替换全局配置"""
    global _config
    _config = config


def reload_config() -> Config:
    """重新加载配置"""
    global _config
    _config = load_config()
    return _config
