# ChangeLog

## [0.1.1] - 2026-08-11

### Fixed

- **`pip install -e .` 失败：`BackendUnavailable: Cannot import 'setuptools.backends._legacy'`**
  - `pyproject.toml` 中 `build-backend` 指向了不存在的模块 `setuptools.backends._legacy:_Backend`（该路径在任何 setuptools 版本中均不存在，无法完成 PEP 660 editable 构建）。
  - 已改为稳定兼容的后端：`setuptools.build_meta:__legacy__`（setuptools>=64 即支持 editable 安装）。
  - 修复后 `pip install -e .` 与 `pip install -e . --no-build-isolation` 均可正常安装。

### Changed

- 版本号由 `1.0.0` 调整为 `0.1.1`，同步更新 `pyproject.toml` 与 `benchscore/__init__.py` 中的 `__version__`。
