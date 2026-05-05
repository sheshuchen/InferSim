# L6 · 打包与发布代码

> **原讲义链接**：https://missing.csail.mit.edu/2026/
> **本讲主题**：Packaging and Shipping Code
> **本文性质**：基于本讲主题编写的中文学习笔记（原创内容）

---

## 本讲要解决的问题

你写完代码，想让别人（或自己半年后）在另一台机器上跑起来。常见痛点：

- "在我电脑上好好的啊" —— 依赖版本不同
- "这个包怎么装？" —— 没有 `pyproject.toml` / `setup.py`
- "发布给全世界用怎么发？" —— 不懂 PyPI / Docker / Release
- "用户装完怎么跑起来？" —— 没有命令行入口

本讲讲从**一个脚本**到**一个可安装、可分发、可运行的软件产品**之间的全部工程工作。

---

## 1. 打包的三个层次

| 层次 | 做什么 | 典型产物 |
|------|--------|----------|
| **1. 源码级打包** | 整理成"可安装的项目" | `pyproject.toml` + `pip install .` |
| **2. 依赖固定** | 锁死版本，保证复现 | `requirements.txt` / `poetry.lock` |
| **3. 环境打包** | 连运行环境一起打走 | Docker 镜像 / conda env / 二进制可执行文件 |

---

## 2. Python 项目打包

### 2.1 现代标准：`pyproject.toml`

Python 打包历经 `setup.py` → `setup.cfg` → **`pyproject.toml`** 的演进，新项目**直接用 `pyproject.toml`**。

最简例子：
```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "infersim"
version = "0.1.0"
description = "A pure-Python LLM inference performance simulator"
authors = [{name = "sheshuchen", email = "you@example.com"}]
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
dependencies = [
    # 纯 Python 零依赖也可以是空的
]

[project.optional-dependencies]
dev = ["pytest>=7", "ruff", "mypy"]

[project.scripts]
infersim = "infersim.cli:main"       # 装完可直接执行 infersim

[project.urls]
Repository = "https://github.com/sheshuchen/InferSim"
```

### 2.2 目录结构约定

```
infersim/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── infersim/           # 真正的 Python 包
│       ├── __init__.py
│       ├── cli.py
│       └── ...
└── tests/
    └── test_cli.py
```

**建议**：用 `src/` 布局（不是把包直接丢在仓库根）。好处是：导入失败时能立刻暴露，而不是 "本地 import 成功但装出来的包 broken"。

### 2.3 本地开发安装

```bash
pip install -e .             # editable 模式，改代码立刻生效
pip install -e ".[dev]"      # 连 dev 依赖一起装
```

---

## 3. 依赖管理：requirements.txt vs lock 文件

### 3.1 最朴素：requirements.txt

```txt
numpy>=1.24
pandas>=2.0
requests
```

问题：`pandas>=2.0` 装出来可能是 2.0.3、2.1.0、2.2.1……不同机器会有差异。

### 3.2 lock 文件：锁死到**具体版本 + 哈希**

| 工具 | lock 文件 |
|------|-----------|
| pip-tools | `requirements.lock` |
| **Poetry** | `poetry.lock` |
| **uv** | `uv.lock` |
| pipenv | `Pipfile.lock` |

流程：`pyproject.toml` 声明语义范围 → 工具解析 → 写入 lock 文件 → 其他机器用 lock 精确复现。

### 3.3 推荐：用 uv

```bash
# 创建项目
uv init my-project
cd my-project

# 添加依赖（自动写进 pyproject.toml 和 uv.lock）
uv add numpy pandas
uv add --dev pytest ruff

# 安装
uv sync                      # 根据 lock 文件装

# 运行命令
uv run python main.py
uv run pytest
```

uv 的优势：**极快**（Rust 实现）、一个工具同时管虚拟环境 + 依赖 + lock + 运行。

---

## 4. 发布到 PyPI

```bash
# 1. 构建
pip install build
python3 -m build              # 生成 dist/ 下的 .whl 和 .tar.gz

# 2. 上传（首先注册 PyPI 账号，配 ~/.pypirc 或 API token）
pip install twine
twine upload dist/*

# 3. 其他人就可以
pip install infersim
```

**测试环境**：先传到 TestPyPI 验证没问题：
```bash
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ infersim
```

---

## 5. 命令行入口（CLI）

让用户装完直接敲 `infersim ...`：

```python
# src/infersim/cli.py
import argparse

def main():
    p = argparse.ArgumentParser(prog="infersim")
    p.add_argument("--config-path", required=True)
    p.add_argument("--device-type", default="h800")
    p.add_argument("--world-size", type=int, default=8)
    args = p.parse_args()

    # ...核心逻辑...
    print(f"模拟 {args.config_path} @ {args.device_type}")

if __name__ == "__main__":
    main()
```

配合 `pyproject.toml` 里的：
```toml
[project.scripts]
infersim = "infersim.cli:main"
```

`pip install .` 后就会生成一个 `infersim` 可执行文件。

**更强的 CLI 框架**：`click`、`typer`、`fire`。`typer` 尤其推荐——用类型注解自动生成参数解析。

---

## 6. Docker：把环境也打进镜像

Python 项目很少会纯 Python 跑起来；再加上 CUDA、系统库、字体……想让别人一键复现，**Docker** 是终极答案。

### 最小 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 先拷依赖文件利用 layer cache
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && uv sync --frozen

# 再拷源码
COPY src/ ./src/

ENTRYPOINT ["uv", "run", "infersim"]
```

```bash
docker build -t infersim:0.1 .
docker run --rm infersim:0.1 --help
```

**多阶段构建**（减小镜像）：
```dockerfile
FROM python:3.11 AS builder
WORKDIR /build
COPY . .
RUN pip install build && python3 -m build

FROM python:3.11-slim
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install /tmp/*.whl
ENTRYPOINT ["infersim"]
```

### GPU 镜像

基础镜像换成 `nvidia/cuda:12.1.0-devel-ubuntu22.04` 或 `pytorch/pytorch:...`，用 `docker run --gpus all ...` 启动。

---

## 7. 版本号与发布流程

### 语义化版本（SemVer）

`MAJOR.MINOR.PATCH`：
- **MAJOR**：破坏性变更
- **MINOR**：向后兼容的新功能
- **PATCH**：向后兼容的 bug 修复

示例：`1.2.3` → 改 API 接口 → `2.0.0`；加新功能 → `1.3.0`；修 bug → `1.2.4`。

### Git tag + GitHub Release

```bash
# 更新版本号
# pyproject.toml: version = "0.2.0"

git commit -am "release: v0.2.0"
git tag v0.2.0
git push origin main --tags

# GitHub 网页 Releases → Draft a new release → 选 v0.2.0 tag
# 写 changelog，附上构建产物
```

### Changelog

维护一个 `CHANGELOG.md`：
```markdown
# Changelog

## [0.2.0] - 2026-05-03
### Added
- 支持 Qwen3-Next 80B 模型
### Fixed
- 修正 prefill 吞吐率公式

## [0.1.0] - 2026-04-01
- 初始发布
```

---

## 8. 其他打包方式速览

| 目标 | 工具 |
|------|------|
| 打成单文件可执行 Python | **PyInstaller** / **Nuitka** |
| 打成 macOS `.app` / Windows `.exe` | PyInstaller + 平台特定配置 |
| 打成 Linux 包 | `.deb` (debian)、`.rpm` (fedora)、**AppImage**（跨发行版） |
| 科学计算/ML 环境 | **conda** / **micromamba**，跨 Python + C 库 |
| 不可变、声明式、跨平台 | **Nix / Nix Flakes** |
| 前端/Node 项目 | npm publish / 单文件 bundle（esbuild） |

---

## 9. 小结

- [x] 会写 `pyproject.toml`，知道 `[build-system]`、`[project]`、`[project.scripts]` 各字段
- [x] 用 `src/` 布局组织项目
- [x] 用 `pip install -e .` 做本地开发
- [x] 用 `uv` / `poetry` 维护 lock 文件
- [x] 会给项目加命令行入口
- [x] 能写一个最小可用的 Dockerfile
- [x] 理解 SemVer 版本号规则
- [x] 发布流程：tag → release → changelog
