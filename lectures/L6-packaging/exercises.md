# L6 · 打包与发布代码 —— 习题解答

---

## 习题 1：把一个脚本变成可安装的包

**题目**：有一个 `hello.py`，里面有 `def main(): print("hi")`，改造成可以 `pip install .` 再用 `hello` 命令执行。

**解答**：目录结构：
```
hello-pkg/
├── pyproject.toml
├── README.md
└── src/
    └── hello/
        ├── __init__.py
        └── __main__.py
```

`pyproject.toml`：
```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "hello-sheshuchen"
version = "0.1.0"
description = "A tiny greeting CLI"
requires-python = ">=3.10"

[project.scripts]
hello = "hello:main"
```

`src/hello/__init__.py`：
```python
def main():
    print("hi")
```

验证：
```bash
pip install -e .
hello                        # 输出：hi
```

---

## 习题 2：用 uv 从零搭项目

**题目**：用 uv 创建项目、装依赖、跑一段代码。

**解答**：
```bash
uv init my-demo
cd my-demo
uv add requests rich
uv add --dev pytest

cat > main.py <<'EOF'
from rich import print
import requests

r = requests.get("https://api.github.com")
print({"status": r.status_code, "headers_count": len(r.headers)})
EOF

uv run python main.py
# 还会在项目根生成 uv.lock 记录精确版本
```

查看锁文件：`cat uv.lock | head`，会发现每个包都锁到了 commit 级别的 hash。

---

## 习题 3：写一个带 argparse 的 CLI 工具

**题目**：写 `wordcount` 命令，接受文件路径参数，输出单词数、行数、字符数。

**解答**：
```python
# src/wordcount/__init__.py
import argparse
import sys

def main():
    p = argparse.ArgumentParser(prog="wordcount")
    p.add_argument("path", help="文件路径")
    p.add_argument("--chars", action="store_true", help="输出字符数")
    args = p.parse_args()

    try:
        with open(args.path, encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"文件不存在: {args.path}", file=sys.stderr)
        sys.exit(1)

    lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
    words = len(text.split())
    chars = len(text)

    print(f"行数: {lines}")
    print(f"单词数: {words}")
    if args.chars:
        print(f"字符数: {chars}")
```

`pyproject.toml`：
```toml
[project.scripts]
wordcount = "wordcount:main"
```

```bash
pip install -e .
wordcount README.md
wordcount README.md --chars
```

---

## 习题 4：对比三种依赖声明方式

**题目**：同一个项目分别用 `requirements.txt`、`pyproject.toml + uv`、`conda env` 三种方式声明依赖，说明差异。

**解答**：

| 方式 | 声明位置 | lock 机制 | 跨语言支持 | 适合场景 |
|------|----------|-----------|------------|----------|
| `requirements.txt` | 一个文本文件，每行一个包 | 需配合 pip-tools 生成 `.lock` | ❌ 仅 Python | 最小可行、老项目 |
| `pyproject.toml` + uv/poetry | 标准化的 `[project.dependencies]` | `uv.lock` / `poetry.lock` | ❌ 仅 Python | 现代 Python 项目首选 |
| `conda env` | `environment.yml` | 版本精度有限 | ✅ 能装 CUDA、C 库 | 深度学习、科学计算 |

示例文件：
```txt
# requirements.txt
numpy==1.26.4
pandas==2.2.1
```

```toml
# pyproject.toml
[project]
dependencies = [
    "numpy>=1.26",
    "pandas>=2.2",
]
```

```yaml
# environment.yml
name: my-ml
channels: [pytorch, nvidia, conda-forge]
dependencies:
  - python=3.11
  - pytorch=2.3
  - pytorch-cuda=12.1
  - pip:
      - rich
```

---

## 习题 5：写最小 Dockerfile 并跑起来

**题目**：给习题 1 的 `hello-sheshuchen` 包写 Dockerfile，构建并运行。

**解答**：
```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .
ENTRYPOINT ["hello"]
```

```bash
docker build -t hello:0.1 .
docker run --rm hello:0.1           # 输出：hi
```

镜像大小优化：
```dockerfile
FROM python:3.11-alpine AS builder   # alpine 更小
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .

FROM python:3.11-alpine
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/hello /usr/local/bin/hello
ENTRYPOINT ["hello"]
```

---

## 习题 6：版本升级与 Git tag

**题目**：给项目做一次 `0.1.0 → 0.2.0` 的发版，包括更新 pyproject、打 tag、更新 changelog。

**解答**：
```bash
# 1. 改版本号
# pyproject.toml: version = "0.2.0"

# 2. 更新 CHANGELOG.md
cat >> CHANGELOG.md <<'EOF'

## [0.2.0] - 2026-05-03
### Added
- 支持命令行 --verbose 选项
### Fixed
- 修复空文件导致崩溃的问题
EOF

# 3. 提交
git add pyproject.toml CHANGELOG.md
git commit -m "release: v0.2.0"

# 4. 打 tag
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin main
git push origin v0.2.0

# 5. 构建并发布
python3 -m build
twine upload dist/*                  # 或 uv publish
```

---

## 习题 7：把 Python 脚本打成单文件可执行程序

**题目**：把 `wordcount` 打成一个不依赖 Python 解释器的单文件二进制。

**解答**：用 PyInstaller：
```bash
pip install pyinstaller
pyinstaller --onefile src/wordcount/__init__.py --name wordcount

./dist/wordcount README.md           # 无需 Python 环境
```

注意：
- 产物体积较大（几十 MB，因为内嵌了 Python 解释器）
- 跨平台需在目标平台上构建（Mac 上打的包只能在 Mac 上跑）

想更小的产物可用 **Nuitka**：
```bash
pip install nuitka
python3 -m nuitka --standalone --onefile src/wordcount/__init__.py
```

---

## 习题 8：综合——给 InferSim 写完整打包配置

**题目**：给当前 InferSim 项目补齐 `pyproject.toml`，使其能 `pip install -e .`，并提供 `infersim` 命令。

**解答**：
```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "infersim"
version = "0.1.0"
description = "纯 Python 实现的 LLM 推理性能模拟器"
authors = [{name = "sheshuchen"}]
readme = "README.md"
requires-python = ">=3.10"
license = {text = "Apache-2.0"}
dependencies = []                    # 纯 Python 零依赖

[project.optional-dependencies]
dev = ["pytest", "ruff", "pre-commit"]

[project.scripts]
infersim = "main:main"               # 或你实际的入口

[project.urls]
Repository = "https://github.com/sheshuchen/InferSim"

[tool.setuptools]
py-modules = ["main"]
packages = [
    "comm", "config", "flops", "hardware", "hf_configs",
    "kvcache", "layers", "mfu", "models", "params",
]

[tool.ruff]
line-length = 100
```

验证：
```bash
pip install -e .
infersim --help
```

---

## 本讲学习自检

- [ ] 写过至少一个 `pyproject.toml` 并成功 `pip install -e .`
- [ ] 用过 uv/poetry 生成 lock 文件
- [ ] 知道 `src/` 布局的好处
- [ ] 能给项目加命令行入口
- [ ] 能写一个最小可用 Dockerfile 并运行
- [ ] 理解 SemVer 的 MAJOR/MINOR/PATCH 含义
- [ ] 跑通一次 tag → GitHub Release 的发布流程
