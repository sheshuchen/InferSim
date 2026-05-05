# L9 · 代码质量

> **原讲义链接**：https://missing.csail.mit.edu/2026/
> **本讲主题**：Code Quality
> **本文性质**：基于本讲主题编写的中文学习笔记（原创内容）

---

## 本讲要解决的问题

项目越做越大以后，你会遇到：
- 今天改的代码明天就忘了为什么要这么写
- 新人进项目读不懂，或者改一处就崩一片
- "在我机器上没问题"，但 CI 或同事机器就错
- 每次 review 都要花大量时间纠格式、纠命名、纠低级 bug

**代码质量**就是让这些痛苦**提前自动消除**的工程手段。本讲串讲五类工具：**格式化、静态检查、测试、覆盖率、CI**。

---

## 1. 代码质量的五道防线

```
       ┌─────────────────────────────────────┐
       │  CI（云端 PR gate）                  │
       ├─────────────────────────────────────┤
       │  Pre-commit hook（提交前本地）       │
       ├─────────────────────────────────────┤
       │  测试（pytest / unittest）           │
       ├─────────────────────────────────────┤
       │  静态检查（ruff / mypy / shellcheck）│
       ├─────────────────────────────────────┤
       │  格式化（black / ruff format / prettier）│
       └─────────────────────────────────────┘
        越往下越便宜、越早发现越便宜
```

**原则**：**问题越早发现越便宜**。
- 编辑器里发现：0 成本
- 本地 pre-commit 拦住：秒级
- CI 拦住：分钟级，但要回头改
- 线上发现：**灾难**

---

## 2. 格式化（Formatting）

### 2.1 为什么自动格式化比人工好

- **消除主观争论**：缩进 2 还是 4、单引号还是双引号——让工具决定，团队不吵
- **diff 更干净**：格式统一，每次 PR 的 diff 只包含真正的语义改动
- **review 更聚焦**：审阅者不用纠结格式，专心看逻辑

### 2.2 Python 主流选择

| 工具 | 定位 |
|------|------|
| **black** | 零配置、统一风格，业界事实标准 |
| **ruff format** | black 兼容但 10-100x 快（Rust 写的） |
| **isort** | 整理 import 顺序（现在 ruff 也能做） |
| **autopep8 / yapf** | 老牌，已被 black/ruff 取代 |

**推荐组合**：**ruff format + ruff check**，一个工具打包所有事。

### 2.3 配置范例

`pyproject.toml`：
```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # bugbear（容易出 bug 的写法）
    "UP",  # pyupgrade（用新 Python 语法）
]
ignore = ["E501"]   # 行长由 format 管，check 忽略
```

日常使用：
```bash
ruff format .              # 自动格式化
ruff check .               # 静态检查
ruff check --fix .         # 检查 + 可自动修的直接修
```

### 2.4 其他语言

| 语言 | 工具 |
|------|------|
| JavaScript / TypeScript / JSON / YAML / Markdown | **Prettier** |
| Go | `gofmt`（语言自带） |
| Rust | `rustfmt` |
| C/C++ | `clang-format` |
| Shell | `shfmt` |

---

## 3. 静态检查（Linting / Type Checking）

格式化管"长相"，静态检查管"逻辑/正确性"——在不运行代码的前提下发现问题。

### 3.1 Python 静态检查工具

| 工具 | 覆盖 |
|------|------|
| **ruff** | 合并了 flake8/pylint/isort 等几十种检查 |
| **mypy** | 类型检查（需配合类型注解） |
| **pyright** | 微软出，比 mypy 快，VSCode 内置 |
| **bandit** | 安全检查（硬编码密码、eval 等） |

### 3.2 为什么要写类型注解

```python
# ❌ 不友好
def process(data):
    return data["key"]["nested"]

# ✅ 类型注解
def process(data: dict[str, dict[str, int]]) -> int:
    return data["key"]["nested"]
```

好处：
- 编辑器跳转、补全更准
- mypy 能在**不跑代码**的情况下发现类型错误
- 代码自带文档

### 3.3 shellcheck：shell 脚本的救命工具

shell 脚本里最容易踩的坑（没引号、变量大小写、[[ vs [ 等）都能被 `shellcheck` 抓住。

```bash
brew install shellcheck
shellcheck backup.sh
```

**强烈建议**：任何超过 10 行的 shell 脚本都过一遍 shellcheck。

---

## 4. 测试（Testing）

### 4.1 测试的三个层次

| 层次 | 范围 | 速度 | 比例 |
|------|------|------|------|
| **Unit test（单元）** | 单个函数/类 | 毫秒级 | 最多 |
| **Integration test（集成）** | 多模块配合 | 秒级 | 中等 |
| **End-to-end（端到端）** | 整个系统 | 分钟级 | 最少 |

经典比例：**70% 单元 + 20% 集成 + 10% E2E**（测试金字塔）。

### 4.2 pytest 基础

```python
# tests/test_calc.py
import pytest
from mymod.calc import add, divide

def test_add():
    assert add(2, 3) == 5

def test_divide_normal():
    assert divide(6, 2) == 3

def test_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)

@pytest.mark.parametrize("a,b,want", [(1,1,2), (2,3,5), (10,-1,9)])
def test_add_many(a, b, want):
    assert add(a, b) == want
```

运行：
```bash
pytest                     # 跑所有
pytest tests/test_calc.py  # 跑单个文件
pytest -k divide           # 名字含 divide 的
pytest -x                  # 第一个失败就停
pytest -v                  # 详细
pytest --lf                # 只跑上次失败的（快速迭代神器）
```

### 4.3 Fixture：测试前/后的准备工作

```python
@pytest.fixture
def tmp_data(tmp_path):
    f = tmp_path / "data.txt"
    f.write_text("hello")
    return f

def test_read(tmp_data):
    assert tmp_data.read_text() == "hello"
```

`tmp_path` 是 pytest 内置 fixture，每个测试一个独立临时目录，自动清理。

### 4.4 好测试的特征

- **快**：跑不起来的测试就不会被跑
- **独立**：一个测试不依赖另一个的状态
- **可重复**：今天跑过明天跑还是一样结果（不依赖网络、时间、随机）
- **专一**：一个测试只测一件事
- **可读**：测试本身就是最好的文档

### 4.5 TDD 节奏

**红 → 绿 → 重构**：
1. **红**：先写测试，此时它必然失败
2. **绿**：写最简单的实现让测试通过
3. **重构**：代码整洁化，但保证测试依然通过

---

## 5. 覆盖率（Coverage）

覆盖率告诉你**哪些代码被测试跑到了、哪些没有**。

```bash
pip install coverage pytest-cov

pytest --cov=mymod --cov-report=term-missing
# 会输出每个文件的覆盖率 + 没被覆盖的行号

pytest --cov=mymod --cov-report=html
open htmlcov/index.html    # 可视化每行覆盖情况
```

**重要提醒**：
- 覆盖率 **不等于**代码质量
- 100% 覆盖率也可能没测到边界情况
- 覆盖率**低**是明确的警告；**高**只说明"代码被执行过"
- 推荐**核心模块 ≥ 80%**，辅助代码 60% 也可接受

---

## 6. Pre-commit Hook

**目标**：把格式化 + 静态检查 + 简单 lint 放在 `git commit` 之前自动跑，不合格不许提交。

安装 `pre-commit` 框架：
```bash
pip install pre-commit
```

`.pre-commit-config.yaml`：
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/koalaman/shellcheck-precommit
    rev: v0.10.0
    hooks:
      - id: shellcheck
```

启用：
```bash
pre-commit install        # 装 git hook
pre-commit run --all-files  # 手动过一遍全部文件
```

之后每次 `git commit` 自动跑，不过就拦住。

---

## 7. 持续集成（CI）

**CI**（Continuous Integration）：每次 push/PR 都在云端跑一遍 lint + test，不通过不给 merge。

### 7.1 GitHub Actions 最小示例

`.github/workflows/ci.yml`：
```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install
        run: |
          pip install -e ".[dev]"
      - name: Lint
        run: |
          ruff check .
          ruff format --check .
      - name: Test
        run: |
          pytest --cov=mymod --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
```

每次 push 都会：
- 在 3 个 Python 版本下跑
- 格式/lint 不通过就失败
- 测试失败就失败
- 把覆盖率上传到 Codecov

### 7.2 CI 的其他常见任务

- **构建**：打 wheel、Docker 镜像
- **发布**：tag 触发自动 `twine upload` / 自动 GitHub Release
- **部署**：main 合入触发自动部署到测试/生产环境
- **定时**：每天跑一次完整回归测试

### 7.3 CI 最佳实践

- **快**：慢 CI 没人愿意等，尽量并行、缓存依赖
- **稳**：不要 flaky test，网络/时间相关的测试加重试或 mock
- **分层**：PR 只跑快的（lint + 单元测试），合入后再跑慢的（集成 + E2E）
- **失败可读**：失败日志要清楚，让贡献者不用猜

---

## 8. 代码评审（Code Review）

代码质量不只靠工具，**人工 review** 是最后一道把关。

### 8.1 作为 Reviewer

- 先看整体设计，再看实现细节
- 区分 **blocking**（必须改）和 **nit**（小建议）
- 给 **建议**而不是 **命令**："可以考虑…"比"必须改成…"更易接受
- 表扬好的地方（鼓励持续输出优质代码）

### 8.2 作为 Author

- **先自己 review 一遍** 再请别人看
- 接受批评，但**捍卫**设计（有道理就讨论，无需盲从）
- 小 PR，别让 reviewer 啃 2000 行

---

## 9. 项目健康度检查清单

给任何 Python 项目打分：

- [ ] 有 `pyproject.toml`
- [ ] 有 `.gitignore`
- [ ] 有 `README.md` 含快速开始
- [ ] 有 `LICENSE`
- [ ] 有 `tests/` 目录，`pytest` 能跑
- [ ] 有 `pre-commit` 配置
- [ ] 有 CI（GitHub Actions / GitLab CI / 其他）
- [ ] 有格式化配置（ruff / black）
- [ ] 有类型注解（至少核心 API）
- [ ] 有 CHANGELOG.md

**满 8 项起**，项目才算"工程化合格"。

---

## 10. 小结

- [x] 代码质量五道防线：格式化 → 静态检查 → 测试 → 覆盖率 → CI
- [x] ruff 一个工具管格式化 + lint，比旧组合快 10-100 倍
- [x] 写类型注解 + mypy/pyright 捕获类型错
- [x] pytest 四要素：test_ 前缀、assert、fixture、parametrize
- [x] 覆盖率是参考，不是目标
- [x] pre-commit hook 本地拦低级错误
- [x] GitHub Actions 写一个最小 CI，自动 lint + 测试
- [x] 代码评审注重设计 > 格式

---

## 全课结语

至此 9 讲全部走完。这门课真正想教的**不是具体某个工具**，而是一种态度：

> **把重复劳动交给机器，把时间留给真正需要思考的事。**

课程提到的每个工具你现在可能都没时间深挖，但**知道它的存在**比什么都重要——遇到问题时你会想起"有专门工具"，而不是用土办法硬抗。

祝你在未来的研究生生涯中把这些基础工程能力用成肌肉记忆，然后在 AI 系统研究上走得更远。
