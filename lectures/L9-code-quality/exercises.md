# L9 · 代码质量 —— 习题解答

---

## 习题 1：用 ruff 规范一个杂乱项目

**题目**：给一个没有规范的 Python 项目配好 ruff，一键格式化 + 检查。

**解答**：
```bash
pip install ruff

# 在项目根加 pyproject.toml（或追加）
cat >> pyproject.toml <<'EOF'

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "UP"]
ignore = ["E501"]
EOF

# 运行
ruff format .                        # 格式化
ruff check .                         # 检查问题
ruff check --fix .                   # 检查 + 自动修
ruff check --statistics .            # 按规则统计问题数
```

首次跑完会看到大量自动修复。**建议单独开一个 PR 只做"chore: ruff format"**，方便 review。

---

## 习题 2：给函数加类型注解 + mypy 检查

**题目**：给下面函数加类型注解，并用 mypy 验证：
```python
def find_max(items, key=None):
    best = items[0]
    for x in items[1:]:
        if (key or (lambda y: y))(x) > (key or (lambda y: y))(best):
            best = x
    return best
```

**解答**：
```python
from typing import Callable, TypeVar, Iterable

T = TypeVar("T")

def find_max(
    items: Iterable[T],
    key: Callable[[T], float] | None = None,
) -> T:
    it = iter(items)
    try:
        best = next(it)
    except StopIteration:
        raise ValueError("items 不能为空")

    f = key if key is not None else (lambda y: y)

    for x in it:
        if f(x) > f(best):
            best = x
    return best
```

验证：
```bash
pip install mypy
mypy find_max.py --strict
```

**改进点**：
- 原函数若 `items` 为空会 `IndexError`，改为显式抛 `ValueError`
- 原代码每次比较都调用 `(key or (lambda y: y))`，低效；提前绑定
- 用 `TypeVar` 让类型随输入变化（泛型）

---

## 习题 3：写一组 pytest 测试

**题目**：为习题 2 的 `find_max` 写完整测试。

**解答**：
```python
# tests/test_find_max.py
import pytest
from mymod import find_max

def test_find_max_basic():
    assert find_max([3, 1, 4, 1, 5, 9, 2, 6]) == 9

def test_find_max_single():
    assert find_max([42]) == 42

def test_find_max_negative():
    assert find_max([-3, -1, -7]) == -1

def test_find_max_with_key():
    words = ["a", "bb", "ccc"]
    assert find_max(words, key=len) == "ccc"

def test_find_max_empty_raises():
    with pytest.raises(ValueError):
        find_max([])

@pytest.mark.parametrize("items,want", [
    ([1], 1),
    ([1, 2], 2),
    ([2, 1], 2),
    ([1.5, 2.5, 1.0], 2.5),
])
def test_find_max_many(items, want):
    assert find_max(items) == want

def test_find_max_with_generator():
    # 支持一次性迭代器
    assert find_max(iter([5, 3, 8, 1])) == 8
```

运行：
```bash
pytest -v tests/test_find_max.py
```

**覆盖到**：空列表、单元素、负数、带 key、参数化多组、生成器。

---

## 习题 4：用 coverage 找没测到的分支

**题目**：给上面的 `find_max` 跑 coverage，找出还没覆盖的行。

**解答**：
```bash
pip install coverage pytest-cov

pytest --cov=mymod --cov-report=term-missing
# 输出类似：
# Name          Stmts   Miss  Cover   Missing
# mymod.py         10      1    90%   15

pytest --cov=mymod --cov-report=html
open htmlcov/index.html                # 红色行就是没覆盖到的
```

如果第 15 行是 `raise ValueError`，那说明还没测过空输入——刚好习题 3 补了 `test_find_max_empty_raises` 就覆盖上。

---

## 习题 5：配置 pre-commit hook

**题目**：在项目里启用 pre-commit，强制每次 commit 前跑 ruff + shellcheck。

**解答**：
```bash
pip install pre-commit

# 项目根创建 .pre-commit-config.yaml
cat > .pre-commit-config.yaml <<'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1024']

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
EOF

# 启用
pre-commit install

# 首次对全部文件跑一遍
pre-commit run --all-files
```

之后 `git commit` 时若格式不对，会被拦下并自动修复，重新 `git add` 再 commit。

---

## 习题 6：写一个最小 GitHub Actions CI

**题目**：给项目加一个 CI workflow，对 PR 跑 ruff + pytest，Python 3.10/3.11/3.12 矩阵。

**解答**：
```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Setup Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'

      - name: Install
        run: |
          python -m pip install -U pip
          pip install -e ".[dev]"

      - name: Ruff format check
        run: ruff format --check .

      - name: Ruff lint
        run: ruff check .

      - name: Pytest
        run: pytest -v --cov=mymod --cov-report=xml

      - name: Upload coverage
        if: matrix.python-version == '3.12'
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
```

推送后 GitHub 会自动跑。PR 上会显示绿勾/红叉。

---

## 习题 7：识别 flaky test

**题目**：下面哪些测试是 flaky 的（不稳定的）？如何修？

```python
def test_now():
    assert datetime.now().year == 2026

def test_order():
    xs = set([1, 2, 3])
    assert list(xs) == [1, 2, 3]

def test_api():
    r = requests.get("https://api.example.com/user")
    assert r.json()["name"] == "Alice"

def test_random():
    assert random.randint(0, 10) > 5
```

**解答**：

| 测试 | 为什么 flaky | 怎么修 |
|------|------------|--------|
| `test_now` | 到了 2027 年就失败 | 用 `freezegun` mock 时间：`@freeze_time("2026-05-03")` |
| `test_order` | set 无序，`list(set(...))` 顺序不稳定 | 用 `sorted()` 或用 `==` 比较 set |
| `test_api` | 依赖外部网络 + 返回内容 | mock requests，或用 `responses`/`respx` 库构造固定响应 |
| `test_random` | 有 40% 几率失败 | 固定种子 `random.seed(42)`，或断言范围而非具体值 |

修后版本：
```python
from freezegun import freeze_time

@freeze_time("2026-05-03")
def test_now():
    assert datetime.now().year == 2026

def test_order():
    xs = {1, 2, 3}
    assert xs == {1, 2, 3}

def test_api(monkeypatch):
    class FakeResp:
        def json(self): return {"name": "Alice"}
    monkeypatch.setattr(requests, "get", lambda url: FakeResp())
    r = requests.get("https://api.example.com/user")
    assert r.json()["name"] == "Alice"

def test_random():
    random.seed(42)
    x = random.randint(0, 10)
    assert 0 <= x <= 10            # 断言范围
```

---

## 习题 8：检查一个项目的健康度

**题目**：用本讲清单给当前 InferSim 项目打分。

**解答**：
```bash
cd ~/Desktop/InferSim

# 一项项检查
ls pyproject.toml 2>/dev/null && echo "[✓] pyproject.toml" || echo "[ ] pyproject.toml"
ls .gitignore     2>/dev/null && echo "[✓] .gitignore"     || echo "[ ] .gitignore"
ls README.md      2>/dev/null && echo "[✓] README.md"      || echo "[ ] README.md"
ls LICENSE        2>/dev/null && echo "[✓] LICENSE"        || echo "[ ] LICENSE"
ls tests/         2>/dev/null && echo "[✓] tests/"         || echo "[ ] tests/"
ls .pre-commit-config.yaml 2>/dev/null && echo "[✓] pre-commit" || echo "[ ] pre-commit"
ls .github/workflows/ 2>/dev/null && echo "[✓] CI"         || echo "[ ] CI"
ls CHANGELOG.md   2>/dev/null && echo "[✓] CHANGELOG"      || echo "[ ] CHANGELOG"
```

InferSim 当前状态（截至本笔记编写时）：
- [ ] `pyproject.toml`（可补）
- [x] `.gitignore`
- [x] `README.md`
- [x] `LICENSE`
- [ ] `tests/`（可补）
- [x] `.pre-commit-config.yaml`
- [ ] CI workflow（可补）
- [ ] CHANGELOG.md（可补）

行动项：写 pyproject.toml + 补 tests + 加一个 CI workflow，健康度立刻上 8/10。

---

## 习题 9：TDD 红-绿-重构实战

**题目**：用 TDD 节奏实现一个 `roman(n)` 函数：把 1-3999 的整数转成罗马数字。

**解答**：

**红**：先写一个失败的测试
```python
def test_roman_1():
    assert roman(1) == "I"
```
跑 → `NameError: roman`

**绿**：最小实现
```python
def roman(n):
    return "I"
```
跑 → 通过 ✅

**红 + 绿 迭代**：
```python
def test_roman_2():  assert roman(2) == "II"
def test_roman_3():  assert roman(3) == "III"
def test_roman_4():  assert roman(4) == "IV"
def test_roman_9():  assert roman(9) == "IX"
def test_roman_40(): assert roman(40) == "XL"
def test_roman_90(): assert roman(90) == "XC"
def test_roman_mmxxvi(): assert roman(2026) == "MMXXVI"
def test_roman_3999(): assert roman(3999) == "MMMCMXCIX"
```

一步步演化实现：
```python
def roman(n: int) -> str:
    if not 1 <= n <= 3999:
        raise ValueError("n must be in [1, 3999]")
    table = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100,  "C"), (90,  "XC"), (50,  "L"), (40,  "XL"),
        (10,   "X"), (9,   "IX"), (5,   "V"), (4,   "IV"),
        (1,    "I"),
    ]
    out = []
    for v, s in table:
        while n >= v:
            out.append(s)
            n -= v
    return "".join(out)
```

**重构**：用贪心 + 查表让代码简洁、测试保持全绿。

---

## 习题 10：最终综合——给 InferSim 加完整代码质量防线

**题目**：按本讲 10 条清单给 InferSim 项目补齐工程化设施。

**解答** —— 一次性落地清单：

1. 完善 `pyproject.toml`（见 L6 习题 8）
2. 加 `tests/` 目录 + 至少为 `flops/flops.py`、`mfu/mfu.py` 的核心函数写几条测试
3. 扩充 `.pre-commit-config.yaml`（项目已有，检查是否最新）
4. 加 `.github/workflows/ci.yml`（按习题 6）
5. 加 `CHANGELOG.md`（按 L6 格式）
6. 给核心函数加类型注解（至少 `flops.py`、`mfu.py`）
7. `ruff check --fix . && ruff format .` 全仓过一遍
8. 写 `CONTRIBUTING.md` 说明开发流程（装 pre-commit、提 PR 规范）

执行顺序：**1 → 2 → 3 → 4 → 5 → 6 → 7 → 8**，每一步单独 PR，review 友好。

---

## 本讲学习自检

- [ ] 会配 ruff：一条命令既能格式化也能 lint
- [ ] 能给核心函数加类型注解并跑 mypy
- [ ] 熟练使用 pytest 基础 + fixture + parametrize
- [ ] 会用 coverage 找未覆盖代码
- [ ] 项目里启用了 pre-commit
- [ ] 有一个最小可用 CI workflow
- [ ] 能识别并修 flaky test
- [ ] 按健康度清单给自己的项目打过分

---

## 全课完结

恭喜你走完 9 讲。核心外卖（takeaway）：

1. **工具链是战斗力倍增器**，但**底层原理**才是长期竞争力
2. **把重复劳动交给机器**（git hook、CI、ruff、测试），你的时间用于思考
3. **协作能力 = 写作能力 + 评审能力 + AI 协作能力**
4. **小步迭代、频繁提交、相信测试**，是面对复杂系统时唯一的理智之道

接下来就看实战。所有能想到的东西，都去 `man` 一下、查 `tldr`、写个小脚本试试——**习惯养成**比读完笔记重要 10 倍。
