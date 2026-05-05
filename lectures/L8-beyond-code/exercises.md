# L8 · 代码之外 —— 习题解答

---

## 习题 1：给你的项目写一个完整 README

**题目**：按本讲的模板给 InferSim（或你的个人项目）写一份新的 README.md。

**解答**（InferSim 示例骨架）：
```markdown
# InferSim

纯 Python 实现的 LLM 推理性能模拟器，零第三方依赖。

## ✨ 特性

- 🧮 基于 FLOPs × MFU × 通信延迟的三层建模
- 🎯 TTFT / TPOT / TGS 三指标输出
- 🖥️ 支持 H20 / H800 / H200 / GB200
- 📦 零第三方依赖，开箱即用

## 🚀 快速开始

```bash
git clone https://github.com/sheshuchen/InferSim
cd InferSim
python3 main.py \
    --config-path hf_configs/qwen3-8B_config.json \
    --device-type h800 \
    --world-size 8
```

## 📘 示例

```bash
# DeepSeek-V3 decode 基准
bash example/deepseek-v3/dsv3_decode.sh

# Qwen3-30B-A3B prefill 基准
bash example/qwen3-30B-A3B/prefill.sh
```

## 🛠 开发

```bash
pip install -e .
pre-commit install
```

## 📖 文档

- [架构设计](docs/architecture.md)
- [支持的模型](docs/models.md)
- [性能公式](docs/formulas.md)

## 📄 许可证

Apache-2.0
```

---

## 习题 2：写一个规范的 Issue

**题目**：假设你使用某开源库时遇到 bug，按模板写一份 issue。

**解答**：
```markdown
## 问题描述
使用 `foo.py` 解析中文文件名时报 UnicodeDecodeError，英文文件名正常。

## 如何复现
```bash
git clone https://github.com/me/demo
cd demo
touch "中文.txt"
python3 -c "from foolib import parse; parse('中文.txt')"
```

## 期望行为
返回解析结果，不报错。

## 实际行为
```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File ".../foolib/core.py", line 42, in parse
    return open(path).read()
UnicodeDecodeError: 'ascii' codec can't decode byte 0xe4 in position 0: ordinal not in range(128)
```

## 环境
- OS: macOS 15.3 (arm64)
- Python: 3.11.7
- foolib: 2.1.3
- LANG: (empty)

## 我已尝试
- 手动 `open(path, encoding='utf-8')` 可以读
- 搜过 issue 区，没找到中文文件名相关报告

## 建议
是否可以在 `parse` 内部指定 `encoding='utf-8'`？如果方向合适，我可以提 PR。
```

---

## 习题 3：写一个能被 merge 的 PR

**题目**：给习题 2 的 bug 写一个 PR 描述。

**解答**：
```markdown
## 背景
Fix #42 —— 解析包含非 ASCII 字符的文件名时报 UnicodeDecodeError。

## 改动内容
- `foolib/core.py` 的 `parse` 函数在 `open` 时显式指定 `encoding='utf-8'`
- 补充测试 `tests/test_core.py::test_parse_non_ascii_filename`

## 设计思路
Python 默认编码取决于 locale，在未设置 `LANG` 的系统上会退化为 ascii，导致读非 ASCII 文件失败。
显式指定 utf-8 是最常见解法，兼容所有常见 locale。

考虑过替代方案：
1. 用 `open(path, mode='rb')` + 自行解码 —— 太重，影响现有 API 返回值类型
2. 改 `sys.setdefaultencoding` —— Python 3 中不支持，且属于全局污染

最终选择显式传 encoding，最小侵入。

## 验证
- [x] 新测试 `test_parse_non_ascii_filename` 通过
- [x] 原有 126 个测试全部通过
- [x] 本地 `tox -e py310,py311,py312` 通过

## 向后兼容
完全兼容。文件本身是 utf-8 时行为不变，之前会失败的现在会成功。
```

---

## 习题 4：把"糟糕邮件"改成"好邮件"

**题目**：把下面这封邮件重写：

```
主题：求助

喂你好，我下载了你的库，但是不能用，报错了。
你能帮我看看吗？急！在线等。

xxx
```

**解答**：
```
主题：foolib 2.1.3 在 macOS arm64 下解析中文文件名报错

你好 xxx，

背景：
  我在 macOS 15.3 arm64 + Python 3.11 上使用 foolib 2.1.3，
  解析中文文件名时遇到 UnicodeDecodeError。

问题：
  用 `foolib.parse("中文.txt")` 会抛异常，完整 traceback 见：
  https://github.com/yourlib/foolib/issues/42

已尝试：
  - 手动 `open(..., encoding='utf-8')` 可以正常读
  - 搜过 issue 区，未见相同问题

我的疑问：
  1. 这是已知 bug 还是我的使用方式有问题？
  2. 若是 bug，我愿意提 PR 修复，是否有方向建议？

感谢你抽时间看，不急，等你方便的时候回复即可。

祝好，
sheshuchen
```

**要点**：
- 主题具体可搜索
- 正文先说背景再说诉求
- 明确问题、已尝试的、期望对方做什么
- 尊重对方时间（"不急"）

---

## 习题 5：把四类文档分开写

**题目**：选一个你熟悉的工具（例如 pytest / git / vim），按 tutorial / how-to / reference / explanation 四类各写一段。

**解答**（以 pytest 为例）：

**Tutorial：手把手跑起来第一个测试**
```
1. pip install pytest
2. 新建 test_calc.py：
   def add(a, b): return a + b
   def test_add(): assert add(2, 3) == 5
3. 运行 pytest，看见 "1 passed"

恭喜！你跑通了第一个测试。
```

**How-to：只想在 CI 里跳过慢测试**
```
给慢测试加标记：
  @pytest.mark.slow
  def test_big():
      ...

CI 里跑：pytest -m "not slow"
```

**Reference：pytest.mark 完整参数**
```
pytest.mark.skip(reason="")
pytest.mark.skipif(condition, reason="")
pytest.mark.xfail(condition, reason="", strict=False)
pytest.mark.parametrize("a,b", [(1,2),(3,4)])
pytest.mark.<custom_name>   # 自定义 marker，需在 pyproject.toml 注册
```

**Explanation：为什么 pytest 不用 class 而用函数**
```
pytest 选择函数而非 xUnit 的 class 风格，是因为：
 1. 函数无状态 → 并行更容易
 2. fixture 通过依赖注入显式声明，阅读时清楚每个测试依赖什么
 3. 更 pythonic，学习成本更低

这也是为什么 fixture 不是 setUp/tearDown（面向对象风格），
而是装饰器 + yield（函数式风格）。
```

---

## 习题 6：把"项目参与"写成简历条目

**题目**：你曾经参与/正在做 InferSim 项目，写一条不超过 3 行的简历条目。

**解答**：
```
InferSim —— LLM 推理性能模拟器（个人 / 开源项目）
• 从 0 设计纯 Python 零依赖建模层，覆盖 MHA/MLA/MoE/Grouped GEMM/DSA 等 7 类算子；
  基于 FLOPs × MFU × 通信延迟的三层建模，支持 H20/H800/H200/GB200 四类硬件。
• 支持 Qwen3 系列 + DeepSeek-V3/V3.2 等 8 个主流模型；真机对比 TTFT/TPOT 预测误差 < 5%。
• 代码 3000+ 行，已在团队 5 个推理项目中用于容量规划。
```

**结构**：项目名 + 自己角色 + **做了什么** + **带来什么数字**。

---

## 习题 7：设计你的公开技术存在感

**题目**：规划未来半年，每月做一件能被公开搜到的技术产出。

**解答**（示例）：

| 月份 | 产出 | 形式 | 可验证结果 |
|------|------|------|-----------|
| 1 | dotfiles 仓库 + README 规范 | GitHub | 有 Star 或至少有过访客 |
| 2 | InferSim README 翻新 + 首次 release | GitHub | PyPI 能 `pip install` |
| 3 | 写一篇"MoE TP/EP 协同性能建模"笔记 | 知乎/博客 | ≥ 500 字，有图 |
| 4 | 把 L1-L9 中文笔记整理成博客系列 | 博客 | 9 篇连载 |
| 5 | 给一个大项目提 1 个 merged PR | GitHub | 有 PR 链接 |
| 6 | 在校内/线上做一次技术分享 | 幻灯片 + 录屏 | 有可公开的 slides |

---

## 习题 8：找一个开源项目提 PR（真实任务）

**题目**：找一个你常用的开源项目，按下列步骤完成一次贡献。

**解答**（通用步骤）：
```bash
# 1. 选一个项目（例如 rich / requests / pytest）
# 2. 读 CONTRIBUTING.md

# 3. 找一个 "good first issue"
# 4. Fork + clone + 建分支
git clone git@github.com:sheshuchen/rich.git
cd rich
git switch -c docs/fix-typo

# 5. 修改（例如修个 typo）
# 6. 跑项目的 lint + test

# 7. commit + push
git commit -am "docs: fix typo in README"
git push -u origin docs/fix-typo

# 8. 网页提 PR，填完整描述

# 9. 根据 review 迭代
```

小技巧：
- **第一次 PR** 选文档 typo / 补例子最容易 merge
- 看最近合并的 PR，模仿它们的风格
- 如果被 reject，礼貌询问原因

---

## 本讲学习自检

- [ ] 你的主力项目 README 有快速开始 + 示例 + 文档链接
- [ ] 会写包含复现步骤的 Issue
- [ ] PR 有独立主题、测试、完整描述
- [ ] 分得清 tutorial/how-to/reference/explanation 四类文档
- [ ] 邮件/消息开门见山
- [ ] 至少给一个开源项目提过 1 个合并的 PR
- [ ] 有公开可见的技术存在（博客/GitHub 活跃度/分享）
