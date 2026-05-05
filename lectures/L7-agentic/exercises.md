# L7 · 智能体编程 —— 习题解答

---

## 习题 1：给一个真实项目写项目规则文件

**题目**：为 InferSim 项目写一份 `AGENTS.md`，让任何 agent 读完都能按项目约定工作。

**解答**：
```markdown
# InferSim 项目 · Agent 工作规则

## 项目定位
纯 Python 实现的 LLM 推理性能模拟器，零第三方依赖。
核心建模：FLOPs × MFU × 通信延迟 → TTFT/TPOT/TGS。

## 技术栈硬约束
- Python 3.10+
- **禁止引入任何第三方依赖**（no numpy, no torch, no anything）
- 纯标准库 + 项目自身代码

## 代码风格
- 缩进 4 空格
- 行宽 ≤ 100
- 所有注释用中文
- 函数注释说明输入输出、参数含义、张量形状（如适用）
- 命名：函数/变量 snake_case，类 PascalCase

## 目录约定
- `flops/` 计算量建模
- `mfu/` MFU 基准与算力利用率
- `comm/` 通信延迟建模
- `hardware/` GPU 硬件规格
- `models/` 模型推理主流程
- `layers/` 单层建模（attn/moe/linear_attn）
- `hf_configs/` HuggingFace 模型配置
- `bench_data/` 真机基准 CSV

## 运行方式
- 入口：`python3 main.py --config-path ... --device-type ... --world-size ...`
- 始终用 `python3`，不用 `python`

## Agent 工作边界
- 改动公式前必须在 PR/回复中贴出推导
- 改 kernel benchmark 维度需说明与真机参数的对应
- **禁止**未经同意自动重构/拆分现有函数
- **禁止**增加抽象层（不必要的基类、工厂）

## 验证
- 任何改动后跑一遍 `example/` 下的脚本确保不崩
- 涉及数值公式的改动需跑 Qwen3-8B 和 DeepSeek-V3 两个代表性模型
```

---

## 习题 2：用结构化 prompt 要求 AI 改代码

**题目**：你想让 AI 把 `layers/attn.py` 的某个函数加上详细中文注释。写一条高质量 prompt。

**解答**：
```
【任务】
为 layers/attn.py 中的 `forward` 函数补充中文注释

【输入文件】
- layers/attn.py（重点关注 forward 函数）
- models/model.py（只读，用于理解调用方式）

【注释要求】
1. 函数顶部：docstring 形式，写清楚：
   - 功能一句话
   - 每个参数的含义、类型、张量形状
   - 返回值含义与形状
2. 函数内关键步骤之前：一行中文注释，解释"这一步在做什么、为什么"
3. 复杂的 reshape、transpose：必须标注前后形状
4. 不要翻译成英文再直译回来，直接中文表达

【约束】
- 只改注释，不动任何可执行代码
- 不增加 import
- 不改函数签名

【验证】
- git diff 只有注释行变化
- `python3 main.py --config-path hf_configs/qwen3-8B_config.json` 仍能跑
```

---

## 习题 3：让 AI 分步骤做而不是一步到位

**题目**：你要让 AI 做一个涉及 5 个文件的重构，怎么避免它一次改飞？

**解答**：分步 prompt 模板：
```
这是一个多步任务，分阶段做：

【阶段 1：计划】
先不动代码。列出：
 1. 你打算改哪几个文件
 2. 每个文件改动的大致内容（方法名、目的，不贴完整代码）
 3. 改动顺序以及可能的风险

等我 approve 后再进入阶段 2。

【阶段 2：一个文件一个文件地改】
每改完一个文件：
 - 停下来，贴 diff 给我看
 - 我确认后再改下一个

【阶段 3：整体验证】
所有改完后：
 - git diff 汇总
 - 跑测试命令
 - 报告结果
```

实践经验：**加了"先列计划"这一步，错方向率降低一半以上**。

---

## 习题 4：识别 AI 幻觉

**题目**：AI 给你推荐了下面这段代码，说是 PyTorch 的"官方推荐写法"。快速判断真伪：

```python
import torch
x = torch.randn(10)
y = torch.auto_broadcast_add(x, x.unsqueeze(0))
```

**解答**：

判断步骤：
1. 查官方文档：`pytorch.org/docs/` 搜 `auto_broadcast_add` → 无此 API
2. 本机验证：`python3 -c "import torch; print(hasattr(torch, 'auto_broadcast_add'))"` → False
3. 问 AI："请给出这个 API 的官方文档链接" → 如果给不出或给一个 404 链接，几乎必然是幻觉

**结论**：PyTorch 没有 `auto_broadcast_add` 这个函数。广播加法直接写 `x + x.unsqueeze(0)` 即可，PyTorch 自动做广播。

**防幻觉 checklist**：
- [ ] 这个函数/类名我**亲眼**在官方文档见过吗？
- [ ] 代码能跑通吗？（执行是最终裁判）
- [ ] 这个参数顺序符合该库的惯用风格吗？
- [ ] AI 给的"文档链接"真的能打开吗？

---

## 习题 5：在有测试的项目里放手让 AI 改

**题目**：描述一套工作流：让 AI 帮你修一个 bug，但保证绝不引入新问题。

**解答**：
```bash
# 1. 干净的工作区
git status                          # 确保没有未提交改动
git switch -c fix/xxx               # 新分支

# 2. 先写复现 bug 的测试（人工）
cat > tests/test_bug.py <<'EOF'
def test_divide_by_zero_handling():
    assert safe_divide(1, 0) is None
EOF
pytest tests/test_bug.py            # 确认失败（说明 bug 存在）

# 3. 给 AI 完整上下文 + 让它修
# prompt 里包含：
# - 报错现场
# - 上面那个失败测试
# - "改动必须让这个测试通过，同时原有所有测试继续通过"

# 4. AI 改完
git diff                            # 人工审
pytest                              # 跑全量
# 全通过 → commit

git add -p                          # 挑改动提交
git commit -m "fix: 处理除零情况"

# 5. 有问题随时回滚
git reset --hard HEAD~1             # 不要了
```

---

## 习题 6：让 AI 做大型重构（带保护）

**题目**：想把项目里所有 `print` 换成 `logger.info`，怎么做最安全？

**解答**：
```bash
# 1. 先手动检查一下规模
git grep -n "print(" | wc -l        # 比如 47 处
git grep -l "print(" | sort -u      # 涉及的文件

# 2. 建分支
git switch -c refactor/print-to-logger

# 3. 给 AI 明确 prompt
# """
# 把仓库里所有 print(...) 改成 logger.info(...)，规则：
# 1. 每个涉及的 .py 文件在顶部加 `import logging; logger = logging.getLogger(__name__)`
# 2. 只换 print，不改其他任何代码
# 3. 不要改 tests/ 下的 print
# 4. 做一个文件，我审一个文件
# """

# 4. 每改一个文件人工 git diff 过
git diff layers/attn.py
# 确认 OK 后：
git add layers/attn.py
git commit -m "refactor(attn): print -> logger"

# 5. 全改完后跑一遍 example/
bash example/qwen3-8B/sim.sh
```

**关键**：一定要 **小步提交**，每一步都能回滚。

---

## 习题 7：上下文太长怎么办

**题目**：AI 和你聊了 30 轮，越聊越乱，输出越来越差。怎么挽救？

**解答**：

三个办法：

**1. 新开对话 + 带上重点信息**
```
重开对话时贴上：
- 当前目标（一句话）
- 当前进度（已经完成的事、已经定下的决定）
- 下一步要做什么
- 必要的代码片段
```

**2. 压缩历史**
让 AI 自己总结：
```
在开新对话前，请用不超过 300 字总结我们刚才讨论的：
 1. 最终要解决的问题
 2. 关键决定（选型、架构）
 3. 已经写了哪些代码、改了哪些文件
 4. 待办事项
```
把总结贴进新对话。

**3. 文档化到项目**
把讨论成果写进 `AGENTS.md`、`README.md`、issue 或 `design.md`，让后续会话从文档读取，而不是从历史对话。

---

## 习题 8：安装并体验一个 CLI agent

**题目**：任选一款 CLI 类 agent（aider / claude-code / codex），在一个小项目里完成一次真实的多文件修改。

**解答**（以 aider 为例）：
```bash
pip install aider-chat

# 配 API Key
export OPENAI_API_KEY=sk-xxx   # 或 ANTHROPIC_API_KEY

# 进入项目
cd hello-git
aider --model gpt-4o README.md src/hello/__init__.py
# 进入交互，输入比如：
# > 给 main 函数加一个 --loud 参数，带上会打印大写的 HI
```

Aider 的特性：
- 每次改动自动 `git commit`
- 错了直接 `git reset HEAD~1` 回退
- 可以用 `/add`、`/drop` 管理上下文里包含哪些文件

---

## 习题 9：AI 代码审查 checklist 实战

**题目**：AI 给了你下面这段"改进版" forward：

```python
def forward(self, x, mask=None):
    import numpy as np                          # A
    from typing import Optional                  # B
    q = self.q_proj(x).to(torch.float32)         # C
    k = self.k_proj(x)
    v = self.v_proj(x)
    scores = q @ k.transpose(-2, -1) / np.sqrt(self.head_dim)  # D
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e4)           # E
    attn = scores.softmax(-1)
    out = attn @ v
    return self.o_proj(out)                                    # F
```

指出至少 5 处问题。

**解答**：

| 位置 | 问题 | 说明 |
|------|------|------|
| A | **违反零依赖约定** | 项目规定纯 Python 零依赖，这里 `import numpy` 直接破坏 |
| B | **在函数体内 import** | 应放到文件顶部，否则每次调用都会走一次 import 检查 |
| C | **Q 单独转 FP32，K/V 没转** | 造成精度不一致，也会让 `q @ k.T` 触发类型提升/崩溃 |
| D | **用 numpy.sqrt** | 张量运算里不应混 numpy，会强制 .cpu() 同步，大幅拖慢；应用 `math.sqrt` 或 `self.head_dim ** 0.5` |
| E | **mask_fill 用 -1e4** | 半精度下 -1e4 仍是有效数值，softmax 后可能没有完全屏蔽；常见做法是 `float('-inf')` 或 `torch.finfo(dtype).min` |
| F | **缺失 dropout、scale 变量、shape 断言** | 原版若有 dropout 不该悄悄删；关键形状变化应加注释或 assert |

**复查结论**：**不能接受**。让 AI 改时明确指出上述 6 点，要求修改后再审一次。

---

## 习题 10：设计你的个人 AI 协作工作流

**题目**：结合你的研究方向（AI 系统），设计一套日常开发流程，规定 AI 介入的边界。

**解答（示例）**：

| 场景 | AI 参与方式 |
|------|------------|
| 调研新论文/新算子 | 让 AI 解释论文、对比方法；**最终理解必须自己来** |
| 写 benchmark 脚本 | AI 写初版，人工审后跑 |
| 写 kernel 建模公式 | **完全自己推**，AI 只做"帮我检查一下公式"的角色 |
| 数据处理/画图 | 全权 AI 做，我只定需求 |
| 找 bug | AI 看 trace 给假设，我验证 |
| 写论文/报告 | AI 写初稿，我精修到每一句话都能自己捍卫 |
| 敏感决定（跑真机大实验、改关键代码） | **不用 AI** |

**每日节奏**：
- 上午：研究性工作，独立思考为主
- 下午：工程任务，AI 加速为主
- 每周：总结 AI 产出的代码里有多少被我改过，调整协作方式

---

## 本讲学习自检

- [ ] 能说出 AI 编程三代的差异
- [ ] 知道 AI 的能力边界
- [ ] 为自己的项目写过 `AGENTS.md` 或 `.cursorrules`
- [ ] 写 prompt 会包含"目标/约束/验证"三段
- [ ] 有自己的一套小步提交 + 审 diff 的流程
- [ ] 能 5 秒识别 AI 幻觉 API
- [ ] 至少深度用熟一款 agent 工具
