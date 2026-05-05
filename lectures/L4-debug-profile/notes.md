# L4 · 调试与性能分析

> **原讲义链接**：https://missing.csail.mit.edu/2026/
> **本讲主题**：Debugging and Profiling
> **本文性质**：基于本讲主题编写的中文学习笔记（原创内容）

---

## 本讲要解决的问题

写出的代码报错、结果不对、跑得慢——这三种情况几乎占据了程序员一半的工作时间。本讲系统性地讲：
- **遇到 bug 怎么系统地排查**（不是瞎改、不是复制粘贴到搜索引擎）
- **遇到性能问题怎么定位**（不是凭感觉改代码）

---

## 1. 调试的心态与方法论

在动手之前，先记住一条最重要的原则：

> **Debugging 不是修改代码，而是缩小不确定性。**

常见反面操作：
- "这段代码看着有点怪，我改一下试试" → 改坏了更多东西
- "网上搜到一个一样的错，把答案贴过来" → 改好了也不知道为什么

正确流程：
1. **精确复现**：能稳定重现 bug 才有排查前提
2. **最小化**：把输入、环境、代码缩到最小还能触发 bug
3. **二分定位**：不断把可能的范围减半（哪个函数？哪一行？哪个变量？）
4. **验证假设**：每次有一个"我猜是这里出问题"的假设，**设计一个实验去证伪**
5. **修复后写测试**：保证同样的 bug 不会再回来

---

## 2. 日志与打印

### 2.1 `print` 调试：什么时候够用

对于小脚本、临时实验，`print` 是最快的。但要养成好习惯：

```python
# ❌ 不好
print(x)
print("here")

# ✅ 好
print(f"[train_loop] step={step}, loss={loss:.4f}, grad_norm={norm:.4f}")
```

关键点：**打印变量名 + 位置标签**，方便事后搜索。

### 2.2 真正的日志库（logging）

脚本变大就该用日志库：

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

logger.debug("进入训练循环，batch_size=%d", bs)
logger.info("checkpoint 保存到 %s", path)
logger.warning("梯度出现 NaN，step=%d", step)
logger.error("无法加载模型", exc_info=True)
```

**五个级别**（从低到高）：`DEBUG / INFO / WARNING / ERROR / CRITICAL`，用 `setLevel()` 控制生产环境只输出哪些。

### 2.3 让日志好看：`rich`

```bash
pip install rich
```
```python
from rich.logging import RichHandler
logging.basicConfig(handlers=[RichHandler()], level=logging.INFO)
logger.info("带颜色的日志 [bold red]警告[/]")
```

---

## 3. Python 的交互式调试器 pdb

`print` 看一两个变量还行，看 20 个变量、想在断点处交互式探索就该用调试器。

```python
# 在想停的地方插一行
import pdb; pdb.set_trace()

# Python 3.7+ 更简洁：
breakpoint()
```

运行到此处会进入交互式 shell，常用命令：

| 命令 | 作用 |
|------|------|
| `n` / `next` | 执行下一行（不进入函数） |
| `s` / `step` | 进入当前函数内部 |
| `c` / `continue` | 继续运行到下一个断点 |
| `l` / `list` | 查看当前位置附近的代码 |
| `p x` | 打印变量 x |
| `pp x` | 漂亮地打印（大对象更清晰） |
| `w` / `where` | 查看调用栈 |
| `u` / `d` | 上/下一层栈帧 |
| `q` | 退出 |

推荐 **`ipdb`**（pdb + IPython 的增强版），Tab 补全更好用：
```bash
pip install ipdb
```
```python
import ipdb; ipdb.set_trace()
```

### 事后调试（post-mortem）

程序已经崩了，想回到崩溃现场看状态：
```bash
python3 -m pdb -c continue buggy.py
# 崩了之后自动进入 pdb，可以 p 变量、u/d 看栈
```

---

## 4. 其他语言与场景

| 场景 | 工具 |
|------|------|
| C/C++ | **gdb**、**lldb**（macOS） |
| 浏览器 JS | Chrome DevTools（断点、监视、Network） |
| Node.js | `node --inspect`，配合 Chrome DevTools |
| Rust | **rust-lldb** / **rust-gdb** |
| 大型项目全语言 | VSCode 调试面板（几乎所有语言都有支持） |

**核心思想是共通的**：设断点 → 观察变量 → 单步执行 → 修正假设。

---

## 5. 静态分析与类型检查

有些 bug 其实不用运行就能发现：

| 工具 | 作用 |
|------|------|
| **shellcheck** | 检查 shell 脚本里的坑 |
| **ruff / flake8 / pylint** | Python 静态检查 |
| **mypy / pyright** | Python 类型检查 |
| **eslint** | JavaScript 检查 |
| **clippy** | Rust 检查 |

集成到编辑器里（LSP 或保存时自动跑），写的时候就有红线提示，bug 提前被堵住。

---

## 6. 性能分析（Profiling）

性能优化第一定律：**Don't guess, measure**（别猜，测）。

### 6.1 计时的粗粒度方法

```bash
# shell
time ./run.sh

# Python 单行
python3 -c "import timeit; print(timeit.timeit('x*x', number=10_000_000))"
```

`time` 给你三个数：
- **real**：墙钟时间（你实际等了多久）
- **user**：CPU 在用户代码上的时间
- **sys**：CPU 在内核代码上的时间

如果 user 远大于 real → 多核并行；如果 real 远大于 user → IO 瓶颈或等待。

### 6.2 Python 内置 Profiler：`cProfile`

```bash
python3 -m cProfile -s cumulative my_script.py | head -30
# -s cumulative: 按累计耗时排序
```

**关注字段**：`cumtime`（含下层调用）、`tottime`（只在本函数内）、`ncalls`（被调次数）。

### 6.3 可视化：`snakeviz` / `py-spy`

```bash
# snakeviz：本地可视化 cProfile 结果
pip install snakeviz
python3 -m cProfile -o out.prof my_script.py
snakeviz out.prof

# py-spy：对**正在运行**的进程做采样式 profile（生产友好）
pip install py-spy
py-spy record -o profile.svg --pid 12345
py-spy top --pid 12345              # 实时 top-like 视图
```

**采样式（sampling） vs 插桩式（instrumenting）**：
- 采样式（`py-spy`）：开销小，可在生产环境用
- 插桩式（`cProfile`）：精确但拖慢程序，适合离线分析

### 6.4 行级分析 `line_profiler`

想看某个函数每一行的耗时：
```python
# 装饰想分析的函数
@profile
def slow_func(x):
    a = compute(x)
    b = heavy(a)
    return b + 1
```
```bash
pip install line_profiler
kernprof -l -v my_script.py
```

### 6.5 内存分析

| 工具 | 用途 |
|------|------|
| `memory_profiler` | 逐行看内存增长 |
| `tracemalloc`（内置） | 追踪内存泄漏 |
| `pympler` | 查看对象大小 |

### 6.6 系统级工具

| 场景 | 工具 |
|------|------|
| Linux CPU/IO 实时概览 | `htop`、`iotop`、`iftop` |
| 系统调用追踪 | `strace` (Linux)、`dtruss` (macOS) |
| 网络抓包 | `tcpdump`、`wireshark` |
| GPU 利用率 | `nvidia-smi`、`nvtop`、`nsys`（NVIDIA Nsight） |

---

## 7. 深度学习/大模型场景的 profiling

这是你日常会打交道的场景，单列出来：

| 工具 | 能看什么 |
|------|----------|
| `nvidia-smi -l 1` | GPU 利用率、显存、功耗（秒级） |
| `nvtop` | htop 风格的 GPU 实时面板 |
| **`torch.profiler`** | PyTorch 官方，逐算子耗时、CUDA kernel、通信 |
| **`NVIDIA Nsight Systems (nsys)`** | 算子 + CUDA + 通信 + 内存的时间线，**大模型 profiling 事实标准** |
| `nvidia-smi --query-gpu=...` | 脚本化采集指标 |

**InferSim 项目关联**：本项目建模 TTFT/TPOT/TGS 就是性能分析的上游——**理解一层层算子耗时来源**，本讲的思路直接适用。

---

## 8. 小结

- [x] Debugging = 缩小不确定性，不是瞎改
- [x] `print`/`logging`/`pdb`/`ipdb` 四层阶梯，按复杂度选择
- [x] 静态检查工具（ruff/mypy）提前挡掉一半 bug
- [x] `time`、`cProfile`、`py-spy`、`line_profiler` 的分工
- [x] 深度学习场景用 `torch.profiler` 和 `nsys`
