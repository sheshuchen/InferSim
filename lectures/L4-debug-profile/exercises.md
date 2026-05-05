# L4 · 调试与性能分析 —— 习题解答

---

## 习题 1：用 pdb 定位一个计算错误

**题目**：下面的函数想计算列表平均值，但结果总是少 1，请用 `breakpoint()` 定位 bug：

```python
def average(xs):
    total = 0
    for x in xs:
        total += x
    return total / (len(xs) + 1)   # bug 在这里
```

**解答**：
```python
def average(xs):
    total = 0
    for x in xs:
        total += x
    breakpoint()          # 停下来
    return total / (len(xs) + 1)

print(average([1, 2, 3, 4, 5]))
```

进入 pdb 后：
```
(Pdb) p total
15
(Pdb) p len(xs)
5
(Pdb) p total / len(xs)
3.0
(Pdb) p total / (len(xs) + 1)
2.5              # 错误值！
```

定位到分母多加了 1，改回 `len(xs)` 即可。

---

## 习题 2：事后调试一个崩溃脚本

**题目**：写一段会除零的代码，不加断点，只在崩溃后进入 pdb 查看崩溃现场变量。

**解答**：
```python
# crash.py
def divide(a, b):
    return a / b

def main():
    x = 10
    y = 0
    print(divide(x, y))

main()
```

运行方式：
```bash
python3 -m pdb -c continue crash.py
# 崩溃后自动停在异常处
(Pdb) w                  # 查看调用栈
(Pdb) p a, b             # (10, 0)
(Pdb) u                  # 上一层栈
(Pdb) p x, y             # (10, 0) 看清来源
```

---

## 习题 3：用 logging 替换 print

**题目**：给下面的脚本加上 logging：
- INFO 级别打印每次训练步的 loss
- WARNING 级别在 loss 是 NaN 时打印
- ERROR 级别在异常时打印并包含 traceback

**解答**：
```python
import logging
import math

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")

def train_step(step, loss):
    try:
        if math.isnan(loss):
            logger.warning("loss 为 NaN，step=%d", step)
            return
        logger.info("step=%d loss=%.4f", step, loss)
    except Exception:
        logger.error("训练异常", exc_info=True)

train_step(1, 0.5)
train_step(2, float("nan"))
```

控制生产/调试：
```python
logger.setLevel(logging.WARNING)      # 上线后只看 WARNING 以上
```

---

## 习题 4：用 time 测量 shell 命令

**题目**：对比 `grep` 和 `rg` 在同一次大目录搜索中的耗时。

**解答**：
```bash
# 找一个稍大的目录（用 /usr 或项目根）
time grep -r "def " /usr/lib/python3 >/dev/null
time rg "def " /usr/lib/python3 >/dev/null
```

典型观察：
- `rg` 的 real 时间往往是 `grep` 的 1/5 ~ 1/10
- `user` 时间说明 rg 有多线程并行加速

---

## 习题 5：cProfile 定位慢函数

**题目**：下面脚本哪个函数最慢？用 cProfile 找出来。

```python
# slow.py
import time

def small():
    time.sleep(0.01)

def medium():
    for _ in range(5):
        small()

def big():
    for _ in range(10):
        medium()

def main():
    big()

main()
```

**解答**：
```bash
python3 -m cProfile -s cumulative slow.py | head -20
```

观察输出：`big` 的 `cumtime ≈ 0.5 s`、`medium` 的 `cumtime ≈ 0.05 s`、`small` 的 `cumtime ≈ 0.01 s`。**cumtime 是"自己 + 所有子函数"**，big 最大说明整个耗时都花在它触发的调用链上。

可视化：
```bash
python3 -m cProfile -o out.prof slow.py
snakeviz out.prof               # 浏览器会打开火焰图
```

---

## 习题 6：py-spy 看正在跑的进程

**题目**：启动一个死循环脚本，在另一个终端用 py-spy 查看它耗时集中在哪。

**解答**：
```python
# busy.py
def hot():
    s = 0
    for i in range(10**8):
        s += i * i
    return s

while True:
    hot()
```

```bash
# 终端 1
python3 busy.py &
echo $!                      # 记下 PID，比如 12345

# 终端 2
py-spy top --pid 12345       # 实时 top 风格
# 或保存 SVG 火焰图
py-spy record -o flame.svg --pid 12345 --duration 10
open flame.svg
```

输出会清楚显示 `hot` 函数占了几乎 100%。

---

## 习题 7：line_profiler 行级分析

**题目**：对下面函数，找出哪一行最耗时。

```python
def compute(xs):
    y1 = [x * 2 for x in xs]          # a
    y2 = [x ** 0.5 for x in xs]        # b
    y3 = [str(x) for x in xs]          # c
    return y1, y2, y3
```

**解答**：
```python
# my.py
@profile                         # 由 kernprof 注入
def compute(xs):
    y1 = [x * 2 for x in xs]
    y2 = [x ** 0.5 for x in xs]
    y3 = [str(x) for x in xs]
    return y1, y2, y3

compute(list(range(10_000_000)))
```

```bash
pip install line_profiler
kernprof -l -v my.py
```

输出会显示每一行的 `% Time`。经验上 `str(x)` 最慢（每次都要构造字符串对象），其次是 `x**0.5`（浮点运算 + 类型转换），`x*2` 最快。

---

## 习题 8：用 strace/dtruss 观察系统调用

**题目**：跑 `ls /tmp`，看它到底向内核发了哪些系统调用。

**解答**：
```bash
# Linux
strace -c ls /tmp               # -c = 只给总计
strace -e openat,read ls /tmp   # 只看 openat 和 read

# macOS
sudo dtruss -f ls /tmp
# 或更轻量：
dtrace -qn 'syscall::open*:entry /execname=="ls"/ { printf("%s\n", copyinstr(arg0)); }'
```

会看到 `ls` 打开了 `/tmp`，读目录 entry，打开了 `/etc/localtime`（时间格式化）等。这种工具是排查"为什么这条命令慢/失败"的终极大招。

---

## 习题 9：用 torch.profiler 分析一次前向

**题目**：对一段 PyTorch 代码做一次 profiling，输出 Chrome trace。

**解答**：
```python
import torch
from torch.profiler import profile, ProfilerActivity, record_function

x = torch.randn(1024, 1024, device="cuda")
w = torch.randn(1024, 1024, device="cuda")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    with record_function("matmul_block"):
        for _ in range(10):
            y = x @ w
            y.relu_()

# 表格汇总
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# 导出 Chrome trace，在 chrome://tracing 打开
prof.export_chrome_trace("trace.json")
```

在 `chrome://tracing` 里能看到 CPU-GPU 时间线，清晰看到每个算子占多久。大模型训练/推理优化的第一步永远是**生成这张时间线**。

---

## 习题 10：综合——调试 + profile 一个真实脚本

**题目**：写一个脚本 `analyze.py`，它会：
1. 从命令行读取一个日志文件路径
2. 按时间戳统计每分钟的请求数
3. 打印 Top 10 最忙的分钟

先跑起来，发现慢或错，用本讲工具调试 + 优化。

**解答（第一版，朴素实现）**：
```python
# analyze.py
import sys
from collections import Counter

def parse(line):
    # 假设日志格式：2026-05-03 12:34:56 GET /api ...
    ts = line[:16]         # 截到分钟
    return ts

def main(path):
    counts = Counter()
    with open(path) as f:
        for line in f:
            counts[parse(line)] += 1
    for ts, n in counts.most_common(10):
        print(ts, n)

if __name__ == "__main__":
    main(sys.argv[1])
```

**调试思路**：
1. 先用小文件测试，`python3 analyze.py small.log`，`print` 或 `breakpoint()` 查看 `counts` 的中间状态
2. 大文件慢了，用 `python3 -m cProfile analyze.py big.log | head -20` 看瓶颈
3. 如果卡在 `open`/`read` → IO 瓶颈，考虑 `mmap` 或分块读
4. 如果卡在 `parse` → 换更快的解析方式（正则、切片、`str.split`）
5. 用 `line_profiler` 看哪行最慢，针对性优化

**优化后常见结论**：
- 朴素字符串切片比正则快 3-5 倍
- 避免每行 `datetime.strptime`，能用字符串切片就用字符串切片
- 对于 GB 级日志，`rg` + `sort` + `uniq -c` 组合可能比 Python 还快

---

## 本讲学习自检

- [ ] 能说出 Debugging 的五步方法论
- [ ] 遇到小 bug 会用 `breakpoint()`，遇到大问题会用 `pdb/ipdb`
- [ ] 在项目里已经把 `print` 换成 `logging`
- [ ] 熟悉 `time` 的 real/user/sys 三个值含义
- [ ] 会用 `cProfile` + `snakeviz` 做一次 profiling
- [ ] 知道 `py-spy` 可以贴在生产进程上
- [ ] 至少用过一次 `torch.profiler` 导出 Chrome trace
