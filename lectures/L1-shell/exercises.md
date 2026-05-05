# L1 · Shell 入门 —— 习题解答

> 本讲习题围绕 Shell 基础命令的实际使用。题目参考了 MIT Missing Semester 历年 Shell 讲次的经典练习形式，结合 2026 版讲义主题编排。所有解答都在 **macOS Zsh / Linux Bash** 环境下验证过思路。

---

## 习题 1：熟悉 man 与 --help

**题目**：选一个你不熟悉的命令（例如 `ls`、`echo`、`date`），用 `man` 和 `--help` 两种方式查看其帮助，对比两者信息密度的差异。

**解答**：
```bash
man ls            # 完整手册（长，适合深入查）
ls --help         # 简短帮助（短，适合快速回忆参数）
```

**对比总结**：
- `man` 给出类别、语法、每个参数的详细说明、示例、相关文件等，信息密度最高但查起来慢
- `--help` 输出几十行，直接列参数含义，适合"我只是忘了 `-l` 和 `-a` 怎么写"的场景
- 日常用 `--help`，深入研究用 `man`，找例子用 `tldr`

---

## 习题 2：理解工作目录与路径

**题目**：
1. 在家目录下执行 `pwd`，记录输出
2. 用 `cd /tmp`，再执行 `pwd`
3. 用 `cd -` 回到之前的目录
4. 只用一条 `cd` 命令，从 `/tmp` 跳到 `/usr/local/bin`

**解答**：
```bash
cd ~
pwd                       # /Users/sheshuchen（或 /home/xxx）
cd /tmp
pwd                       # /tmp
cd -                      # 回家目录
cd /usr/local/bin         # 从 /tmp 一步到位
pwd                       # /usr/local/bin
```

关键点：`cd -` 在"两个常用目录间来回切"时非常高效。

---

## 习题 3：用 ls 探索文件属性

**题目**：
1. 列出 `/usr/bin` 下的所有文件，按大小由大到小排序
2. 统计该目录下有多少文件
3. 只显示目录下以 `p` 开头的文件

**解答**：
```bash
ls -lhS /usr/bin                   # -S 按大小排序，-h 人类可读，-l 详细
ls /usr/bin | wc -l                # 统计行数=文件数
ls /usr/bin/p*                     # 通配符 * 展开
```

扩展：
```bash
ls -lh /usr/bin | head -20         # 只看前 20 行
ls /usr/bin | grep '^p'            # 用正则风格，也能过滤开头
```

---

## 习题 4：创建/移动/删除文件

**题目**：在 `/tmp` 下做这样一套操作：
1. 创建一个目录 `playground`
2. 在其中创建一个空文件 `hello.txt`
3. 写入内容 "hi shell"
4. 复制为 `hello_bak.txt`
5. 把 `hello.txt` 改名为 `greeting.txt`
6. 删除整个 `playground` 目录

**解答**：
```bash
cd /tmp
mkdir playground
cd playground
touch hello.txt
echo "hi shell" > hello.txt       # 重定向写入
cat hello.txt                      # 验证：hi shell
cp hello.txt hello_bak.txt
mv hello.txt greeting.txt
ls                                 # greeting.txt  hello_bak.txt
cd ..
rm -r playground
```

⚠️ `rm -r` 不可恢复，生产环境要先 `ls` 核对再执行。

---

## 习题 5：读懂权限位

**题目**：对 `/tmp` 执行 `ls -ld /tmp`，解读每一位权限的含义。

**解答**：
```bash
ls -ld /tmp
# 典型输出：drwxrwxrwt  10 root  wheel  320 Oct 15 10:22 /tmp
```

| 位 | 含义 |
|----|------|
| `d` | 这是个目录 |
| `rwx`（2-4）| 所有者（root）：读、写、执行（进入） |
| `rwx`（5-7）| 所属组（wheel）：读、写、执行 |
| `rwt`（8-10）| 其他用户：读、写、执行 + **粘滞位（sticky bit）** |

**粘滞位 `t`** 的作用：即使目录 world-writable，里面的文件也**只能由文件所有者删除**。这是 `/tmp` 允许所有人写但又互不干扰的关键机制。

---

## 习题 6：重定向与流

**题目**：
1. 把 `ls /` 的输出保存到 `~/root_listing.txt`
2. 把一个必然失败的命令的错误信息单独保存（不要把正常输出混进来）
3. 把一个命令的正常输出和错误输出都合并到同一个文件

**解答**：
```bash
# 1. stdout 写入文件
ls / > ~/root_listing.txt
cat ~/root_listing.txt

# 2. 只捕获 stderr
ls /nonexistent 2> ~/err.txt
cat ~/err.txt
# 输出类似：ls: /nonexistent: No such file or directory

# 3. stdout + stderr 一起
ls / /nonexistent > ~/all.txt 2>&1
# 或 bash 专用简写：
ls / /nonexistent &> ~/all.txt
```

要点：`2>&1` 的意思是"把 fd 2 复制到 fd 1 当前所指的位置"，**顺序很关键**，必须先写 `> file` 再写 `2>&1`。

---

## 习题 7：管道组合

**题目**：用一条管道命令，统计 `/etc/passwd` 里一共有多少个用户使用 bash 作为登录 shell。

**解答**：
```bash
cat /etc/passwd | grep '/bin/bash$' | wc -l
# 或更"Unix 正道"写法（grep 自己能读文件）：
grep -c '/bin/bash$' /etc/passwd
```

拆解：
1. `cat /etc/passwd` → 输出所有行
2. `grep '/bin/bash$'` → 只留以 `/bin/bash` 结尾的行（登录 shell 字段）
3. `wc -l` → 数行数

---

## 习题 8：sudo 与重定向的经典坑

**题目**：解释为什么 `sudo echo hello > /etc/privileged.txt` 在没有写权限时仍会失败，给出正确写法。

**解答**：

**原因**：Shell 在执行命令前**先**解析重定向 `> /etc/privileged.txt`，此时文件打开动作是 **shell（以你的身份）** 完成的，而不是 sudo。所以权限检查用的是你自己的身份，失败。

**正确写法**（二选一）：
```bash
# 方法 1：用 tee
echo hello | sudo tee /etc/privileged.txt
echo more  | sudo tee -a /etc/privileged.txt    # 追加

# 方法 2：开一个 root shell 再执行整条命令
sudo sh -c 'echo hello > /etc/privileged.txt'
```

---

## 习题 9：历史命令与快捷键

**题目**：
1. 查看最近 10 条命令
2. 不重新打字，执行刚刚运行的 `ls -lh /usr/bin`
3. 不重新打字，执行最近一条以 `git` 开头的命令

**解答**：
```bash
history | tail -10
!!                    # 上一条
!ls                   # 最近一条以 ls 开头
!git                  # 最近一条以 git 开头
# 也可以 Ctrl+R 进入反向搜索，输入关键字即可
```

---

## 习题 10：综合小挑战

**题目**：写一条命令，找出 `/usr/bin` 下体积最大的那个可执行文件，并打印它的完整路径和大小。

**解答**：
```bash
ls -lS /usr/bin | head -2
# head -2 的原因：第一行是 "total ..."，第二行才是最大的文件
```

更精确的写法：
```bash
ls -lS /usr/bin | awk 'NR==2 {print $NF, $5}'
# NR==2：第二行
# $NF：最后一列（文件名）
# $5：第 5 列（大小，-l 格式下）
```

（管道 + awk 会在 L2/L4 中详细展开。）

---

## 本讲学习自检

做完这 10 题后，你应当可以：
- 任意敲出 `man`、`cd`、`ls`、`pwd`、`cp`、`mv`、`rm`、`mkdir`、`touch`
- 解释 `drwxr-xr-x` 每一位的含义
- 熟练使用 `>`、`>>`、`<`、`2>`、`2>&1`、`|`
- 知道为什么 `sudo command > /file` 会失败
- Tab 补全、`Ctrl+R`、`!!`、`!前缀` 这四个加速技巧都用过了

若任何一条还不熟练，请回到 [notes.md](./notes.md) 对应章节重读。
