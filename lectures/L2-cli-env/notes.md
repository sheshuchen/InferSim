# L2 · 命令行环境

> **原讲义链接**：https://missing.csail.mit.edu/2026/
> **本讲主题**：Command-line Environment
> **本文性质**：基于本讲主题编写的中文学习笔记（原创内容）

---

## 本讲要解决的问题

L1 让你学会了敲命令，但真实工作里你会立刻遇到：

- 一个命令要跑一小时，但我想让它在后台跑，不占终端
- 我误按了 Ctrl+C，怎么再让它继续？
- SSH 断线后，远程的长任务就全没了，怎么办？
- 每台新机器都要重新配置 `.zshrc`、git 别名……太痛苦了
- 有没有不用每次敲密码就能登录服务器的办法？

这些都归属于"**命令行环境**"要解决的问题。

---

## 1. 任务控制（Job Control）

### 1.1 信号（Signals）

信号是操作系统通知进程"发生了某件事"的机制。最常用的几个：

| 信号 | 编号 | 含义 | 如何触发 |
|------|------|------|----------|
| `SIGINT`  | 2  | 打断（请求优雅退出） | `Ctrl+C` |
| `SIGTSTP` | 20 | 暂停（进程冻结但不退出） | `Ctrl+Z` |
| `SIGTERM` | 15 | 请求终止（可被程序捕获清理后退出） | `kill PID` |
| `SIGKILL` | 9  | 强制终止（不可被捕获） | `kill -9 PID` |
| `SIGHUP`  | 1  | 终端挂断（终端关闭时发出） | 关终端窗口 |

**经验**：先 `kill PID` 给机会清理；实在不走才用 `kill -9`。

### 1.2 前台、后台与 `jobs`

```bash
sleep 1000              # 前台运行，占用终端
# 按 Ctrl+Z 暂停后：
bg                      # 把最近暂停的作业放到后台继续
jobs                    # 查看当前 shell 的作业列表
fg %1                   # 把 1 号作业拉回前台
kill %1                 # 杀掉 1 号作业

sleep 1000 &            # 直接在后台启动（& 符号）
```

`%1`、`%2` 是作业编号；`$!` 是刚刚后台启动的那个进程的 PID。

### 1.3 防止终端关闭时程序被杀

- `nohup command &` —— 忽略 SIGHUP
- `disown %1` —— 把作业从当前 shell"脱钩"
- 更彻底的办法：用 **tmux/screen**（见下文）

---

## 2. 终端多路复用（Terminal Multiplexer）

**核心需求**：在一个 SSH 会话里同时开多个终端；网络断了也不丢失会话。

**tmux** 是事实标准。三层概念：

| 层 | 说明 |
|----|------|
| **Session（会话）** | 最大单位。断网重连后原样恢复。 |
| **Window（窗口）** | 一个 session 里的多个"标签页"。 |
| **Pane（窗格）** | 一个 window 里的分屏区域。 |

### tmux 常用命令

```bash
tmux                          # 启动一个新 session
tmux new -s work              # 新建名为 work 的 session
tmux ls                       # 列出所有 session
tmux attach -t work           # 重连到 work session
tmux kill-session -t work
```

**会话内快捷键**（前缀 `Ctrl+B`，按完松手再按下一个键）：

| 动作 | 组合键 |
|------|--------|
| 从 session 中脱离（不关闭） | `Ctrl+B  d` |
| 新建 window                  | `Ctrl+B  c` |
| 切换到下/上一个 window        | `Ctrl+B  n` / `p` |
| 按编号跳到 window             | `Ctrl+B  0`-`9` |
| 竖直分屏                      | `Ctrl+B  %` |
| 水平分屏                      | `Ctrl+B  "` |
| 在 pane 之间移动              | `Ctrl+B  方向键` |
| 关闭当前 pane                 | `Ctrl+B  x` |

**典型工作流**：SSH 登入服务器后立刻 `tmux new -s work`，断网也不怕；下次 `tmux attach -t work` 一秒恢复。

---

## 3. 别名与函数（Aliases & Functions）

### 3.1 Alias：命令缩写

```bash
alias ll='ls -alh'
alias gs='git status'
alias ..='cd ..'
alias gr='cd $(git rev-parse --show-toplevel)'   # 跳到仓库根
```

**规则**：
- alias 只是**字符串替换**，不支持参数传递
- 写在 `~/.zshrc` 或 `~/.bashrc` 里永久生效
- 改完配置后 `source ~/.zshrc` 重新加载

### 3.2 Function：需要参数时用函数

```bash
# 创建目录并进入
mcd() {
    mkdir -p "$1" && cd "$1"
}

# 找占用某端口的进程
port() {
    lsof -nP -iTCP:"$1" -sTCP:LISTEN
}
```

- `$1`、`$2` 是第 1、2 个参数
- `$@` 是所有参数
- 定义在 `~/.zshrc` 里也可以永久生效

---

## 4. dotfiles 管理

**dotfiles** 指以 `.` 开头的配置文件：`.zshrc`、`.gitconfig`、`.vimrc`、`.tmux.conf`、`.ssh/config`……

**常见做法**：把它们放进一个 git 仓库，各台机器 `clone` 下来 + 建软链接。

### 推荐结构

```
~/dotfiles/
├── zsh/.zshrc
├── git/.gitconfig
├── vim/.vimrc
├── tmux/.tmux.conf
└── install.sh
```

`install.sh` 的核心动作：
```bash
#!/bin/bash
DOT=~/dotfiles
ln -sf $DOT/zsh/.zshrc ~/.zshrc
ln -sf $DOT/git/.gitconfig ~/.gitconfig
ln -sf $DOT/vim/.vimrc ~/.vimrc
ln -sf $DOT/tmux/.tmux.conf ~/.tmux.conf
```

**进阶工具**：
- `stow`：自动创建软链接，不用手写 install.sh
- `chezmoi`：支持模板（不同机器同一份配置里可以有差异）

---

## 5. SSH：远程登录

### 5.1 基本用法

```bash
ssh user@host                        # 登录
ssh user@host 'uname -a'             # 远程执行单条命令
scp file.txt user@host:/path/        # 拷贝文件过去
scp user@host:/path/file.txt .       # 拷贝文件回来
rsync -avz src/ user@host:/dst/      # 增量同步（更好）
```

### 5.2 免密登录：SSH 密钥

1. 本机生成密钥对（如果没有）：
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
# 默认会生成 ~/.ssh/id_ed25519（私钥）+ id_ed25519.pub（公钥）
```

2. 把公钥拷贝到服务器：
```bash
ssh-copy-id user@host
# 等价于：cat ~/.ssh/id_ed25519.pub | ssh user@host 'cat >> ~/.ssh/authorized_keys'
```

3. 再次 `ssh user@host`，就无需密码了。

**安全要点**：
- 私钥永远不外泄
- `~/.ssh` 权限必须是 700，私钥权限 600，否则 SSH 会拒绝用它

### 5.3 ~/.ssh/config：给常用主机取小名

```
Host dev
    HostName 10.0.0.5
    User ubuntu
    Port 22
    IdentityFile ~/.ssh/id_ed25519

Host gpu-*
    User sheshuchen
    ProxyJump bastion
```

之后：`ssh dev` 就等价于一长串命令；`ssh gpu-01` 会自动先跳板 `bastion`。

### 5.4 端口转发（Port Forwarding）

```bash
# 本地转发：把本机 8080 的请求转发到远端 localhost:80
ssh -L 8080:localhost:80 user@host

# 远程转发：反向，把远端 9000 转发到本机 3000
ssh -R 9000:localhost:3000 user@host
```

典型场景：服务器上有个 Jupyter 跑在 8888 端口，你本机浏览器想打开：
```bash
ssh -L 8888:localhost:8888 user@gpu-server
# 之后本机访问 http://localhost:8888 就能看到远程的 Jupyter
```

---

## 6. Shell 脚本速览

### 变量、条件、循环

```bash
#!/usr/bin/env bash
set -euo pipefail                  # 强烈推荐：遇错即停 / 未定义变量报错 / 管道错传播

name="world"                       # 赋值时等号两边不能有空格！
echo "hello, $name"

# 条件
if [[ -f "$1" ]]; then
    echo "文件存在"
elif [[ -d "$1" ]]; then
    echo "是目录"
else
    echo "不存在"
fi

# 循环
for f in *.log; do
    echo "处理 $f"
done

# while
n=0
while (( n < 5 )); do
    echo $n
    (( n++ ))
done
```

### 常用测试运算符 `[[ ]]`

| 表达式 | 含义 |
|--------|------|
| `-e PATH` | 存在（任何类型） |
| `-f PATH` | 是普通文件 |
| `-d PATH` | 是目录 |
| `-z STR`  | 字符串为空 |
| `-n STR`  | 字符串非空 |
| `STR1 == STR2` | 字符串相等（支持通配符） |
| `N1 -eq N2` | 数字相等（还有 -ne/-gt/-lt/-ge/-le） |

### 命令替换与算术

```bash
now=$(date +%F)                    # 命令替换，$(...) 推荐写法
echo "今天是 $now"

sum=$((3 + 5))                     # 算术
echo $sum                          # 8
```

---

## 7. 小结

- [x] 知道 `Ctrl+C`/`Ctrl+Z`、`bg`/`fg`/`jobs` 的区别
- [x] 会用 tmux 新建 / 脱离 / 重连 session
- [x] 会写 alias 和简单 function 放进 .zshrc
- [x] 能用 `ssh-keygen` + `ssh-copy-id` 配置免密登录
- [x] 会写 `~/.ssh/config` 简化主机连接
- [x] 能用 `ssh -L` 做本地端口转发调试远程服务
- [x] 会写简单 shell 脚本（变量、条件、循环）
- [x] 知道 `set -euo pipefail` 是脚本安全网
