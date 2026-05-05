# L2 · 命令行环境 —— 习题解答

> 本讲习题围绕任务控制、tmux、alias、SSH、shell 脚本设计。

---

## 习题 1：用信号打断与恢复一个长任务

**题目**：运行 `sleep 1000`，然后：
1. 用 Ctrl+Z 暂停它
2. 查看当前 shell 有哪些作业
3. 把它放到后台继续
4. 再把它拉回前台
5. 最终用信号优雅地杀掉它

**解答**：
```bash
sleep 1000                  # 前台启动
# 按 Ctrl+Z
#   [1]+  Stopped   sleep 1000

jobs                        # [1]+  Stopped    sleep 1000
bg %1                       # [1]+  sleep 1000 &
jobs                        # [1]+  Running   sleep 1000 &
fg %1                       # 拉回前台
# 按 Ctrl+C 发送 SIGINT
```

扩展：如果想用 PID 杀：
```bash
sleep 1000 &
echo $!                     # 输出刚启动进程的 PID
kill <PID>                  # SIGTERM
kill -9 <PID>               # SIGKILL（迫不得已才用）
```

---

## 习题 2：拦截信号做清理动作

**题目**：写一个脚本，运行后每秒打印一次数字；当用户按 Ctrl+C 时不立即退出，而是打印 "bye" 再退出。

**解答**（`trap.sh`）：
```bash
#!/usr/bin/env bash
set -eu

cleanup() {
    echo
    echo "bye"
    exit 0
}

trap cleanup INT            # 捕获 SIGINT（Ctrl+C）

n=0
while true; do
    echo $n
    n=$((n+1))
    sleep 1
done
```

运行：
```bash
chmod +x trap.sh
./trap.sh
# 按 Ctrl+C 会看到 "bye" 再退出
```

`trap 动作 信号` 是 shell 脚本里处理信号的标准方式。

---

## 习题 3：tmux 基本工作流

**题目**：
1. 启动一个叫 `demo` 的 tmux session
2. 在里面分出左右两个 pane
3. 左边运行 `htop`（没有就用 `top`），右边运行 `tail -f /var/log/system.log`（macOS）或任意日志
4. 从 session 脱离
5. 重连回去

**解答**：
```bash
tmux new -s demo
# 进入 tmux 后：
#   Ctrl+B  %        ← 垂直切一刀
#   左 pane：htop
#   Ctrl+B  左右箭头  ← 切到右 pane
#   右 pane：tail -f ~/.zsh_history   # 任何会持续更新的文件都行
#   Ctrl+B  d        ← 脱离

# 回到普通 shell
tmux ls                     # demo: 1 windows (...)
tmux attach -t demo         # 一切原样恢复
```

---

## 习题 4：写几个最有用的 alias 和 function

**题目**：在 `~/.zshrc` 里加入：
1. `ll` = `ls -alh`
2. `..` = `cd ..`，`...` = `cd ../..`
3. `mcd <目录>`：创建并进入该目录
4. `extract <文件>`：根据扩展名自动选择合适的解压命令

**解答**（追加到 `~/.zshrc`）：
```bash
# --- aliases ---
alias ll='ls -alh'
alias ..='cd ..'
alias ...='cd ../..'
alias ....='cd ../../..'

# --- functions ---
mcd() {
    [[ -z "$1" ]] && { echo "用法: mcd <dir>"; return 1; }
    mkdir -p "$1" && cd "$1"
}

extract() {
    [[ -f "$1" ]] || { echo "文件不存在: $1"; return 1; }
    case "$1" in
        *.tar.bz2)  tar xjf "$1"   ;;
        *.tar.gz)   tar xzf "$1"   ;;
        *.tar.xz)   tar xJf "$1"   ;;
        *.tar)      tar xf "$1"    ;;
        *.tbz2)     tar xjf "$1"   ;;
        *.tgz)      tar xzf "$1"   ;;
        *.zip)      unzip "$1"     ;;
        *.gz)       gunzip "$1"    ;;
        *.bz2)      bunzip2 "$1"   ;;
        *)          echo "未知格式: $1" ;;
    esac
}
```

重新加载：`source ~/.zshrc`

---

## 习题 5：SSH 免密登录配置

**题目**：假设有一台远程机器 `gpu.example.com`，用户名是 `sheshuchen`。完成免密登录并通过配置文件给它起名 `gpu`。

**解答**：
```bash
# 1. 生成密钥（若没有）
ssh-keygen -t ed25519 -C "sheshuchen@laptop"

# 2. 拷贝公钥到服务器
ssh-copy-id sheshuchen@gpu.example.com

# 3. 在 ~/.ssh/config 写入
cat >> ~/.ssh/config <<'EOF'
Host gpu
    HostName gpu.example.com
    User sheshuchen
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
    ServerAliveCountMax 3
EOF

# 4. 修正权限（SSH 很挑权限）
chmod 700 ~/.ssh
chmod 600 ~/.ssh/config ~/.ssh/id_ed25519

# 5. 使用
ssh gpu                     # 一秒登录
scp file.txt gpu:/home/sheshuchen/
rsync -avz data/ gpu:/home/sheshuchen/data/
```

`ServerAliveInterval`：每 60 秒发心跳包，避免长连接被中间设备切断。

---

## 习题 6：SSH 端口转发

**题目**：服务器上跑了一个 Jupyter Notebook 绑定在 `localhost:8888`，但你本机没法直接访问。用 SSH 端口转发打开它。

**解答**：
```bash
# 本机执行
ssh -L 8888:localhost:8888 gpu -N
# -N 表示不开交互 shell，只做转发
# 然后本机浏览器访问 http://localhost:8888

# 若本机 8888 被占用，换一个端口
ssh -L 18888:localhost:8888 gpu -N
# 浏览器访问 http://localhost:18888
```

端口转发的本质：在本机 8888 上监听，把所有流量通过 SSH 加密隧道送到远端，再由远端 localhost:8888 响应。

---

## 习题 7：写一个有实际价值的 shell 脚本

**题目**：写 `backup.sh <源目录> <目标目录>`，把源目录按日期打包成 `源目录名-YYYY-MM-DD.tar.gz` 存到目标目录。遇到以下情况必须能报错退出：
- 参数不对
- 源目录不存在
- 目标目录不存在

**解答**：
```bash
#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "用法: $(basename "$0") <源目录> <目标目录>"
    exit 1
}

# 参数校验
[[ $# -ne 2 ]] && usage
src="$1"
dst="$2"
[[ ! -d "$src" ]] && { echo "错误: 源目录不存在: $src"; exit 2; }
[[ ! -d "$dst" ]] && { echo "错误: 目标目录不存在: $dst"; exit 2; }

# 打包
name="$(basename "$(realpath "$src")")"
stamp="$(date +%F)"                       # 2026-05-03
outfile="$dst/${name}-${stamp}.tar.gz"

echo "正在打包 $src -> $outfile"
tar czf "$outfile" -C "$(dirname "$src")" "$name"
echo "完成，大小：$(du -h "$outfile" | cut -f1)"
```

使用：
```bash
chmod +x backup.sh
./backup.sh ~/my_project ~/backups
# -> ~/backups/my_project-2026-05-03.tar.gz
```

**要点**：
- `set -euo pipefail` 防止脚本在错误下继续运行
- `basename` / `dirname` / `realpath` 是路径处理三剑客
- `-C dir file` 让 tar 进到 dir 再打包，归档里不会带绝对路径

---

## 习题 8：综合——远程长任务 + tmux + SSH 配置

**题目**：描述并演示一个完整流程：在本机用 VSCode 写好训练脚本，ssh 到 GPU 服务器运行，中途想断网回家，回家后继续观察训练进度。

**解答**：
```bash
# 本机
rsync -avz --exclude=.git ./project gpu:/home/sheshuchen/
ssh gpu

# 远程
cd /home/sheshuchen/project
tmux new -s train
# 在 tmux 内：
python3 train.py > train.log 2>&1

# 准备断网前：Ctrl+B  d  脱离 session
exit            # 退出 ssh，tmux 里的 train.py 继续跑

# ---- 回家后 ----
ssh gpu
tmux attach -t train
# 直接看到实时输出，可随时脱离
```

要点：
- **tmux 是断网保险**，没它一断网 `train.py` 就被 SIGHUP 杀掉
- 若完全不想用 tmux，最低限度也要 `nohup python3 train.py > train.log 2>&1 &`
- 离线查看进度：`tail -f train.log`

---

## 本讲学习自检

- [ ] 能用 `Ctrl+Z`/`bg`/`fg`/`jobs` 自由切换任务
- [ ] 能在 tmux 里创建 session、分屏、脱离、重连
- [ ] `~/.zshrc` 里至少有 3 个你常用的 alias
- [ ] 已配置 SSH 密钥登录，`~/.ssh/config` 给常用机器起了别名
- [ ] 会用 `ssh -L` 调试远程 Web 服务
- [ ] 能写一个带参数校验、`set -euo pipefail` 的 shell 脚本
