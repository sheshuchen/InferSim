# L3 · 开发环境与工具 —— 习题解答

---

## 习题 1：Vim 十分钟肌肉记忆训练

**题目**：打开任意一个 `.py` 文件，不用鼠标完成：
1. 跳到第 50 行
2. 把光标下的单词删掉
3. 复制当前行，粘贴到第 100 行之前
4. 把全文所有 `foo` 替换为 `bar`
5. 保存退出

**解答**：
```
vim demo.py

50G             # 跳到第 50 行
dw              # 删除光标下一个单词
yy              # 复制当前行
99G             # 到第 99 行（粘贴到之后 = 100 行前）
p               # 粘贴

:%s/foo/bar/g   # 全文替换
:wq             # 保存退出
```

---

## 习题 2：自定义 .vimrc

**题目**：创建 `~/.vimrc`，启用：行号、相对行号、缩进 4 空格、搜索高亮、忽略大小写但有大写时精确。

**解答**：
```vim
" ~/.vimrc
set number
set relativenumber
set expandtab
set tabstop=4 shiftwidth=4 softtabstop=4
set hlsearch incsearch
set ignorecase smartcase
set cursorline
set clipboard=unnamed
syntax on
```

重启 vim 生效。建议每次新机器配好这个文件放进 dotfiles 仓库。

---

## 习题 3：用 find 做批量处理

**题目**：
1. 找出当前项目下所有超过 1MB 的文件
2. 找出最近 24 小时内修改过的 `.py` 文件
3. 找出所有 `.log` 文件并删除（做前先 `ls` 确认）

**解答**：
```bash
# 1. 大于 1MB
find . -type f -size +1M

# 2. 最近 24h 修改的 .py
find . -type f -name "*.py" -mtime 0

# 3. 先确认再删
find . -type f -name "*.log"            # 先看清楚
find . -type f -name "*.log" -delete    # 确认无误再删

# 更安全的写法：用 -i 让 rm 每个问一次
find . -type f -name "*.log" -exec rm -i {} \;
```

---

## 习题 4：用 rg 代替 grep

**题目**：在 InferSim 项目里：
1. 搜所有出现 `num_experts` 的 `.py` 文件
2. 搜所有 TODO 注释（带上下文 2 行）
3. 只看出现 `TODO` 的文件名，不要具体内容

**解答**：
```bash
cd ~/Desktop/InferSim

rg -t py "num_experts"
rg -C 2 "TODO"
rg -l "TODO"              # -l = list files only
```

对比：`grep -r --include='*.py' "num_experts" .` 写起来麻烦得多，rg 一个 `-t py` 搞定。

---

## 习题 5：Python 虚拟环境

**题目**：为当前项目新建一个干净的虚拟环境，装上 `numpy` 和 `pandas`，导出依赖清单，然后在另一台机器上精确复现。

**解答**：
```bash
# 当前机器
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas
pip freeze > requirements.txt
deactivate

# 另一台机器
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**要点**：
- `requirements.txt` 一定要放进 git
- 不要把 `.venv/` 提交（放进 `.gitignore`）

**进阶**（用 uv 更快）：
```bash
uv venv
source .venv/bin/activate
uv pip install numpy pandas
uv pip freeze > requirements.txt
```

---

## 习题 6：zoxide 配置

**题目**：安装 zoxide，配合 fzf 达到"按 zi 模糊跳任何目录"。

**解答**：
```bash
# macOS
brew install zoxide fzf

# 在 ~/.zshrc 末尾加
eval "$(zoxide init zsh)"

# 启用 fzf 的交互模式（zi 会调用 fzf）
# 先重新加载
source ~/.zshrc

# 使用
cd ~/Desktop/InferSim       # zoxide 记录
cd /tmp                     # zoxide 记录
z infersim                  # 模糊跳回
zi                          # 不输入参数，弹出 fzf 列表让你选
```

---

## 习题 7：VSCode Remote-SSH 开发

**题目**：描述完整流程——本机 macOS 上的 VSCode 打开 GPU 服务器上的项目，像在本地一样编辑。

**解答**：
1. 本机安装 VSCode + Remote-SSH 插件
2. 配好 `~/.ssh/config`（参考 L2 习题 5），确认 `ssh gpu` 能免密登录
3. VSCode 左下角绿色角标 → "Connect to Host" → 选 `gpu`
4. 首次连接会在远端自动安装 VSCode Server（约 100MB）
5. "Open Folder" 选服务器上的项目路径
6. 所有插件、终端、调试器都运行在**远端**，本机只负责 UI

**优点**：
- 本机不用装 CUDA、不用装各种依赖
- 多设备切换无缝，项目在服务器上是唯一事实源
- VSCode 终端直接就是服务器的终端，无需另开 iTerm

---

## 习题 8：综合——为新机器快速搭好开发环境

**题目**：写一个 `bootstrap.sh`，在新 macOS 上一键完成：
- 安装 Homebrew
- 装 git、neovim、tmux、ripgrep、fd、fzf、zoxide、uv
- 克隆个人 dotfiles 仓库并软链接配置

**解答**：
```bash
#!/usr/bin/env bash
set -euo pipefail

# 1. Homebrew
if ! command -v brew >/dev/null; then
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# 2. CLI 工具
brew install git neovim tmux ripgrep fd fzf zoxide
brew install uv                 # 若 tap 里有

# 3. dotfiles（假设你仓库在 github.com/you/dotfiles）
DOT=~/dotfiles
if [[ ! -d "$DOT" ]]; then
    git clone git@github.com:you/dotfiles.git "$DOT"
fi

# 4. 软链接
ln -sf "$DOT/zsh/.zshrc"       ~/.zshrc
ln -sf "$DOT/git/.gitconfig"   ~/.gitconfig
ln -sf "$DOT/vim/.vimrc"       ~/.vimrc
ln -sf "$DOT/tmux/.tmux.conf"  ~/.tmux.conf
mkdir -p ~/.config/nvim
ln -sf "$DOT/nvim/init.vim"    ~/.config/nvim/init.vim

echo "完成！重启终端或执行 source ~/.zshrc"
```

使用：
```bash
curl -fsSL https://raw.githubusercontent.com/you/dotfiles/main/bootstrap.sh | bash
```

---

## 本讲学习自检

- [ ] 盲敲 Vim 完成 5 项基本编辑操作无需翻 cheatsheet
- [ ] 有自己的 `.vimrc` 放在 dotfiles 仓库
- [ ] 写出 3 条 `find`/`fd` 命令应对文件查找
- [ ] `rg` 已经代替 `grep` 成为日常搜索工具
- [ ] 每个 Python 项目都有自己的 `.venv`，没有全局乱装包
- [ ] 知道 Node/Python 版本管理如何做
- [ ] 至少用过一次 VSCode Remote-SSH
