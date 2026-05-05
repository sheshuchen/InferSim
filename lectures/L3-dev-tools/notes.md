# L3 · 开发环境与工具

> **原讲义链接**：https://missing.csail.mit.edu/2026/
> **本讲主题**：Development Environment and Tools
> **本文性质**：基于本讲主题编写的中文学习笔记（原创内容）

---

## 本讲要解决的问题

- 写代码用什么编辑器？Vim 到底怎么用才有生产力？
- 项目里有无数文件、无数行代码，**怎么快速找到**某个函数定义或某段文本？
- Python/Node.js 项目依赖怎么隔离？如何避免"装包装坏了系统"？
- 不同项目的 Python 版本、Node 版本如何切换？

---

## 1. 编辑器：为什么是 Vim 值得学

Vim 有陡峭学习曲线，但它是几乎所有服务器默认自带的编辑器，也是**肌肉记忆一次学会、用一辈子**的工具。

### 1.1 Vim 的模式

| 模式 | 进入 | 作用 |
|------|------|------|
| **Normal**（常规） | `Esc` | 移动、删除、复制、粘贴 |
| **Insert**（插入） | `i` / `a` / `o` | 输入文字（类似普通编辑器） |
| **Visual**（可视） | `v` / `V` / `Ctrl+V` | 选中区域 |
| **Command**（命令行） | `:` | 保存、退出、查找替换等 |

**关键理解**：大部分时间应该待在 **Normal 模式**，只有真的在打字时才进 Insert。

### 1.2 最必要的命令

**移动**（Normal 模式）：
```
h j k l         ← ↓ ↑ →（手不离主键位）
w / b           下/上一个单词
0 / $           行首/行尾
gg / G          文件开头/结尾
{number}G       跳到第 number 行
Ctrl+u / d      向上/下翻半页
```

**编辑**（Normal 模式）：
```
i / a           在光标前/后进入 Insert
o / O           在下/上新开一行并进入 Insert
x               删除一个字符
dd              删除一整行
yy              复制一整行
p               粘贴到光标后
u               撤销
Ctrl+r          重做
.               重复上一次操作（非常强大）
```

**组合动作**（"动词 + 作用范围"）：
```
dw              删除一个单词（delete word）
d$              删除到行尾
di"             删除双引号内部（delete inside "）
ci(             改写括号内部（change inside ()
y3j             复制当前行 + 下 3 行
```

**命令行模式**（`:` 开头）：
```
:w              保存
:q              退出
:wq  /  :x      保存并退出
:q!             不保存强制退出
:%s/old/new/g   全文替换 old -> new
:5,20s/a/b/g    5-20 行替换
```

**查找**：`/pattern` 向下找，`?pattern` 向上找，`n`/`N` 下/上一个。

### 1.3 让 Vim 真正好用

配置 `~/.vimrc`（或 Neovim 的 `~/.config/nvim/init.vim`）：

```vim
syntax on                     " 语法高亮
set number                    " 显示行号
set relativenumber            " 相对行号（配合 5j/10k 极快跳）
set expandtab
set tabstop=4 shiftwidth=4    " Tab 宽 4 空格
set hlsearch incsearch        " 搜索高亮+边打边搜
set ignorecase smartcase      " 默认忽略大小写，有大写字母就区分
set clipboard=unnamed         " 和系统剪贴板共享
```

**进阶**：装个 Neovim + LazyVim / NvChad 配置包，就能获得 LSP（跳转到定义、自动补全、重构）体验。

### 1.4 编辑器选择建议

| 场景 | 推荐 |
|------|------|
| 本地大型项目、团队协作 | **VSCode / Cursor**（功能全、插件多） |
| 远程服务器临时改文件 | **Vim / Neovim**（几乎所有机器都有） |
| 想长期投资、追求极致键盘流 | **Neovim + Tmux** |

最稳的组合：**VSCode 远程 SSH 插件 + 服务器上装 Neovim 兜底**。

---

## 2. 查找：find、fd、grep、rg

### 2.1 `find`：按文件名/属性查找

```bash
find . -name "*.py"                    # 当前目录下所有 .py 文件
find . -type d -name "test"            # 叫 test 的目录
find . -size +10M                      # 大于 10MB 的文件
find . -mtime -7                       # 最近 7 天修改过
find . -name "*.log" -delete           # 找并删除
find . -name "*.py" -exec wc -l {} \;  # 找到后对每个执行命令
```

### 2.2 `fd`：find 的现代替代（更快、更友好）

```bash
fd                         # 列当前目录所有文件（跳过 .gitignore 里的）
fd py                      # 文件名包含 py 的
fd -e py                   # 扩展名 py
fd -H pattern              # 包括隐藏文件
fd pattern /etc            # 在 /etc 下找
```

**默认忽略 `.git/` 和 `.gitignore`**，这在项目里非常省事。

### 2.3 `grep`：在文件内容里查

```bash
grep -r "TODO" .                       # 递归搜 TODO
grep -rn "TODO" .                      # 显示行号
grep -ri "error" logs/                 # 忽略大小写
grep -v "DEBUG" app.log                # 反向匹配（排除）
grep -E "foo|bar" file                 # 扩展正则
grep -A 3 -B 2 "panic" log             # 匹配行+前 2 后 3 行
```

### 2.4 `rg` (ripgrep)：grep 的现代替代

```bash
rg "TODO"                              # 递归搜，默认忽略 .gitignore
rg -t py "def main"                    # 只在 .py 文件里搜
rg -g '*.rs' "unsafe"                  # glob 过滤
rg -C 3 "panic"                        # 上下文各 3 行
rg --files                             # 只列文件（常配合管道）
```

**经验**：`fd` 找文件，`rg` 搜内容，两者都比上一代快 5-10 倍。

---

## 3. 目录栈与跳转：z / zoxide

反复 `cd` 长路径是很烦的事。**zoxide** 学习你常去的目录，之后一两个字母就能跳过去。

```bash
# 安装（macOS）
brew install zoxide

# 在 ~/.zshrc 里追加（自动生成 z 命令）
eval "$(zoxide init zsh)"

# 用法
z infersim         # 模糊跳到你曾经待过的 InferSim 目录
z foo bar          # 匹配同时含 foo 和 bar 的路径
zi                 # 弹出 fzf 交互菜单选择
```

类似工具：`autojump`、`fasd`。

---

## 4. 版本管理：语言运行时怎么共存

### 4.1 Python

不推荐直接用系统 Python 装包，容易破坏系统。三种做法：

| 工具 | 定位 |
|------|------|
| `venv` | Python 自带，项目级虚拟环境 |
| `conda` / `miniconda` | 学术界常用，能管 Python 版本 + C 库 |
| `uv`（2024+ 新宠） | Rust 写的，10-100x 快于 pip |

**venv 最小流程**：
```bash
python3 -m venv .venv
source .venv/bin/activate      # 进入
pip install -r requirements.txt
deactivate                     # 退出
```

**uv 推荐流程**（更快）：
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 4.2 Node.js

用 **nvm** 或更快的 **fnm**、**volta**：
```bash
nvm install 20                  # 装 Node 20
nvm use 20
nvm ls                          # 列已装版本
```

### 4.3 跨语言版本管理

- **asdf** / **mise**：一个工具管所有语言（Python/Node/Ruby/Go/...）
- 项目根放一个 `.tool-versions` 文件，进入目录自动切版本

---

## 5. Dotfiles 与开发环境可复现

（L2 已讲基本做法）在"开发环境"维度补充：

- **Containerization**：把整个开发环境装进 Docker 镜像，任何新机器一 `docker run` 就能干活
- **Dev Containers**（VSCode）：项目里放 `.devcontainer/devcontainer.json`，VSCode 会自动在容器里打开项目，全团队环境一致
- **Nix / Nix Flake**：声明式、精确到字节的可复现环境（学习曲线陡，但终极方案）

---

## 6. 跨机器无缝工作：同步 + 远程开发

| 场景 | 推荐工具 |
|------|----------|
| 少量文件手动同步 | `scp` / `rsync` |
| 项目整体同步 | `git push/pull` 或 `rsync` |
| 双向实时同步 | `unison` / `mutagen` |
| 远程开发（本机 VSCode + 服务器 CPU/GPU） | **VSCode Remote-SSH** |
| 本机 Jupyter Notebook 跑远程 | `ssh -L 8888:localhost:8888` + 远程 `jupyter lab` |

---

## 7. 小结

- [x] Vim 四大模式及核心移动/编辑命令
- [x] 写一份 `.vimrc`，启用行号、相对行号、搜索高亮
- [x] `find`/`fd`、`grep`/`rg` 的分工与用法
- [x] `zoxide` 加速目录跳转
- [x] Python 用 `venv`/`uv` 做项目隔离，绝不污染系统
- [x] 跨语言版本管理（asdf/mise）
- [x] VSCode Remote-SSH 是 GPU 服务器开发的主力组合
