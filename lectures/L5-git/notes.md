# L5 · 版本控制与 Git

> **原讲义链接**：https://missing.csail.mit.edu/2026/
> **本讲主题**：Version Control and Git
> **本文性质**：基于本讲主题编写的中文学习笔记（原创内容）

---

## 本讲要解决的问题

- 为什么不能用"文件名加日期"（`report_final_v3_真的最终.docx`）来管理代码？
- `git add`、`git commit`、`git push` 这些命令到底在做什么？
- 冲突了怎么办？回滚怎么做？
- 怎么和团队配合开发、提 Pull Request？

---

## 1. Git 的数据模型：先理解再用命令

很多人用了多年 Git 依然一知半解，就是因为**直接背命令**而没搞懂背后的数据结构。

### 1.1 三个核心概念

| 概念 | 英文 | 说白了是什么 |
|------|------|------------|
| **Blob** | blob | 一个文件的**内容**（只管内容，不管文件名） |
| **Tree** | tree | 一个**目录**：记录"哪些文件名对应哪些 blob/子 tree" |
| **Commit** | commit | 一次提交：指向一个 tree（当时项目的完整快照）+ 父提交 + 作者 + 信息 |

整个仓库的历史就是一棵**由 commit 组成的有向无环图（DAG）**，每个 commit 像一颗珠子，指向上一颗。

### 1.2 几个运行时概念

| 概念 | 说白了 |
|------|--------|
| **Working directory**（工作区） | 你当前看到的那堆文件 |
| **Staging area / Index**（暂存区） | 准备放入下一次 commit 的清单 |
| **Repository**（仓库） | `.git/` 目录，存所有历史 |

### 1.3 一次提交的数据流

```
[工作区]  --git add-->  [暂存区]  --git commit-->  [仓库]
```

**关键**：Git 的所有动作本质都是在这三者之间搬运数据。

---

## 2. 基础命令

```bash
git init                     # 新建仓库
git clone <url>              # 克隆远程仓库
git status                   # 查看当前状态（最常敲的一个命令）
git add <file>               # 把改动放进暂存区
git add -p                   # 逐块交互式 add（挑改动）
git commit                   # 提交（会弹编辑器写 message）
git commit -m "msg"          # 直接给 message
git log                      # 查看历史
git log --oneline --graph    # 紧凑图形化（强烈推荐别名）
git diff                     # 工作区 vs 暂存区
git diff --staged            # 暂存区 vs 上一次提交
git show <commit>            # 查看某次 commit 的改动
```

---

## 3. 分支（Branch）：Git 的灵魂

一个分支 = **一个指向某个 commit 的"标签"**。新建分支几乎免费，创建/切换都是 O(1)。

```bash
git branch                         # 列出本地分支
git branch -a                      # 包含远程
git branch feat/new-thing          # 创建（不切换）
git switch feat/new-thing          # 切换（新语法，推荐）
git switch -c feat/new-thing       # 创建并切换
# 老语法：
git checkout -b feat/new-thing

git merge feat/new-thing           # 把别的分支合并到当前
git branch -d feat/new-thing       # 删除已合并的分支
git branch -D feat/new-thing       # 强制删除
```

### merge vs rebase

| 方式 | 历史形状 | 适合 |
|------|----------|------|
| **merge** | 保留分叉，会产生一个 merge commit | 公共分支、保留真实历史 |
| **rebase** | 把你的 commits "挪到"目标分支顶部，线性历史 | 个人分支整理后再合入 |

**黄金法则**：**不要对已经推送给别人的公共分支做 rebase**（会改写历史，坑队友）。

---

## 4. 远程（Remote）

```bash
git remote -v                        # 查看远端
git remote add origin <url>          # 添加
git fetch origin                     # 拉取远端更新到本地（不合并）
git pull                             # = fetch + merge
git pull --rebase                    # = fetch + rebase（更干净）
git push                             # 推送当前分支
git push -u origin feat/xxx          # 首次推新分支 + 建立跟踪
git push origin --delete feat/xxx    # 删除远端分支
```

---

## 5. 撤销的几种方式（最容易搞混）

**原则**：先搞清你要撤销**哪一阶段**的改动。

### 5.1 改了工作区，还没 add

```bash
git restore <file>               # 丢弃工作区改动（新语法）
# 老语法：git checkout -- <file>
```

### 5.2 已经 add 了，还没 commit

```bash
git restore --staged <file>      # 只退出暂存区，保留工作区改动
# 老语法：git reset HEAD <file>
```

### 5.3 已经 commit 了（本地，还没 push）

```bash
git commit --amend               # 修改最近一次 commit（加文件、改 message）
git reset --soft HEAD~1          # 撤销最近一次 commit，保留工作区+暂存
git reset --mixed HEAD~1         # 撤销 commit + 清暂存，保留工作区（默认）
git reset --hard HEAD~1          # 🔥 彻底丢弃！连改动都没了，慎用
```

### 5.4 已经 push 的 commit

不要 `reset --hard` 强推（会改历史，坑队友）。用 `revert`：
```bash
git revert <commit>              # 生成一个"反做"commit，历史可追
```

---

## 6. 合并冲突的正确流程

```bash
git merge other-branch
# CONFLICT (content): Merge conflict in app.py

# 1. git status 看哪些文件冲突
# 2. 打开每个冲突文件，会看到：
#       <<<<<<< HEAD
#       你的版本
#       =======
#       对方版本
#       >>>>>>> other-branch
# 3. 人工编辑：保留想要的、删除所有 <<< === >>> 标记
# 4. git add <file>
# 5. git commit      # 不用 -m，会自动给出合并 commit 信息
```

工具推荐：VSCode 自带冲突 3 列对比界面，`git mergetool` 可以配 meld/kdiff3。

---

## 7. 查历史与找 bug 的利器

### 7.1 `git log` 进阶

```bash
git log --oneline --graph --all           # 全部分支的图
git log --author="sheshuchen"             # 只看某人
git log --since="2 weeks ago"             # 时间过滤
git log -S "function_name"                # 增/减过这个字符串的 commits
git log -p <file>                         # 某文件的逐次改动
```

### 7.2 `git blame`：这行代码是谁写的、哪个 commit 来的

```bash
git blame <file>
```

### 7.3 `git bisect`：二分定位引入 bug 的 commit

```bash
git bisect start
git bisect bad                    # 当前（已损坏）
git bisect good v1.0              # v1.0 时是好的
# Git 自动 checkout 中间的 commit，你测试
git bisect good   /   git bisect bad      # 根据测试结果
# 反复几次 Git 会告诉你首个坏掉的 commit
git bisect reset                  # 结束
```

**O(log n)** 定位 bug，非常强大。

---

## 8. 暂存零碎工作：stash

开发到一半突然要切分支修个紧急 bug：

```bash
git stash                        # 把当前未提交改动暂存
git switch main                  # 切过去做事
git stash pop                    # 回来后恢复
git stash list                   # 查看 stash 栈
git stash drop stash@{0}         # 删除
```

---

## 9. 协作工作流（PR / MR）

现代团队最常见的工作流（GitHub Flow）：

```
1. 从 main 拉一条 feature 分支
   git switch -c feat/add-login

2. 改、加测试、commit

3. 推到远端
   git push -u origin feat/add-login

4. 在 GitHub/GitLab 网页发起 Pull Request
   - 同事 review、评论、要求修改
   - 你继续 commit + push，PR 自动更新

5. CI 通过 + 同事 approve → Merge 到 main

6. 删掉 feature 分支
```

**经验法则**：
- 一个 PR 只做一件事，尽量**小**
- Commit message 写清楚"做了什么 + 为什么"
- 合入前跑 rebase/squash，保持历史整洁

---

## 10. 必备配置

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git config --global init.defaultBranch main
git config --global pull.rebase true         # 默认 rebase 拉取

# 实用别名
git config --global alias.s  'status -sb'
git config --global alias.lg 'log --oneline --graph --decorate --all'
git config --global alias.co 'checkout'
git config --global alias.cm 'commit -m'

# SSH 密钥（配合 L2 的 ssh-keygen）
# 把 ~/.ssh/id_ed25519.pub 内容贴到 GitHub Settings -> SSH Keys
```

`.gitignore` 必须配：
```
# Python
__pycache__/
*.pyc
.venv/

# 系统
.DS_Store

# 编辑器
.vscode/
.idea/

# 输出产物
*.log
dist/
build/
```

---

## 11. 小结

- [x] 理解 blob/tree/commit 三个核心对象
- [x] 区分工作区、暂存区、仓库
- [x] 熟练使用 add / commit / push / pull / switch / merge
- [x] 四种撤销方式对应哪一阶段改动分得清
- [x] 冲突能手动解决
- [x] 知道 `git bisect` 的价值
- [x] 熟悉 PR 工作流，能提交一个像样的 PR
