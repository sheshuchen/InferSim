# L5 · 版本控制与 Git —— 习题解答

---

## 习题 1：从零建一个仓库并推到 GitHub

**题目**：在本地创建 `hello-git` 项目，写一个 README，首次提交并推送到 GitHub。

**解答**：
```bash
mkdir hello-git && cd hello-git
git init
echo "# hello-git" > README.md
git add README.md
git commit -m "初始提交"

# GitHub 网页建同名空仓库（不要勾选 README），拿到 URL
git branch -M main
git remote add origin git@github.com:sheshuchen/hello-git.git
git push -u origin main
```

---

## 习题 2：分支的创建、合并、删除

**题目**：在 `hello-git` 里新开一个 `feat/hi` 分支，加一个 `hi.txt`，提交，合回 `main` 后删掉分支。

**解答**：
```bash
git switch -c feat/hi
echo "hi there" > hi.txt
git add hi.txt
git commit -m "feat: 添加 hi.txt"

git switch main
git merge feat/hi
git branch -d feat/hi

git log --oneline --graph --all
```

---

## 习题 3：四种撤销练习

**题目**：分别模拟以下情况并撤销：
1. 改了工作区的文件但还没 add
2. add 了但还没 commit
3. 已 commit 但还没 push
4. 已经 push 了

**解答**：
```bash
# 1) 工作区撤销
echo "oops" >> README.md
git restore README.md                    # 回到上次提交版本

# 2) 暂存区撤销
echo "oops" >> README.md
git add README.md
git restore --staged README.md           # 退出暂存
git restore README.md                    # 再丢弃工作区

# 3) 本地 commit 撤销
git commit --allow-empty -m "要撤销的 commit"
git reset --soft HEAD~1                  # 撤销 commit，改动仍在暂存
# 或 git reset --mixed HEAD~1            # 改动在工作区
# 谨慎：git reset --hard HEAD~1          # 连改动都丢

# 4) 已 push 的 commit
git revert HEAD                          # 生成一个"反做 commit"
git push
```

---

## 习题 4：解决合并冲突

**题目**：在两个分支里改同一行，制造冲突，手动解决后合并。

**解答**：
```bash
# main 分支
echo "line A" > conflict.txt
git add conflict.txt && git commit -m "main: line A"

# 建分支改
git switch -c feat/edit
echo "line B" > conflict.txt
git commit -am "feat: line B"

# 回主分支再改一遍
git switch main
echo "line C" > conflict.txt
git commit -am "main: line C"

# 合并 → 冲突
git merge feat/edit
# CONFLICT (content): Merge conflict in conflict.txt
cat conflict.txt
# <<<<<<< HEAD
# line C
# =======
# line B
# >>>>>>> feat/edit

# 编辑成想要的内容（比如保留 line B）
echo "line B" > conflict.txt
git add conflict.txt
git commit                  # 不加 -m，让 Git 自动写合并 message
```

---

## 习题 5：用 git stash 切换任务

**题目**：做到一半有紧急任务要切到 main 修 bug，回来后继续。

**解答**：
```bash
# 正在 feat 分支改代码，但还没做完
echo "wip" >> feature.py

git stash push -m "feature wip"
git switch main
# ...修 bug，commit, push...
git switch feat/xxx
git stash list                     # stash@{0}: On feat/xxx: feature wip
git stash pop                      # 恢复并从栈里移除
```

---

## 习题 6：用 git bisect 定位坏提交

**题目**：假设 `v1.0` 时测试通过，现在 `main` 测试失败，用 bisect 找到第一个坏提交。

**解答**：
```bash
git bisect start
git bisect bad                        # 当前坏
git bisect good v1.0                  # v1.0 好

# Git 自动 checkout 中间提交，你测试：
pytest                                # 假设这是你的测试
git bisect good                       # 或 bad

# 反复几次后：
# b3c9d20 is the first bad commit
git bisect reset                      # 恢复原位
git show b3c9d20                      # 查看这次提交改了什么
```

---

## 习题 7：git log 玩出花

**题目**：
1. 打印最近 10 次 commit 的紧凑图
2. 找出涉及 `attn.py` 文件的所有 commit
3. 找出谁某次引入了 `def forward` 这个字符串

**解答**：
```bash
git log --oneline --graph --decorate -10
git log --oneline -- layers/attn.py
git log -S "def forward" --oneline
git blame layers/attn.py              # 逐行看是谁改的
```

---

## 习题 8：rebase 清理 commit 历史

**题目**：在 feature 分支上有 5 个零碎 commit，合并前想合并成一个整洁 commit。

**解答**：
```bash
git switch feat/xxx
git rebase -i HEAD~5
# 编辑器里：
#   pick abc111 feat: add module
#   squash abc222 fix typo
#   squash abc333 fix typo again
#   squash abc444 forgot semicolon
#   squash abc555 rename variable
# 保存，再编辑最终 commit message

git log --oneline          # 只剩 1 个 commit
git push --force-with-lease   # 安全的强推（比 --force 更稳）
```

**重点**：`--force-with-lease` 会在远端有新改动时拒绝强推，防止覆盖同事工作。

---

## 习题 9：配置常用别名和 .gitignore

**题目**：给自己配置 5 条好用的 Git 别名，并给 Python 项目写一个完整 `.gitignore`。

**解答**：
```bash
git config --global alias.s  'status -sb'
git config --global alias.lg 'log --oneline --graph --decorate --all'
git config --global alias.co 'checkout'
git config --global alias.cm 'commit -m'
git config --global alias.unstage 'restore --staged'
```

`.gitignore` 模板：
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.venv/
venv/
.eggs/
*.egg-info/

# Jupyter
.ipynb_checkpoints/

# 系统
.DS_Store
Thumbs.db

# 编辑器
.vscode/
.idea/
*.swp
*.swo

# 构建/缓存
dist/
build/
.pytest_cache/
.mypy_cache/
.ruff_cache/
htmlcov/
.coverage

# 输出
*.log
logs/
```

---

## 习题 10：综合——完整的 PR 工作流模拟

**题目**：描述并演示一个从 fork 开源项目到合并 PR 的完整流程。

**解答**：
```bash
# 1. 在 GitHub 上 fork 目标仓库到自己账号
# 2. clone 自己 fork
git clone git@github.com:sheshuchen/infersim.git
cd infersim

# 3. 添加上游
git remote add upstream git@github.com:alibaba/InferSim.git
git remote -v

# 4. 从上游 main 开新分支
git fetch upstream
git switch -c feat/fix-throughput upstream/main

# 5. 改代码 + 写测试
# ...edit files...
git add .
git commit -m "fix: 修正 prefill 吞吐率计算"

# 6. 推送到自己 fork
git push -u origin feat/fix-throughput

# 7. 在 GitHub 网页发起 PR（从 sheshuchen:feat/fix-throughput -> alibaba:main）
# 8. 根据 review 继续 commit+push，PR 自动更新
# 9. 合入前同步上游
git fetch upstream
git rebase upstream/main
git push --force-with-lease

# 10. 合入后删分支
git switch main
git branch -D feat/fix-throughput
git push origin --delete feat/fix-throughput
```

---

## 本讲学习自检

- [ ] 能徒手画出工作区/暂存区/仓库的数据流
- [ ] 熟练在 `add / commit / push / pull / switch / merge / rebase` 间切换
- [ ] 四种撤销场景都能对应到正确命令
- [ ] 遇到冲突不紧张，会手动解决
- [ ] 用过 `git bisect` 或 `git blame` 查过历史
- [ ] 有自己的一套 alias 和 .gitignore 模板
- [ ] 跑通过一次完整的 PR 工作流
