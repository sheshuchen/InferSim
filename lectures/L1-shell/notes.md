# L1 · Shell 入门

> **原讲义链接**：https://missing.csail.mit.edu/2026/course-shell/ 或 https://missing-semester-cn.github.io/
> **许可证**：CC BY-NC-SA 4.0
> **本文性质**：中文学习笔记（知识点提炼，非逐字翻译）

---

## 1. 为什么要学 Shell

图形界面（GUI）、语音、甚至 LLM 聊天框都能操控电脑，但它们本质上**只能做程序员预先做好的按钮**。当你要把多个小工具自由组合、批量处理文件、或者配置远程服务器时，图形界面往往无能为力。

**Shell 的核心价值**是：
- 把电脑里成百上千个小程序当作"积木"，用文本命令把它们**拼接、串联**
- 一旦写成脚本，可重复、可记录、可版本化
- 是理解 Linux/macOS 系统、操作远程机器、玩转开源项目的**必备基础**

---

## 2. 基本概念辨析

| 术语 | 含义 |
|------|------|
| **Terminal（终端）** | 显示文字并接收键盘输入的**窗口程序**（如 macOS 的 Terminal.app、iTerm2） |
| **Shell（外壳）** | 真正解释命令的**程序**（如 bash、zsh、fish）。在终端里跑 |
| **Prompt（提示符）** | Shell 等待你输入命令时显示的一行提示，比如 `user@host:~$` |
| **Command（命令）** | 你敲的一整行指令，第一个词通常是程序名，后面是参数 |

**经典提示符解读**：`missing:~$`
- `missing`：主机名
- `~`：当前工作目录（`~` 是家目录的缩写）
- `$`：普通用户（`#` 代表 root 用户）

---

## 3. 命令是怎么执行的

当你输入 `echo hello world`：

1. Shell 按**空白**把这行拆成若干单词：`["echo", "hello", "world"]`
2. 第一个单词 `echo` 被当作"要运行的程序"
3. 剩下的 `hello`、`world` 作为参数传递给这个程序
4. Shell 在若干目录里按顺序查找 `echo` 这个可执行文件（由 `$PATH` 环境变量决定）

**带空格/特殊字符的参数**，必须保护起来：
```bash
cd "My Photos"      # 双引号（允许变量展开）
cd 'My Photos'      # 单引号（完全原样）
cd My\ Photos       # 反斜杠转义
```

---

## 4. 最重要的几个命令

### `man` —— 使用手册
```bash
man ls              # 查看 ls 的完整说明
man man             # man 自己也有手册
```
翻页：`空格`（下一页）、`b`（上一页）、`q`（退出）、`/关键词`（搜索）。

**补充工具（推荐安装）**：
- `tldr ls`：只看最常用的几个例子，比 `man` 更贴近实战
- `ls --help`：大多数程序都带简短帮助

### `cd` —— 切换目录
```bash
cd /bin         # 绝对路径
cd bin          # 相对路径
cd ..           # 上一级
cd ~            # 家目录
cd -            # 返回上一次所在目录（常用）
cd              # 不带参数=回家目录
```

⚠️ `cd` 不是独立程序，而是 shell 内置命令（`which cd` 会失败）。

### `pwd` —— 打印当前路径
```bash
pwd             # 输出当前工作目录
echo $PWD       # 等价写法
```

### `ls` —— 列目录
```bash
ls              # 普通列表
ls -l           # 详细信息（权限、大小、时间）
ls -a           # 显示隐藏文件（以 . 开头的）
ls -lh          # 大小用 KB/MB 等人类可读单位
ls -la /tmp     # 参数可以组合，路径可以指定
```

---

## 5. 路径（Path）

| 类型 | 形式 | 举例 |
|------|------|------|
| **绝对路径** | 以 `/` 开头 | `/usr/local/bin/python3` |
| **相对路径** | 其他 | `./script.sh`、`../docs`、`src/main.py` |

两个特殊目录名（每个目录都有）：
- `.` 当前目录
- `..` 上一级目录

所以 `cd bin/../bin/./` 绕一圈还是 `cd bin`。

**注意**：很多命令接受绝对路径和相对路径都可以，效果一样。

---

## 6. 权限（ls -l 输出的第一列）

示例：`drwxr-xr-x`

| 位置 | 含义 |
|------|------|
| 第 1 位 | 类型：`d` 目录、`-` 文件、`l` 软链接 |
| 第 2-4 位 | 所有者的 rwx 权限 |
| 第 5-7 位 | 所属组的 rwx 权限 |
| 第 8-10 位 | 其他人的 rwx 权限 |

- `r` = 读（对目录：能不能 `ls`）
- `w` = 写（对目录：能不能增删文件）
- `x` = 执行（对目录：能不能 `cd` 进去/穿越）

**重点**：想进入 `/home/alice/` 查看其下某文件，需要对 `/`、`/home`、`/home/alice` 这一路的**每一层都有 `x` 权限**。

---

## 7. 文件操作常用命令

```bash
mv src.txt dst.txt            # 改名
mv file.txt ~/Documents/      # 移动
cp src.txt dst.txt            # 复制
cp -r dirA dirB               # 递归复制目录
mkdir new_dir                 # 新建目录
mkdir -p a/b/c                # 多层一次创建
rm file.txt                   # 删除文件（不可恢复！）
rm -r dir                     # 递归删目录
rmdir empty_dir               # 只删空目录（更安全）
touch newfile                 # 新建空文件/更新时间戳
```

---

## 8. 程序输出与流（Streams）

每个程序运行时都有三条**默认流**：

| 编号 | 名字 | 作用 |
|------|------|------|
| 0 | **stdin**（标准输入） | 读取输入，默认连到键盘 |
| 1 | **stdout**（标准输出） | 正常打印结果，默认连到屏幕 |
| 2 | **stderr**（标准错误） | 错误信息，默认也到屏幕 |

**重定向**（把流改接到文件或其他程序）：

```bash
echo hello > out.txt       # 把 stdout 覆盖写入 out.txt
echo more >> out.txt       # 追加
cat < out.txt              # 把 out.txt 作为 stdin 喂给 cat
ls notexist 2> err.txt     # 把 stderr 写入 err.txt
ls / > all.txt 2>&1        # stderr 合并进 stdout，一起写入 all.txt
ls / &> all.txt            # 上一条的简写（bash）
```

### 管道（Pipe）`|`：把一个程序的 stdout 接到下一个的 stdin

```bash
ls -l /usr/bin | wc -l     # 统计 /usr/bin 下有多少项
history | grep ssh         # 从历史记录里搜 ssh 相关命令
cat log.txt | sort | uniq  # 排序后去重
```

管道是 Unix 哲学的精髓：**每个工具做一件小事，靠管道组合出强大能力**。

---

## 9. 根用户与 sudo

- 普通用户提示符是 `$`，root 是 `#`
- root 可以无视权限为所欲为（危险！）
- 临时借用 root 权限执行**单条**命令：`sudo 命令`

```bash
sudo apt install htop          # Debian/Ubuntu
sudo systemctl restart nginx
```

**坑点**：重定向是 **shell 做的**，不是 sudo 做的。
```bash
sudo echo 1 > /proc/sys/...    # ❌ 依然失败：shell 以你自己的身份开文件
echo 1 | sudo tee /proc/sys/...  # ✅ 正确用法
```

---

## 10. 常用小技巧

| 技巧 | 效果 |
|------|------|
| `Tab` | 自动补全命令/路径（关键省时技巧） |
| `Ctrl+C` | 终止当前前台程序 |
| `Ctrl+D` | 发送 EOF（关闭 stdin / 退出 shell） |
| `Ctrl+L` 或 `clear` | 清屏 |
| `Ctrl+R` | 反向搜索历史命令 |
| `history` | 查看历史命令 |
| `!!` | 上一条命令 |
| `!ssh` | 最近一条以 ssh 开头的命令 |
| `Ctrl+A` / `Ctrl+E` | 光标跳到行首/行尾 |
| `Ctrl+U` / `Ctrl+K` | 删除光标前/后整行 |

---

## 11. 小结：今天应该掌握的

- [x] Terminal、Shell、Prompt 的区别
- [x] 命令是"程序 + 参数"的拆分规则，以及空格/特殊字符如何转义
- [x] `man`、`cd`、`pwd`、`ls`、基本文件操作命令
- [x] 绝对路径 vs 相对路径，`.` 和 `..` 的含义
- [x] `ls -l` 的权限位如何解读
- [x] stdin/stdout/stderr，以及 `>`、`>>`、`<`、`|`、`2>&1` 等重定向/管道
- [x] `sudo` 的正确用法与常见陷阱

> **下一讲预告（L2）**：命令行环境——Shell 脚本、任务控制（前台/后台/jobs）、终端多路复用（tmux）、SSH 与 dotfiles。
