# Git 协作入门：从零上手我们的分支流水线

> 面向：从未用过 Git 的队友。读完本文你不需要懂 Git 内部原理，只需要照着敲命令。
> 我们的方案：**problem 分支（写代码、出图）→ main 分支（汇合点）→ paper 分支（写论文、取图）**，数据单向流动。

---

## 一、Git 是什么（3 分钟版）

把 Git 想象成一个**带历史记录的共享网盘**：

| 概念 | 网盘类比 | 我们项目里的样子 |
|---|---|---|
| 仓库（repository） | 一个共享文件夹 | GitHub 上的 `wuwwc/2025B` |
| 提交（commit） | 一次"保存存档"，记录了改了啥 | "问题1：修复EESM单位bug" |
| 分支（branch) | 一条平行的时间线/副本 | `problem1`、`problem2`、`main`、`paper` |
| 工作区 | 你电脑上正在编辑的文件 | `C:\Users\你\Desktop\code\2025B` |
| 推送 / 拉取（push / pull） | 上传存档 / 下载别人的存档 | 把你本地提交同步到 GitHub / 反过来 |

三个关键认知：

1. **提交是存档，不是覆盖整个文件夹。** 每次 commit 只记录"改了哪些行"，所以历史可以任意回退、对比。
2. **分支之间互不打扰。** 你在 `problem1` 上怎么改，都不会影响别人看到的 `main`，直到你主动合并。
3. **push 之前的一切都可以反悔；push 之后就要跟队友协调了。** 这是新手的黄金守则。

---

## 二、一次性准备（每人只做一次）

### 1. 安装 Git

到 <https://git-scm.com/download/win> 下载安装，一路默认即可。装完后**重开** PowerShell 窗口。

验证：

```powershell
git --version
```

### 2. 配置姓名和邮箱（会显示在提交记录里）

```powershell
git config --global user.name "你的名字"
git config --global user.email "你的邮箱@example.com"
```

### 3. 配置 SSH 密钥（我们连 GitHub 必须走 SSH，HTTPS 会超时）

```powershell
# 生成密钥（一路回车）
ssh-keygen -t ed25519 -C "你的邮箱@example.com"

# 打印公钥，复制整行输出
cat ~/.ssh/id_ed25519.pub
```

把公钥粘贴到 GitHub → 头像 → Settings → SSH and GPG keys → New SSH key。

然后**必须告诉 Git 用 SSH 地址**克隆：

```powershell
git clone git@github.com:wuwwc/2025B.git
```

### 4. 防止"自动合并"捣乱（重要，只做一次）

```powershell
git config --global pull.ff only
```

不设置的话，Git 会在你拉取时悄悄制造一堆看不懂的合并提交（我们仓库历史上的混乱就是这么来的）。设置后遇到需要抉择的情况，Git 会停下来报错，你再按本文第五节的场景处理。

### 5. 克隆后自检

```powershell
cd 2025B
git status          # 应显示 On branch main ... nothing to commit
git branch -a       # 应能看到 main / problem1 / problem2 / paper
```

---

## 三、我们的流水线规则（先背熟这 5 条）

```
 problem1 ──┐
            ├──→ main ──→ paper
 problem2 ──┘
（单向流动，箭头绝不反向）
```

1. **写代码、跑实验、出图 → 只在自己的 problem 分支上。**
2. **图是"产物"：图只在 problem 分支生成并 commit，paper 分支只使用、绝不修改图文件。**
3. **problem → main 通过 Pull Request（PR）合并**，在 GitHub 网页上点按钮完成，不本地直接改 main。
4. **main → paper 通过同步命令完成**（第五节场景 C），每天下班前同步一次。
5. **一切"拉取/更新"前先 `git fetch origin`**，保证看到的是远端最新状态。

每条分支谁负责：

| 分支 | 内容 | 谁能推提交 |
|---|---|---|
| `problem1` / `problem2` | 各自题目代码、结果图（`output/figs/`） | 题目负责人 |
| `main` | 所有人的成品汇合点 | 只通过 PR 进入，谁都不直接 push |
| `paper` | LaTeX 论文、文档 | 论文负责人（建议 1~2 人，人多了冲突多） |

---

## 四、日常循环（每天就这三步）

以下命令都在仓库目录里执行（先 `cd 项目路径\2025B`）。

### 第 1 步：看看现在在哪个分支、改了什么

```powershell
git status
```

输出会告诉你两件事：**当前分支**（`On branch xxx`）和**改动的文件**。任何时候不确定状态，就敲这个命令，它是最安全的"体检仪"。

### 第 2 步：保存进度（提交）

```powershell
git add 文件名        # 把指定文件放入"待提交清单"
# 或者
git add -A            # 把本次所有改动都放入清单（新手推荐先用这个）

git commit -m "问题1：修正速率图坐标轴单位"
```

提交信息写法：**"模块：做了什么"**，一句话让人看懂。禁止 `111`、`update` 这类信息。

> 小步提交：每完成一个小事项就 commit 一次，比憋一天提交一个巨型 commit 好回退、好查问题。

### 第 3 步：上传到 GitHub（推送）

```powershell
git push
```

第一次推某个新分支时会提示设置上游，按它提示的敲 `git push -u origin 分支名` 即可，以后直接 `git push`。

### 每天开工前：先更新本地

```powershell
git fetch origin      # 下载远端最新状态（只读，绝对安全）
git pull              # 把我所在分支的远端更新取下来
```

`git fetch` 只下载不改动你的任何文件，任何时候敲都不会出错，分不清状况就先 fetch。

---

## 五、按角色 / 场景的操作手册

直接对号入座，照抄命令。

### 场景 A：我是 problem1 负责人，今天写了代码出了图

```powershell
cd 2025B
git checkout problem1        # 切换到 problem1 分支（相当于"进入那条时间线"）
git pull                     # 确保拿到 problem1 上最新进度

# ……写代码、跑 make_figures.py 出图……

git add -A
git commit -m "问题1：新增MCS阶跃速率对比图"
git push
```

图放在固定位置：`problem1/output/figs/图名_v1.png`。**改版时换新文件名**（`_v2`、`_final`），不要覆盖旧图，方便论文回溯引用了哪一版。

### 场景 B：我的 problem 分支成果要进 main（发 PR）

代码已经 push 到 `problem1` 之后：

1. 打开 <https://github.com/wuwwc/2025B>；
2. 页面顶部会出现黄色横幅 **"Compare & pull request"**，点击它；
   （没出现就手动点 Compare → 源分支选 `problem1`、目标选 `main` → Create pull request）
3. 标题写清楚内容，比如"问题1：结果图与最终代码合入"，点 **Create pull request**；
4. 让一位队友打开这个 PR 页面，确认改动列表没毛病，点 **Merge pull request** → **Confirm merge**。

完成后，你的图就"上架"到 main 了，paper 分支的人就能取到。

> PR 是什么：一次"申请入库"的单据，把合并前需要看的 diff、讨论、点合并全部集中在网页上，避免在命令行里操作错分支。我们要求 main 只能从 PR 进，就是为了这一步永远有人把关。

### 场景 C：我是 paper 负责人，要把 main 上最新的图取进论文

在 paper 分支上执行（**这就是 main → paper 的全部动作**）：

```powershell
git checkout paper
git fetch origin
git rebase origin/main      # 把论文提交垫到最新 main 之上，同时拿到新图
git push --force-with-lease # rebase 改写了历史，推送时需要这个参数
```

解释一下最后两行：`rebase` 让 paper 的历史保持一条直线（第四节提过它比 merge 干净）；改写历史后普通 push 会被拒绝，`--force-with-lease` 是"礼貌的强推"：只在没有别人比你更新的提交时才覆盖，安全。

> 前提：paper 分支只有你（或你们两人约定好轮流）在推。如果 rebase 后 push 被 `--force-with-lease` 拒绝，说明队友刚推了新提交——先 `git pull --rebase origin paper` 再继续，**绝不用 `--force` 硬推**。

在 LaTeX 里引用图（路径从项目根算起，固定写法）：

```latex
\includegraphics{problem1/output/figs/rate_compare_v2.png}
```

### 场景 D：我改了文件但改坏了，想丢弃本地改动

```powershell
git status                          # 先看改坏了哪些文件
git checkout -- 文件路径            # 丢弃单个文件的本地改动（回到上次提交的状态）
```

危险等级高一点：被丢弃的本地改动**找不回来**，敲之前确认该文件里没有你想留的东西。

### 场景 E：提交错了信息 / 漏了文件，想修正"最后一次"提交

```powershell
git add 漏掉的文件
git commit --amend -m "问题1：新增MCS阶跃速率对比图（补上说明）"
git push --force-with-lease
```

只能修正**最近一次**且**还没被队友拉走**的提交。已经 push 给别人用过的提交别 amend。

### 场景 F：完全乱了，不敢动

在命令行敲任何 Git 命令前，乱不了的是**已 push 到 GitHub 的历史**。自救三步：

```powershell
git status          # 把输出完整读一遍，Git 会告诉你下一步能干嘛
git fetch origin    # 同步远端认知
```

然后打开 GitHub 网页看目标分支的最新提交，确认远端没被弄坏，再向队友描述 `git status` 的输出。绝大多数"乱了"只是本地分叉，按报错提示处理即可。

---

## 六、冲突：怎么避免 >> 怎么处理

### 为什么会冲突

同一个文件的**同一片行**，两个人改成了不同的内容，Git 不知道该听谁的，就会报冲突。我们的流水线已经把冲突面压到最小：

- problem1 / problem2 各自目录，天然不重叠；
- 图只在 problem 生成，paper 只读；
- **最大风险点是论文源文件**：两个人同时编辑 `MathModel.tex` 最容易撞车。解决办法：
  1. 论文按章节拆文件（`sec1.tex`、`sec2.tex`……，`\input` 进主文件），一人一次只改一章；
  2. 或者同一时段只允许一个人在 paper 分支 push。

### 真冲突了怎么解

执行 pull / rebase 后看到 `CONFLICT` 提示时：

1. `git status` 查看哪些文件冲突（状态是 `both modified`）；
2. 打开冲突文件，找到标记块：

   ```
   <<<<<<< HEAD
   你这边写的版本
   =======
   对方（或 main）的版本
   >>>>>>> origin/main
   ```

3. 手动编辑：删掉三行标记符号，留下正确内容（可以只留 A、只留 B，或融合两者）；
4. 标记已解决并继续：

   ```powershell
   git add 解决好的文件
   git rebase --continue     # 若当时是 rebase 流程
   # 或 git commit           # 若当时是 merge 流程
   ```

5. 中途想放弃、回到冲突前：`git rebase --abort`。

> 图片二进制文件冲突不用打开编辑，直接二选一：`git checkout --theirs 图路径`（取 main 的图，图以 problem 产出为准时就这么选）或 `--ours`，然后 `git add` 继续。

---

## 七、常用命令速查表

| 想干嘛 | 命令 |
|---|---|
| 我在哪、改了啥 | `git status` |
| 看所有分支 | `git branch -a` |
| 切换分支 | `git checkout 分支名` |
| 下载远端最新状态（安全，随便敲） | `git fetch origin` |
| 更新我当前分支 | `git pull` |
| 保存进度 | `git add -A` → `git commit -m "说明"` |
| 上传到 GitHub | `git push` |
| 看最近提交历史 | `git log --oneline -10` |
| 看某个文件改了什么 | `git diff 文件名` |
| 丢弃某文件本地改动 | `git checkout -- 文件名` |
| 同步 main 的图到 paper | `git fetch origin` → `git rebase origin/main` → `git push --force-with-lease` |
| 中止搞了一半的 rebase | `git rebase --abort` |

---

## 八、新手红线（违反任何一条先停下来问队友）

1. 🚫 不直接 `git push` 到 `main`（都走 PR）；
2. 🚫 不敲 `git push --force`（要重写历史只用 `--force-with-lease`）；
3. 🚫 不 amend / 不改写**已经 push 且别人用过**的提交；
4. 🚫 不在 paper 分支修改 `output/figs/` 里的任何图；
5. 🚫 看不懂 Git 报错时**不要连敲命令试运气**，先 `git status` 看清状态、读报错提示，或直接截图发到群里问；
6. 🚫 大数据文件（数据集、缓存 `.npz`、zip）不 `git add`——它们在 `.gitignore` 里，如果你发现某个该忽略的文件被提交了，马上告诉队友。

---

## 附：第一天上手练习（不影响任何真实分支）

1. 克隆仓库（第二节）；
2. `git checkout problem1`，在 `problem1/` 下新建一个 `notes_你的名字.md` 随便写点东西；
3. `git add -A` → `git commit -m "练习：我的第一次提交"` → `git push`；
4. 打开 GitHub 网页，确认能看到你的提交；
5. 玩明白后可以删掉这个文件再提交一次，或让队友在 PR 时帮你看看。

完成这五步，你就已经掌握了我们流水线 90% 的日常操作。
