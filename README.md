# Work1

DSH 会话与 VS Code、Git 绑定的工作目录。

## 已完成

| 项 | 状态 |
| --- | --- |
| Git 仓库 | `git init`，分支 `main`，首次提交 `4c5d5e1` |
| Git 身份 | 沿用全局 `LinkCutY <linkcuty@qq.com>` |
| 换行规范 | `.gitattributes` 强制仓库内 LF，`*.ps1/*.cmd/*.bat` 保持 CRLF |
| VS Code 工作区 | `.vscode/` 四个配置文件，打开即生效 |
| 终端可运行性 | 任务统一用 `.cmd` 入口，规避本机 Restricted 策略 |

### 新增文件

| 文件 | 作用 |
| --- | --- |
| `.gitignore` | 忽略依赖、构建产物、密钥与本地 DSH 运行产物 |
| `.gitattributes` | 换行统一 |
| `.vscode/settings.json` | Git 行为、缩进与行尾、终端、文件排除 |
| `.vscode/extensions.json` | 推荐扩展（GitLens、Git Graph、Prettier 等） |
| `.vscode/tasks.json` | 一键跑 DSH 与常用 Git 操作 |
| `.vscode/launch.json` | PowerShell / Node / Python 当前文件调试 |
| `README.md` | 本文件 |

## 在 VS Code 里怎么用

1. 用 VS Code 打开本文件夹（工作区配置会自动加载）。
2. `Ctrl+Shift+B` → 运行生成任务：`DSH: 启动 Web GUI` / `Git: 查看当前状态` 等。
   `Ctrl+Shift+P` → `Tasks: Run Task` 可看全部。
3. `Ctrl+Shift+G` 源代码管理面板直接提交、同步。
4. `F5` 调试当前打开的脚本。

## 两个已知限制

### 1. DSH Web 的 Open In 认不出你的 VS Code

VS Code 装在 `D:\Microsoft VS Code`（1.139.0），而 DSH 的 Open In 只探测这两个路径：

```
%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe
%ProgramFiles%\Microsoft VS Code\Code.exe
```

所以 Web 界面顶部的 Open In 列表里不会出现 VS Code。
想让它出现，最省事的办法是在默认位置建一个目录联接（需要管理员权限）：

```powershell
New-Item -ItemType Junction `
  -Path "$env:LOCALAPPDATA\Programs\Microsoft VS Code" `
  -Target "D:\Microsoft VS Code"
```

### 2. PowerShell 执行策略为 Restricted

`*.ps1` 包装脚本无法运行，直接敲 `npm`、`dsh` 会报
`running scripts is disabled on this system`。
因此任务里统一使用 `.cmd` 入口（`npm.cmd`、`dsh.cmd`）。

想彻底修好，用管理员身份的 PowerShell 执行一次：

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

## 下一步：绑远程仓库

```powershell
git remote add origin <你的仓库地址>
git push -u origin main
```

本机没有安装 GitHub CLI（`gh`），所以远程仓库需要先手动建好。
