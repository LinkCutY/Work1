# Work1

DSH 会话与 VS Code、Git 绑定的工作目录。

## 绑定结果

| 项 | 状态 |
| --- | --- |
| Git 仓库 | 分支 `main`，工作区干净 |
| Git 身份 | 沿用全局 `LinkCutY <linkcuty@qq.com>` |
| 换行规范 | 仓库内强制 LF，`*.ps1/*.cmd/*.bat` 保持 CRLF |
| VS Code 工作区 | `.vscode/` 四个配置文件，打开即生效 |
| DSH Open In | 已可识别 VS Code（经 `%LOCALAPPDATA%\Programs\Microsoft VS Code` 联接） |
| PowerShell 策略 | `CurrentUser = RemoteSigned`，`npm` / `dsh` 可直接调用 |

### 文件

| 文件 | 作用 |
| --- | --- |
| `.gitignore` | 忽略依赖、构建产物、密钥与本地 DSH 运行产物 |
| `.gitattributes` | 换行统一 |
| `.vscode/settings.json` | Git 行为、缩进与行尾、终端、文件排除 |
| `.vscode/extensions.json` | 推荐扩展（GitLens、Git Graph、Prettier 等） |
| `.vscode/tasks.json` | 一键跑 DSH 与常用 Git 操作 |
| `.vscode/launch.json` | PowerShell / Node / Python 当前文件调试 |

## 环境说明

### VS Code 的实际位置与联接

VS Code 1.139.0 实体安装在 `D:\Microsoft VS Code`。
DSH 的 Open In 只探测两个固定路径：

```
%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe
%ProgramFiles%\Microsoft VS Code\Code.exe
```

因此 `%LOCALAPPDATA%\Programs\Microsoft VS Code` 被建成指向 `D:\Microsoft VS Code` 的目录联接
（Junction）。这是让 Open In 认出来的关键，**不要删除该联接**。
验证：

```powershell
Get-Item "$env:LOCALAPPDATA\Programs\Microsoft VS Code" | Select-Object LinkType, Target
```

补充：注册表卸载记录里的 `InstallLocation` 仍写的是 `D:\Microsoft VS Code\`，
`Code.exe` 只在 D 盘一份，联接是唯一让默认探测路径命中的方式。

### PowerShell 执行策略

已设为 `CurrentUser = RemoteSigned`（注册表已确认），`npm`、`dsh` 现在可以直接敲，
不再需要 `.cmd` 后缀绕过。任务里仍保留 `.cmd` 入口，好处是与终端种类无关、
且以后策略被重置也不会失效。

## 在 VS Code 里怎么用

1. 用 VS Code 打开本文件夹（工作区配置自动加载）。
2. `Ctrl+Shift+B` → 运行生成任务：`DSH: 启动 Web GUI` / `Git: 查看当前状态` 等。
   `Ctrl+Shift+P` → `Tasks: Run Task` 可看全部。
3. `Ctrl+Shift+G` 源代码管理面板直接提交、同步。
4. `F5` 调试当前打开的脚本。

## 下一步：绑远程仓库

本机没有安装 GitHub CLI（`gh`），远程仓库需要先手动建好，然后：

```powershell
git remote add origin <你的仓库地址>
git push -u origin main
```
