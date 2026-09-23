# Work1

DSH 会话与 VS Code、Git 绑定的工作目录。

## 已配置内容

| 文件 | 作用 |
| --- | --- |
| `.gitignore` | 忽略依赖、构建产物、密钥与本地 DSH 运行产物 |
| `.gitattributes` | 仓库内统一 LF；`*.ps1/*.cmd/*.bat` 保持 CRLF |
| `.vscode/settings.json` | Git 行为、缩进与行尾规范、终端与文件排除 |
| `.vscode/extensions.json` | 推荐扩展（GitLens、Git Graph、Prettier 等） |
| `.vscode/tasks.json` | 一键跑 DSH 与常用 Git 操作 |
| `.vscode/launch.json` | PowerShell / Node / Python 当前文件调试 |

## 在 VS Code 里使用

打开本文件夹，然后：

- `Ctrl+Shift+B` → 运行生成任务，可选「DSH: 启动 Web GUI」或 Git 相关任务。
- `Ctrl+Shift+P` → `Tasks: Run Task` 查看全部任务。
- 源代码管理面板（`Ctrl+Shift+G`）直接提交 / 同步。

## 终端注意事项

本机 PowerShell 执行策略为 `Restricted`，`*.ps1` 包装脚本无法运行（`npm.ps1`、`dsh.ps1` 都会报
`running scripts is disabled`）。因此：

- 任务里统一使用 `.cmd` 入口。
- 若要在 PowerShell 里直接用 `npm`，请先以管理员身份执行一次：

  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
  ```

## 待办

- [ ] 绑定远程仓库：`git remote add origin <URL>` 后 `git push -u origin main`
