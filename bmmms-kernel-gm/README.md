# bmmms-kernel-gm —— 少穿 GM + 把 512KiB L1 用起来

**交付物：`kernel.asc`**。基线是队友 push 的最新 `kernel.asc`（真机 15/15）。本版在
**不改数学语义、不改任务划分、不改输出写回方式**的前提下，继续压 AIV/AIC 之间的 GM 往返，
并按「L1 = 512KiB」这条已知事实把 AIC 侧的操作数复用打开。

| 项 | 值 |
| --- | --- |
| 基线 | `origin/main` @ `f5b3995`，`kernel.asc` blob `3e96fd39`（真机 **15/15**） |
| 基线副本 | `teammate-latest.asc`（同目录，逐字节 = `git show origin/main:.../kernel.asc`） |
| 本版 | `kernel.asc`：分片 128×128（方阵、dbL0C=2）+ AtomicMax 就地归并 + **M 窗口** |
| 提交方式 | 覆盖 `npu-v1/project/kernel.asc`（判题侧唯一 `editable: true` 的文件） |

## 一、这一版相对上一版的三处变化

1. **分片回到方阵 `128×128`（16384 元素、64KB、`dbL0C = 2`）**，丢弃"L0C 满存"的
   `128×256`（`dbL0C = 1`）。理由：AIC 从 GM 取操作数的量是
   `(n/baseN)·m·k·2 + (m/baseM)·n·k·2`，在 `baseM·baseN ≤ L0C/4` 的约束下，
   这个和由 **AM-GM 在 `baseM ≈ baseN` 处最小**；而 `dbL0C = 1` 会让 Mmad 与 Fixpipe
   无法重叠。方阵同时把 L0C 留出双缓冲。
2. **新增 M 窗口**：一次 `SetSingleShape + SetTensorA/B` 覆盖**最多 4 个行块（512 行）**，
   分片按 `mBlk = tileIdx % windowBlocks` 路由到各行块自己的归并区，AIV 在**窗口末尾**
   逐行块读回一次（每个行块仍是 1 次读，只是不再与 AIC 逐片握手）。目的见第三节。
3. **保留上一版的 abort 修复**（这不是可选项）：分片请求阶梯的最后一档 = 基线配置、
   回读校验只留必要上界、内部同步超时 3s→60s。理由见第六节（真机上 abort 会被
   `ReplayOnce` 重放，直接违反"一次迭代一个 kernel"）。

## 二、为什么"穿 GM"绕不开

`910D_knowledge_extra/220x到351x架构变更.md:120-165` 把「**新增 L0C Buffer 到 UB 的
单向数据通路**」列为 **351x 的新增能力**。本题实测架构是 **220x（DAV_2201）**，没有它 ——
每块 C 分片只能 `Fixpipe: L0C→GM` 再 `MTE2: GM→UB`。

## 三、L1 = 512KiB 到底能用在哪里（本版核心）

L1 是 **AIC 的操作数缓冲**（A1/B1）。文档把它的用法写得很具体
（`Matmul_Tiling侧接口/Matmul_Tiling类/TCubeTiling结构体.md`）：

- `:18-19`　`depthA1` = A1 里全载 `baseM×baseK` 的份数；`stepM` = A1 缓存的 **M 方向
  baseM 的倍数**（`stepN` 同理是 B 方向的）；
- `:48`　`AL1Size + BL1Size ≤ L1_size`，其中 `AL1Size = baseM·baseK·depthA1·2`、
  `BL1Size = baseN·baseK·depthB1·2`，且 `depthA1 = stepM·stepKa·db`、`depthB1 = stepN·stepKb·db`
  ⇒ **512KiB 这个数字直接决定 `stepM/stepN/stepKa/stepKb` 能开多大**；
- `:55`　**"Ka 不全载时，即 Ka/baseK > stepKa，stepM = 1"** ⇒ 想跨 M 行块复用 B 面板，
  前提是整个 K 的 A 面板能装进 L1。

推论（本版据此做的三件事）：

| 手段 | 依据 | 效果 |
| --- | --- | --- |
| 给足**单核 M**（窗口 512 行） | `:55` + `stepM` 的定义 | 框架才有机会让 `stepM > 1`（K 装得下时 B 面板复用 `stepM` 倍），同时 `depthA1/stepKa` 能把 K 方向多级缓冲铺到 512KiB，减少停等 |
| 分片取**方阵** | `AL1Size/BL1Size` 与操作数流量公式 | 操作数流量在 `baseM≈baseN` 最小；`dbL0C=2` 让 Mmad/Fixpipe 重叠 |
| `SetBufferSpace(-1,-1,-1)` 保持不变 | `SetBufferSpace.md`：`-1` = 用处理器实际 L1/L0C/UB 大小 | 框架本来就知道 512KiB，不需要（也不该）手工塞常数 |

⚠ **诚实边界**：`stepM > 1` 要求整个 K 的 A 面板进 L1（`:55`）。K 很大时（例如 k=4096、
`baseM=128`：A 面板 1MB ≫ 512KB）这条**不成立**，框架只能 `stepM = 1`。此时窗口带来的
收益就只剩"K 方向多级缓冲铺得更满 → 少停等"，**B 面板仍会按 `m/baseM` 次重读**。
要判断到底吃到了哪一档，上板打印 `stepM/stepN/depthA1/depthB1`（见 `kernel.asc` 末尾 VERIFY 4）。

## 四、GM 账（单 batch）

**AIV↔AIC 之间（C 分片）**：一次"写" = Fixpipe 落一块 `[baseM,baseN]`；一次"读" = AIV 搬回一块。

| shape (M×N) | 基线 写+读 | 本版 写+读 | 倍数 |
| --- | --- | --- | --- |
| 512×512 | 64 + 64 = 128 | 16 + 4 = 20 | **6.4×** |
| 1000×1000（尾块 104） | 252 + 252 = 504 | 64 + 16 = 80 | **6.3×** |
| 4096×4096 | 4096 + 4096 = 8192 | 1024 + 32 = 1056 | **7.8×** |
| 512×8192 | 1024 + 1024 = 2048 | 256 + 4 = 260 | **7.9×** |

（基线按 `baseM=16`、`baseN=min(256, align16(n))` 计；本版按 `128×128` 计。
基线 `baseN` 的真机实际值日志里没有 —— `bmmms-kernel-fix/DIAGNOSIS.md` 未验证项 V1 ——
若它更小，基线分片数更多、倍数只会更大。）

**AIC 操作数（GM→L1，这才是 GM 上的大头）**：
`A: (n/baseN)·m·k·2` + `B: (m/baseM)·n·k·2`。以 4096³ 为例，`128×128` 分片、
`stepM=stepN=1` 时是 `1.07GB + 1.07GB ≈ 2.1GB`；**每次 C 分片的 GM 写只有 64KB**，
所以操作数流量比 C 侧大 30 倍 —— 这也是本版把力气花在窗口/L1 上的原因。

**GM 槽位**：每 worker `2 半 × 4 行块 × 128×128 × 4B = 512KiB`，总 `3·blocks·512KiB`
（blocks=24 时 36MB）。

## 五、为什么正确性不受影响

1. **max 可交换可结合** ⇒ AtomicMax 的合并顺序无关，结果**逐位确定**（题面要求多次执行一致）。
2. **路由解码有文档依据**：`Iterate.md:21` 明确"默认以先 **M 轴**再 N 轴的迭代顺序"，
   `TCubeTiling` 里 `iterateOrder=0` 也是"先往 M 轴方向偏移再往 N 轴" ⇒
   `mBlk = tileIdx % mBlocks`、`nBlk = tileIdx / mBlocks`。**并且做了双保险**：
   host 侧只有 `tiling.iterateOrder == 0`（或窗口退化成 1 个行块）时才启用多行块窗口，
   否则自动退成单行块（解码无歧义）。本地 Oracle 有专门的 `order_n` 反向验证。
3. **归并区初值**：每个行块的第 0 个 N 分片（`nBlk == 0`）用普通写；原子操作不清零
   （`SetAtomicAdd.md:47`），而 M 优先顺序保证它一定先到。
4. **N 尾块不参与归并**：连续写在尾块按 `validN` 紧凑打包，平坦下标与整块不同构
   （`123b2b8` 的实测结论），混进去会跨行串列 —— 尾块落独立区，走基线已实测的标量逐行读法。
5. **不信任 tiler**：回读校验 `baseM ≤ 128`、`baseN ≤ 128`、`baseN % 8 == 0` 三条必要条件；
   请求阶梯全部由 `SetFixSplit/GetTiling` 的返回值驱动，最后一档是基线配置 ⇒ 不会因为
   分片/窗口请求而 abort。

## 六、上一版被拒的根因（本版保留的修复）

真机回报 `Expected 75 launches, got 110` ⇒ **`110 = 75 + 35 = 75 + 7×5`：约 7 个 case
每轮都 abort，每次 abort 被 `<ReplayOnce>` 重放、多算一个 launch**。三条自伤逻辑已修：

| # | 缺陷 | 修法 |
| --- | --- | --- |
| 1 | 分片阶梯下限写死 64 ⇒ `n < 64` 的 case 一档都试不成 → `Fail()` | 阶梯退到 **8**，并加**基线(16, splitN) 兜底档** |
| 2 | 回读校验要求 `baseM ≥ min(splitM, m)` ⇒ tiler 合法收敛也被判死 | 只留必要上界 |
| 3 | `aclrtSynchronizeStreamWithTimeout(stream, 3000)` ⇒ 任一大 case 超 3s 即 abort | 提到 **60s**（判题有自己的 TLE） |

修复后 abort 面 = **基线原有的 7 处 + 2 处构造上不可达的兜底**；本地 `launch_check.py` 守
"启动点互斥、无同块并联"（本版 6 处启动点，与基线同为互斥的 2 小 + 4 转置分支）。

## 七、本地验证（都重跑过）

| 门禁 | 结果 |
| --- | --- |
| `tools/launch_check.py` | **通过**：6 处启动点全在互斥分支、无同块并联；列出全部 `Fail()` 复核点 |
| `tools/reduce_oracle_gm.py` | **20 组 shape × 5 档配置**（窗口 1/2/4 × 方阵分片，以及兜底 `16×N` × 窗口 1/4）与 golden **逐位一致**；反向 5 个变体（尾块混入归并、行和多算残留、关掉 AtomicMax、漏尾块、**迭代顺序误判**）**全部被检出** |
| `tools/syntax_check.py` | 基线 ok / 本版 **ok**（本版比基线少一个 warning） |
| `tools/api_delta.py` | 相对基线只多 3 个 API：`WholeReduceMax`、`SetAtomicNone`、`HardEvent::V_MTE2`；少了 `ReduceMax` |

**没有做（本机做不到）**：真机编译、上板精度、真机用时、AtomicMax 的硬件行为、
tiler 是否真的把 512KiB L1 用起来（`stepM/stepN/depth*` 的实际值）。

## 八、上板必查（详细版在 `kernel.asc` 末尾 VERIFY，共 10 条）

0. 日志里**是否还有 `<ReplayOnce>` / `status 134` / launch 数超标**；若再次超标，
   把新的差值给我（`差值 ÷ 迭代次数 = 出问题的 case 数`）。
1. **AtomicMax 是否生效**（`enAtomic=2` + `enSequentialWrite=true`，无真机先例）。
   失败特征：**所有 case 结果偏小**（只剩最后一个 N 分片）。退路：`teammate-latest.asc`。
2. **打印 `baseM/baseN/iterateOrder/stepM/stepN/depthA1/depthB1`** —— 这五个数决定
   L1 到底吃到了多少（第 3 节）。
3. `WholeReduceMax` 的行距/顺序参数（mask=64、repeat=validM、srcRepStride=baseN/8、
   ORDER_ONLY_VALUE）；失败特征：行最大值取到别的行。
4. UB 67.6KB / 192KB、GM 槽位 36MB 是否都分配成功。
5. 同输入两次执行是否逐位一致（题面要求；AtomicMax 理论上保证）。

## 九、文件

| 文件 | 说明 |
| --- | --- |
| **`kernel.asc`** | ★ 交付物 |
| `teammate-latest.asc` | 基线原文（真机 15/15）—— 退路 |
| `tools/launch_check.py` | 判题硬规则门禁（单 kernel / 无同块并联 / 列出 Fail 点） |
| `tools/syntax_check.py` + `tools/stubs/` | 语法门禁 |
| `tools/reduce_oracle_gm.py` | 搬运/路由/归约的离线逐位复算 + 反向验证 |
| `tools/api_delta.py` | 相对基线的 API 差集 |

```bash
python tools/launch_check.py kernel.asc      # 判题硬规则（必须 0 退出）
python tools/syntax_check.py kernel.asc      # 语法门禁
python tools/reduce_oracle_gm.py             # 逐位复算（退出码 0 = 全过）
python tools/api_delta.py                    # 新 API 差集
```
