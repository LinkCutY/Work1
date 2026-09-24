# bmmms-kernel-gm —— 让 AIV/AIC 之间少穿 GM、且每次搬运都"满"

**交付物：`kernel.asc`**（本目录）。基线是**队友 push 的最新 `kernel.asc`**，本文件在
**不改数学语义、不改任务划分、不改输出写回方式**的前提下做两件事：

1. 把 AIV 与 AIC 之间**穿过 GM 的次数**压到平台 220x 允许的下限；
2. 让**缓冲尽量满存**：分片占满 L0C，AIV 每次 GM→UB 搬运尽量铺满 UB。

> "满存"在本版落在两个可量化的点上：**L0C 分片 = 128×256×4 = 131072 B = L0C 的 100%**；
> **AIV 每次归并缓冲读 = 128KB = UB(192KB) 的 68%**（余下的容量留给归约用的 4 个小缓冲，
> 它们各 512B）。若你指的"US/满存"是别的缓冲（例如 L1 里 B 的重载），见第七节。

| 项 | 值 |
| --- | --- |
| 基线 | `origin/main` @ `f5b3995`，`kernel.asc` blob `3e96fd39`（真机 **15/15** 的那版） |
| 基线副本 | `teammate-latest.asc`（同目录，逐字节 = `git show origin/main:.../kernel.asc`） |
| 本版 | `kernel.asc`（改动集中在 `MatmulMaxKernel` 与 host 的 tiling/workspace） |
| 提交方式 | 覆盖 `npu-v1/project/kernel.asc`（判题侧唯一 `editable: true` 的文件，其余只读文件未动） |

## 零、上一版为什么被拒（本轮修复）

真机回报：`Profiling rule violated: each iteration must launch exactly 1 kernel.
Expected 75 launches, got 110.`

**判题硬规则**：一次迭代只能启动 1 个 kernel。而且 —— 这一条是本轮才吃透的 ——
**任何 abort（host `Fail()` → `status 134`）都会被平台 `<ReplayOnce>` 重放，每次重放多算一次
launch**。`110 = 75 + 35 = 75 + 7×5`，即约 **7 个 case 每轮都 abort**。根因全在 host 侧两条
"我方主动 Fail"，与 kernel 本身、与分片策略无关：

| # | 缺陷 | 触发条件 | 修法 |
| --- | --- | --- | --- |
| 1 | 分片请求阶梯下限写死 `64`：`for (request = splitN; request >= 64; request -= 64)` | **`n < 64` 的 case 一档都试不成** → `chosenN == 0` → `Fail()` | 阶梯改两档，baseN 一路退到 **8**；`baseM` 档加了**基线(16, splitN) 兜底档** |
| 2 | 回读校验要求 `baseM >= min(splitM, m)` | tiler 自己把 baseM 收敛到更小值（完全合法）→ `Fail()` | 删掉这一条；只留"UB/GM 缓冲真正的上界 + baseN 8 对齐" |
| 3 | `aclrtSynchronizeStreamWithTimeout(stream, 3000)` | 任一 case 超过 3s → `CheckAcl` → `Fail()` | 提到 **60s**（这个超时只是本地安全网，判题有自己的 TLE） |

修复后的 abort 面：**基线原有的 7 处 + 2 处构造上不可达的兜底**（`chosenN == 0`、
回读越界）—— 因为兜底档就是真机 15/15 那版用的 `(16, splitN)`，一定拿得到 tiling。
本地新增门禁 `tools/launch_check.py` 守"启动点互斥、无同块并联启动"（本版 6 处启动，
与基线同为互斥的 2 小 + 4 转置分支）。

> 为什么"分片请求阶梯"不会再把合法情况变成 abort：kernel 侧的行块步长就是 `baseM`，
> `validM = min(baseM, m - rowStart)` 恒 ≤ `baseM` ⇒ `mBlocks` 恒为 1，所以
> **任何** `baseM ≤ 128 / baseN ≤ 256` 的回读值都是安全的几何 —— 退让只是少赚。

## 一、为什么"穿 GM"绕不开

`910D_knowledge_extra/220x到351x架构变更.md:120-165` 把「**新增 L0C Buffer 到 UB 的
单向数据通路**」明确列为 **351x 的新增能力**（"无需先从 L0C 搬运到 GM，再从 GM 搬运到 UB"）。
本题实测架构是 **220x（DAV_2201）**，没有这条通路 —— 所以每块 C 分片只能：

```
Fixpipe: L0C -> GM      （每片必写一次，次数下限 = 分片数）
MTE2:    GM  -> UB      （AIV 每读一次都要付搬运代价）
```

## 二、两条杠杆

**(1) 分片顶到 L0C 满存 → 分片数最少 → 搬运次数最少、单次搬运最大。**
L0C = 128KB、C 为 float ⇒ `baseM*baseN*4 ≤ 131072` ⇒ `baseM*baseN ≤ 32768`。
本版取 **`128×256 = 32768`**（正好 128KB，`dbL0C = 1`）。
基线是 `16×256 = 4096`（L0C 只用 **16KB/128KB**）⇒ **分片数与同步次数都差 8 倍**。
`SetFixSplit`/`GetTiling` 若拒绝 256 列（L0A/L0B/L1 放不下对应 `baseK`），
host 会**按 64 列一档退让**（256→192→128→64），全部失败才 `Fail()`；
最终几何一律以 `GetTiling` 回读值为准，所以"退让"只是少赚，不改语义。

**(2) 同一行块的 N 个分片在 GM 上就地归并 → AIV 每行块只读 1 次，且这次读就是 128KB。**
`mm.GetTensorC(cAcc, enAtomic, true)`，`enAtomic = 2` 即 **AtomicMax**
（`高阶API/矩阵计算/Matmul_Kernel侧接口/GetTensorC.md:100`：0=无、1=AtomicAdd、2=AtomicMax、3=AtomicMin）。
A2/A3 的原子通路包含 **L0C→GM** 且支持 **float**（`基础API/原子操作/SetAtomicAdd.md:31`、
`SetAtomicMax（ISASI）.md:9-27`）。
每行块**第 0 片用普通写**（`enAtomic=0`）—— 它同时充当累加缓冲的初值：原子操作**不会**清零
（`SetAtomicAdd.md:47`），所以不能用 `-inf` 预置 GM；第 1 片起全部 AtomicMax 到同一地址，
一个行块的 N 个分片最后只剩一个 `[baseM, baseN]` 缓冲 = **128KB，正好是 UB 的 68%**。

## 三、GM 次数与"满存"账（单 batch，`B=1`）

计数口径：一次"写"= Fixpipe 把一块分片落到 GM；一次"读"= AIV 从 GM 搬回一块。
N 尾块（`n % baseN != 0`）在两侧都是额外 1 写 + 1 读。

| shape (M×N) | 基线 写+读 | 本版 写+读 | 倍数 |
| --- | --- | --- | --- |
| 512×512 | 64 + 64 = 128 | 8 + 4 = 12 | **10.7×** |
| 1000×1000（尾块 232） | 252 + 252 = 504 | 32+8 + 16+8 = 64 | **7.9×** |
| 4096×4096 | 4096 + 4096 = 8192 | 512 + 32 = 544 | **15.1×** |
| 512×8192 | 1024 + 1024 = 2048 | 128 + 4 = 132 | **15.5×** |
| 8192×4096 | 8192 + 8192 = 16384 | 1024 + 64 = 1088 | **15.1×** |

（基线按 `baseM=16`、`baseN=min(256, align16(n))` 计；本版按 `128×256` 计。
**真机 baseN 的实际取值日志里没有**（`bmmms-kernel-fix/DIAGNOSIS.md` 未验证项 V1），
若真机 baseN 比 256 小，基线的分片数会更多，倍数只会更大。）

| 缓冲 | 基线 | 本版 |
| --- | --- | --- |
| L0C 分片占用 | 16×256×4 = 16KB / 128KB（12.5%） | 128×256×4 = 131072 = **128KB / 128KB（100%）** |
| AIV 每次 GM→UB 搬运 | 16×256×4 = 16KB / 192KB | 128×256×4 = **128KB / 192KB（68%）** |
| UB 总占用 | ≈17KB | ≈130KB（cUb 128KB + 4×512B） |
| 每次搬运的数据量 | 16KB 级 | **128KB 级（8×）** |

附带效应（分片放大的必要条件）：向量侧归约调用数从 `m × (n/baseN)` 次 `ReduceMax`
降到 `(m/baseM) × (4 次 WholeReduceMax + 4 次 Max)`，例如 4096×4096：**65536 次 → 256 次**。

## 四、为什么正确性不受影响

1. **max 可交换可结合** ⇒ AtomicMax 的合并顺序无关，结果**逐位确定**。
   比 AtomicAdd 强：后者会因调度顺序漂末位，而题面要求"多次执行结果一致"。
2. **N 尾块不参与归并**。连续写（`enSequentialWrite=true`）在尾块按 `validN` 紧凑打包，
   平坦下标与整块**不同构**（`123b2b8` 的实测结论），混进归并缓冲会跨行串列。
   尾块落**独立槽位**，沿用基线已实测的标量逐行读法，再 `Max` 回行最大值。
3. **行归约语义与基线一致**：先跨 N 取 max（尾块后补），再按行升序求和；
   行和只在 `row < validM` 上做，`validM < baseM` 时不会把上一行块的残留算进来。
4. **不信任 tiler**：host 侧回读校验 `baseM ≥ min(splitM, m)`、`baseM ≤ 128`、
   `baseN ≤ 256`、`baseN % 8 == 0`，不满足直接 `Fail()`。
   `baseM ≥ min(splitM,m)` 保证 `mBlocks == 1`，这是"一个行块只用一个归并缓冲"的前提。

## 五、本地做了哪些验证

| 门禁 | 结果 |
| --- | --- |
| **`tools/launch_check.py`（判题硬规则：iter 内单 kernel）** | **通过**：6 处启动点全在互斥分支、无同块并联；并列出全部 `Fail()` 调用点供复核 |
| `tools/syntax_check.py`（g++ `-fsyntax-only` + 手抄 8.5 文档签名的 stubs） | 基线 ok / 本版 **ok**，本版比基线**少**一个 warning |
| `tools/reduce_oracle_gm.py`（搬运/归约顺序的离线逐位复算） | **20 组 shape × 两档配置**（激进 `128×256` 与**兜底 `16×…`**）× `np.array_equal` **全部逐位一致**（含 `baseN=256`、`tailN` 0→255、`n<baseN`、`m%baseM!=0`、`n=8192`） |
| 同上，**反向验证（4 个变体，确定性构造）** | 尾块混进归并 10/10、行和多算残留 8/8、关掉 AtomicMax 9/9、漏掉尾块 10/10 —— **全部被检出** |
| `tools/api_delta.py`（相对 15/15 基线多用了哪些 API） | 只多 3 个：`WholeReduceMax`、`SetAtomicNone`、`HardEvent::V_MTE2`（外加 `GetTensorC` 的 `enAtomic=2` 取值）；少了 `ReduceMax` |
| 容量算术 | 见第三节表格（L0C 100%、UB 68%、GM 槽位 = 3×blocks×256KB，blocks=24 时 19MB） |

**没有做（本机做不到）**：真机编译、上板精度、真机用时、AtomicMax 的实际硬件行为、
tiler 到底会不会接受 256 列的分片。

## 六、上板必查（按优先级，详细版在 `kernel.asc` 末尾 VERIFY，共 10 条）

0. **先看日志里还有没有 `<ReplayOnce>` / `status 134` / launch 数超标** —— 本轮修的就是
   这条：任何 abort 都会被重放并多算一个 launch。若再次出现，把新的差值（`got - expected`）
   告我，`差值 / 迭代次数 = 出问题的 case 数`，能直接锁定是哪些形状。
1. **Fixpipe 的 AtomicMax 是否生效**（本版唯一没有真机先例的组合：`enAtomic=2` +
   `enSequentialWrite=true`）。失败特征：**所有 case 结果偏小**（只剩最后一个 N 分片）。
2. **打印回读的 `baseM/baseN/dbL0C`**：
   `baseN==256 且 dbL0C==1` ⇒ L0C 满存、目标达成；若 `baseN` 退到 128 ⇒ tiler 拒绝了 256 列，
   下一步该调 `baseK`/`SetBufferSpace`，而不是继续放大 `baseN`。
3. **`dbL0C == 1` 意味着 Mmad 与 Fixpipe 不能重叠**：如果"分片数变少了但总时间没变"，
   下一档是把 `MAX_TILE_N` 降到 128（`dbL0C=2`，两种流水重叠）再比一次。
4. **UB 预算**：本版 UB = 133120 B / 192KB。若上板 `InitBuffer` 报 UB 不足，
   把 `MAX_TILE_N` 降到 192 或 128（kernel 用回读值工作，改常量即可）。
5. `WholeReduceMax` 的掩码/顺序参数、原子模式是否残留到 y、两次执行是否逐位一致 —— 同 VERIFY 7~9。

## 七、如果"满存"指的不是 L0C/UB

本版能直接量化的只有上面两处。另外两处**没有**动，因为它们的收益不如杠杆 (1)(2) 大、
而改动面更大（需要改循环嵌套或迭代顺序）：

- **L1 里 B 的重载**：当前每个行块都会把 B 的 `[baseK, baseN]` 分片重新从 GM 取一遍
  （`n/baseN × m/baseM` 次）。要把 B 常驻 L1，得把迭代顺序换成 N 外层 / M 内层
  （`TCubeTiling.iterateOrder`），代价是行最大值要跨行块累积、AIV 侧状态变复杂。
- **把归并缓冲加宽到 UB 上限**（128 行 × 384 列 = 192KB）：那会让 AIV 把**未压缩**的
  N 数据读回（读量 ×1.5），与"减少读写"相反 —— 现在的做法是先在 GM 上把 N 压成
  `[baseM, baseN]` 再读，读量最小。

## 八、文件

| 文件 | 说明 |
| --- | --- |
| **`kernel.asc`** | ★ 交付物：覆盖 `npu-v1/project/kernel.asc` 提交 |
| `teammate-latest.asc` | 基线原文（真机 15/15）—— 退路 |
| `tools/launch_check.py` | **判题硬规则门禁**：启动点互斥、无同块并联、列出全部 `Fail()` 复核点 |
| `tools/syntax_check.py` + `tools/stubs/` | 本地语法门禁（g++ `-fsyntax-only`） |
| `tools/reduce_oracle_gm.py` | 离线逐位复算 + 反向验证（含兜底档配置） |
| `tools/api_delta.py` | 相对基线的 API 差集（列出无真机证据的新用法） |

```bash
python tools/launch_check.py kernel.asc      # 判题硬规则（必须 0 退出）
python tools/syntax_check.py kernel.asc      # 语法门禁
python tools/reduce_oracle_gm.py             # 归约逻辑逐位复算（退出码 0 = 全过）
python tools/api_delta.py                    # 新 API 差集
```
