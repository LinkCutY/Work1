# bmmms-kernel-gm —— 让 AIV/AIC 之间少穿 GM

**交付物：`kernel.asc`**（本目录）。它是对**队友 push 的最新 `kernel.asc`** 的定向优化：
在**不改数学语义、不改任务划分、不改输出写回方式**的前提下，把「AIV 与 AIC 之间
穿过 GM 的次数」压到平台 220x 允许的下限。

| 项 | 值 |
| --- | --- |
| 基线 | `origin/main` @ `f5b3995`，`kernel.asc` blob `3e96fd39`（真机 **15/15** 的那版） |
| 基线副本 | `teammate-latest.asc`（同目录，逐字节 = `git show origin/main:.../kernel.asc`） |
| 本版 | `kernel.asc`（1893 行；改动集中在 `MatmulMaxKernel` 与 host 的 tiling/workspace） |
| 提交方式 | 覆盖 `npu-v1/project/kernel.asc`（判题侧唯一 `editable: true` 的文件，其余只读文件未动） |

## 一、为什么"穿 GM"绕不开

`910D_knowledge_extra/220x到351x架构变更.md:120-165` 把「**新增 L0C Buffer 到 UB 的
单向数据通路**」明确列为 **351x 的新增能力**（"无需先从 L0C 搬运到 GM，再从 GM 搬运到 UB"）。
本题实测架构是 **220x（DAV_2201）**，没有这条通路 —— 所以每块 C 分片只能：

```
Fixpipe: L0C -> GM      （每片必写一次，次数下限 = 分片数）
MTE2:    GM  -> UB      （AIV 每读一次都要付搬运代价）
```

在"必须穿"这个前提下只剩两个杠杆。

## 二、两条杠杆

**(1) 分片面积最大化 → 分片数（= 写次数）最少。**
L0C = 128KB、C 为 float、`dbL0C = 2` ⇒ `baseM*baseN*4*2 ≤ 131072` ⇒ `baseM*baseN ≤ 16384`。
基线是 `16×256 = 4096`（L0C 只用 16KB/128KB）；本版取 **`128×128 = 16384`**（64KB/片，
`dbL0C=2` 仍合法）⇒ **分片数降 4 倍**。

**(2) 同一行块的 N 个分片在 GM 上就地归并 → AIV 每行块只读 1 次。**
`mm.GetTensorC(cAcc, enAtomic, true)`，`enAtomic = 2` 即 **AtomicMax**
（`高阶API/矩阵计算/Matmul_Kernel侧接口/GetTensorC.md:100`：0=无、1=AtomicAdd、2=AtomicMax、3=AtomicMin）。
A2/A3 的原子通路包含 **L0C→GM** 且支持 **float**（`基础API/原子操作/SetAtomicAdd.md:31`、
`SetAtomicMax（ISASI）.md:9-27`）。
每行块**第 0 片用普通写**（`enAtomic=0`）—— 它同时充当累加缓冲的初值：原子操作**不会**清零
（`SetAtomicAdd.md:47`），所以不能用 `-inf` 预置 GM；第 1 片起全部 AtomicMax 到同一地址，
一个行块的 N 个分片最后只剩一个 `[baseM, baseN]` 缓冲。

## 三、GM 次数账（单 batch，`B=1`）

计数口径：一次"写"= Fixpipe 把一块 `[baseM,baseN]` 分片落到 GM；一次"读"= AIV 从 GM 搬回一块。
N 尾块（`n % baseN != 0`）在两侧都是额外 1 写 + 1 读。

| shape (M×N) | 基线 写+读 | 本版 写+读 | 倍数 |
| --- | --- | --- | --- |
| 512×512 | 64 + 64 = 128 | 16 + 4 = 20 | **6.4×** |
| 1000×1000（尾块 104） | 252 + 252 = 504 | 64+8 + 8+8 = 88 | **5.7×** |
| 4096×4096 | 4096 + 4096 = 8192 | 1024 + 32 = 1056 | **7.8×** |
| 512×8192 | 1024 + 1024 = 2048 | 256 + 4 = 260 | **7.9×** |
| 8192×4096 | 8192 + 8192 = 16384 | 2048 + 64 = 2112 | **7.8×** |

（基线按 `baseM=16`、`baseN=min(256, align16(n))` 计；本版按 `128×128` 计。
注意**真机 baseN 的实际取值日志里没有**（`bmmms-kernel-fix/DIAGNOSIS.md` 未验证项 V1），
若真机 baseN 比 256 小，基线的分片数会更多，倍数只会更大。
`1000×1000` 一行的本版读数是 `8 行块 × (1 归并 + 1 尾块)`。）

附带效应（不是本版目标，但是分片放大的必要条件）：向量侧归约调用数从
`m × (n/baseN)` 次 `ReduceMax` 降到 `(m/baseM) × (≤2 次 WholeReduceMax + 1 次 Max)`。
例如 4096×4096：**65536 次 → 96 次**。

## 四、为什么正确性不受影响

1. **max 可交换可结合** ⇒ AtomicMax 的合并顺序无关，结果**逐位确定**。
   比 AtomicAdd 强：后者会因调度顺序漂末位，而题面要求"多次执行结果一致"。
2. **N 尾块不参与归并**。连续写（`enSequentialWrite=true`）在尾块按 `validN` 紧凑打包，
   平坦下标与整块**不同构**（`123b2b8` 的实测结论），混进归并缓冲会跨行串列。
   尾块落**独立槽位**，沿用基线已实测的标量逐行读法，再 `Max` 回行最大值。
3. **行归约的语义与基线一致**：先跨 N 取 max（尾块后补），再按行升序求和；
   行和只在 `row < validM` 上做，`validM < baseM` 时不会把上一行块的残留算进来。
4. **不信任 tiler**：host 侧回读校验 `baseM ≥ min(splitM, m)`、`baseM ≤ 128`、
   `baseN ≤ 128`、`baseN % 8 == 0`，不满足直接 `Fail()`（宁可本地不跑，不带错假设上板）。
   `baseM ≥ min(splitM,m)` 保证 `mBlocks == 1`，这是"一个行块只用一个归并缓冲"的前提。

## 五、本地做了哪些验证

| 门禁 | 结果 |
| --- | --- |
| `tools/syntax_check.py`（g++ `-fsyntax-only` + 手抄 8.5 文档签名的 stubs） | 基线 ok / 本版 **ok**，且本版比基线**少**一个 warning（删掉了基线里未使用的 `totalRowTiles`） |
| `tools/reduce_oracle_gm.py`（搬运/归约顺序的离线逐位复算） | 12 组 shape（含 `n%baseN!=0`、`m%baseM!=0`、`n<baseN`、`n=8192`）× `np.array_equal` **全部逐位一致** |
| 同上，**反向验证** | 4 个故意做错的变体（尾块混进归并 / 行和多算块尾残留 / 关掉 AtomicMax / 漏掉尾块）**全部被检出（7/7、5/5、7/7、7/7）**，证明 Oracle 不是空转 |
| `tools/api_delta.py`（相对 15/15 基线多用了哪些 API） | 只多 3 个：`WholeReduceMax`、`SetAtomicNone`、`HardEvent::V_MTE2`（外加 `GetTensorC` 的 `enAtomic=2` 取值）；少了 `ReduceMax` |
| L0C 容量算术 | `128*128*4*2 = 131072 = L0C` 恰好不超（`dbL0C=2`）；UB 占用 `128*128*4 + 4×128*4 ≈ 66KB` / 192KB |

**没有做（本机做不到）**：真机编译、上板精度、真机用时、AtomicMax 的实际硬件行为。

## 六、上板必查（按优先级，详细版在 `kernel.asc` 末尾 VERIFY）

1. **Fixpipe 的 AtomicMax 是否生效** —— 这是本版**唯一**没有真机先例的组合
   （`enAtomic=2` + `enSequentialWrite=true`）。失败特征：**所有 case 结果偏小**
   （只剩最后一个 N 分片的贡献）。退路：换回 `teammate-latest.asc`（15/15 基线）。
2. `WholeReduceMax` 的 `mask=64 / repeat=validM / dstRepStride=1 / srcBlkStride=1 / srcRepStride=baseN/8 / ORDER_ONLY_VALUE`。
   失败特征：行最大值取到别的行的数据（系统性偏大/错位）。
3. 原子模式是否残留到 y 的写出（本版在尾片前显式 `SetAtomicNone()`）。
4. host 侧回读校验是否触发 `Fail()`；实际 `baseM/baseN` 是否为 128/128
   （若 tiler 下调，kernel 会自动按更小分片工作：正确性不变，GM 次数变多）。
5. 两次执行同输入是否逐位一致（题面要求；AtomicMax 理论上保证）。

## 七、文件

| 文件 | 说明 |
| --- | --- |
| **`kernel.asc`** | ★ 交付物：优化后的内核（覆盖 `npu-v1/project/kernel.asc` 提交） |
| `teammate-latest.asc` | 基线原文（`origin/main` 的 `kernel.asc`，真机 15/15）—— 退路 |
| `tools/syntax_check.py` + `tools/stubs/` | 本地语法门禁（g++ `-fsyntax-only`） |
| `tools/reduce_oracle_gm.py` | 离线逐位复算 + 反向验证 |
| `tools/api_delta.py` | 相对基线的 API 差集（列出无真机证据的新用法） |

```bash
python tools/syntax_check.py kernel.asc      # 语法门禁
python tools/reduce_oracle_gm.py             # 归约逻辑逐位复算（退出码 0 = 全过）
python tools/api_delta.py                    # 新 API 差集
```
