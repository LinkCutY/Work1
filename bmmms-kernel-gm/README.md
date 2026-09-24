# bmmms-kernel-gm —— 把 AIV↔AIC 的**握手次数**打到下限

**交付物：`kernel.asc`**。基线是队友 push 的最新 `kernel.asc`（真机 15/15）。本版不改数学
语义、不改任务划分、不改输出写回方式，只改**搬运与调用的组织方式**。

| 项 | 值 |
| --- | --- |
| 基线 | `origin/main` @ `f5b3995`，`kernel.asc` blob `3e96fd39`（真机 **15/15**） |
| 基线副本 | `teammate-latest.asc`（同目录，逐字节 = `git show origin/main:.../kernel.asc`） |
| 本版 | `kernel.asc`：**`IterateAll` 一次算完一个行块** + 大块读回 + 两级归约 |
| 提交方式 | 覆盖 `npu-v1/project/kernel.asc`（判题侧唯一 `editable: true` 的文件） |

## 一、为什么"没有变快"——上一版的诊断

上一版（128×128 分片 + AtomicMax 就地归并 + M 窗口）确实把**分片数**降下来了
（4096² 从 4096 片降到 1024 片），但**每片仍然要一次 `while (mm.Iterate()) { GetTensorC }`**：
即每片一次 AIV↔AIC 握手 + 一次同步等待。分片大了 4 倍，握手次数只降 4 倍，
而握手本身的**延迟**基本不变 —— 所以看起来"只快了一点"。

**真正的杠杆是把"每片一次握手"换成"每行块一次"**：

```cpp
mm.SetSingleShape(validM, shape.n, shape.k);
mm.SetTensorA(...); mm.SetTensorB(...);
mm.IterateAll(cStage, /*enAtomic=*/0, /*enSequentialWrite=*/true);   // ← 一个行块只有这一次
```

文档依据（`IterateAll.md:20-70`）："调用一次IterateAll，会计算出
singleCoreM * singleCoreN大小的C矩阵"、**默认同步**（"需要同步等待IterateAll执行结束"）、
A2/A3 支持，示例就是 `REGIST_MATMUL_OBJ + SetTensorA/B + IterateAll(gm_c)` —— 与本文件同构。
约束只有一条：C 的地址空间 ≥ `singleCoreM * singleCoreN`（我们用 staging 满足）。

## 二、三次改版的握手/搬运账（单 batch，4096×4096，baseM=baseN=128）

| 版本 | AIV↔AIC 同步次数 | Fixpipe 写片数 | AIV 读回次数 |
| --- | --- | --- | --- |
| 队友基线（16×256，每片握手+每片读） | 4096 | 4096（16KB/片） | 4096（16KB/次） |
| 上一版（128×128 + AtomicMax 归并 + M 窗口） | **1024** | 1024（64KB/片） | 32（64KB/次） |
| **本版（IterateAll + 大块读）** | **32** | 1024（64KB/片） | **512**（128KB/次，2 片一读） |

握手次数 4096 → 32（**128×**）。既然原来的瓶颈是"每片一次握手 + 同步等待"，
这一项就是本版要吃的收益；单次握手按基线结构估算在 µs 量级，乘以 4096 就是原来那几毫秒。
**这是估算，不是实测** —— 本机没有 CANN/NPU，无法给真机数字。

顺带的取舍（都写进 VERIFY）：

- 读回从"归并缓冲 1 次 64KB"变成"每行块 ceil(n/baseN/2) 次 128KB"：**读次数变多、每次更大**
  （UB 一次铺满），总读字节数从 2MB/batch 升到 64MB/batch —— 相比 AIC 侧操作数流量
  （同 shape 约 2GB）可以忽略，而它换来的是握手次数两个数量级的下降。
- **不再用 AtomicMax**（`enAtomic=2`）：上一版唯一的"无真机先例"组合，本版换成
  "跨分片只做 Max"，语义更弱、更容易验证。
- 每个 worker 需要一块 staging：`baseM × n × 4B`（n=8192 时 4MB/worker），
  总 `min(b, 2*blocks)` 块。

## 三、本版的搬运/归约组织（"合理分配存储空间和调用顺序"）

```
每个 batch（属主 AIV 独占）：
  for 每个行块 (baseM 行)：
      IterateAll(cStage, 0, true)        ← 1 次同步；连续写把 ceil(n/baseN) 个分片
                                           按 [validM, baseN] 依次铺在 staging 里
      while (还有整片)：
          DataCopyPad 读 READ_TILES=2 片   ← rows=2*validM 行、每行 baseN 个 float，
                                            128KB，一次把 UB 铺满
          第一级：逐"列组"WholeReduceMax    ← 每 64 列一组（余列单独一组），
                                            行距 = baseN/8 个 datablock，结果连续落 groupMax
          第二级：元素级 Max 折各列组        ← 组值之间隔 8 个 float，只能靠 Max 折，
                                            用归约的连续 mask 会读到别的行（本地复算抓到过）
          按分片把行最大值 Max 进行最大值向量 ← 跨 N 取 max，与分片顺序无关
      N 尾块：紧凑布局（行距 = validN），逐行标量读（沿用 15/15 已验证的读法）
      行和 → batchSum（只在 row < validM 上）
  y[batch] = batchSum
```

存储分配（都用满/留足余量）：

| 存储 | 用法 | 占用 |
| --- | --- | --- |
| L0C 128KB | 分片 `128×128×4 = 64KB` ⇒ `dbL0C = 2`（Mmad 与 Fixpipe 可重叠） | 100%（含双缓冲） |
| L1 512KB | 框架自己按 `stepM/stepN/stepKa/depthA1` 铺（`TCubeTiling结构体.md:18-19,48`）；本版给的形状让 `stepN` 最多可到 `n/baseN` | 框架支配 |
| UB 192KB | `cUb` 2 片 = 128KB（一次读满） + `groupMax` 3×256×4 = 3KB + 3 个小缓冲 | ≈ 136KB |
| GM staging | 每 worker `baseM×n×4`（n=8192 → 4MB），块数 = `min(b, 2*blocks)` | b=1 时 4MB；b=48 时 192MB |

## 四、为什么正确性不受影响

1. **跨分片只做 Max**（max 可交换可结合）⇒ 与分片产出顺序**完全无关**；
   单核 M = `validM ≤ baseM` ⇒ 一个行块内只有一个 M 块，连迭代顺序都不用关心。
2. **N 尾块**：连续写在尾片按 `validN` 紧凑打包、行首不保证 32B 对齐，
   所以走基线已实测的标量逐行读法（`123b2b8` 的结论）。
3. **两级归约对任意 baseN 都对**：完整 64 列组 + 余列组（余列 `mask` 为该组列数、
   `srcRepStride = baseN/8`），第二级用元素级 `Max` 折组 —— 与 baseN 是不是 64 的倍数无关。
4. **行和只在 `row < validM` 上做**，最后每 batch 只写一个 float。
5. **不信任 tiler**：回读校验只有三条必要条件（`baseM ≤ 128`、`baseN ≤ 128`、`baseN % 8 == 0`）；
   请求阶梯全部由 `SetFixSplit/GetTiling` 返回值驱动，末档 = 基线配置 ⇒ 不会因分片请求 abort。

## 五、上一版被拒的根因（本版保留的修复）

真机回报 `Expected 75 launches, got 110` ⇒ **`110 = 75 + 7×5`：约 7 个 case 每轮都 abort，
每次 abort 被 `<ReplayOnce>` 重放、多算一个 launch**。三条自伤逻辑已修：
分片阶梯下限 64→**8**、回读校验删掉多余的 `baseM ≥ min(splitM,m)`、
内部同步超时 3s→**60s**。修复后 abort 面 = 基线原有 7 处 + 2 处构造上不可达的兜底。

## 六、本地验证（都重跑过）

| 门禁 | 结果 |
| --- | --- |
| `tools/launch_check.py` | 通过：6 处启动点全在互斥分支、无同块并联；列出全部 `Fail()` 复核点 |
| `tools/reduce_oracle_gm.py` | **25 组 shape × 2 档配置**（方阵档 + 基线 `16×N` 兜底档）与 golden **逐位一致**（含 `baseN<64`、`baseN` 非 64 倍数、`n<baseN`、`m%baseM!=0`、`n=8192`）；反向 5 个变体**全部检出**：分片步进按补齐尺寸、尾片行距误用 baseN、漏尾片、行和多算残留、**折叠方式用错** |
| `tools/syntax_check.py` | 基线 ok / 本版 ok（本版比基线少一个 warning） |
| `tools/api_delta.py` | 新用 `IterateAll`、`WholeReduceMax`、`HardEvent::V_MTE2`；不再用 `Iterate`/`GetTensorC`/`ReduceMax` |

**没有做（本机做不到）**：真机编译、上板精度、真机用时、`IterateAll` 在 MIX kernel 里的实际行为、
连续写的分片步进到底按有效尺寸还是补齐尺寸。

## 七、上板必查（详细版在 `kernel.asc` 末尾 VERIFY，共 10 条）

0. 日志里**是否还有 `<ReplayOnce>` / `status 134` / launch 数超标**；若再次超标，
   把新的差值给我（`差值 ÷ 迭代次数 = 出问题的 case 数`）。
1. **`IterateAll` 能不能用**（编译期就能看出来：编译过了就说明重载被接受）。
2. **只有 `m % baseM != 0` 的 case 错** ⇒ 连续写的步进是**补齐尺寸**而不是有效尺寸：
   把 kernel 里两处 `validM * baseN`（片偏移、`cStage` 片基址）改成 `baseM * baseN` 即可。
3. **两级归约参数**（`mask=min(64, 余列)`、`srcRepStride=baseN/8`；第二级用元素级 Max）。
   失败特征：行最大值取到别的行/别的组。
4. 打印 `baseM/baseN/depthA1/depthB1/stepN`，确认框架把 512KiB L1 用起来了。
5. UB 136KB / 192KB、staging 分配成功与否。

## 八、第 8 个点 RE（status 134）的诊断与本版修复

真机日志（profiling 阶段）：
`MatmulMaxKernel<bf16, true, true>` 的某个用例报
`<ReplayOnce> Kernel run on device 0 No. 1 time failed.` → `Child process exited with status 134`
→ `Get profiling data failed.`。同一模板**前面 Block Dim 4 的用例跑完 5 次都正常**，
崩的那次是 **Block Dim 9** 的另一个 shape ⇒ 是**随 shape 变化**的失败，不是模板本身。

`134 = SIGABRT` 只可能来自 host 侧 abort。v5 相比基线新增的、随 shape 变化的 abort 路径有三条，
本版把最可能的两条堵死：

| # | 路径 | 为什么随 shape 变化 | 修法 |
| --- | --- | --- | --- |
| 1 | **staging 的 `aclrtMalloc` 失败** → `CheckAcl` → `abort` | v5 的 workspace 是 `min(b,2·blocks) × baseM × n × 4B`：n 大、b 大时要到几百 MB（基线只有 ~1.2MB，**大了两个数量级**） | 总占用按 **64MB 预算收缩 `blocks`**；`aclrtMalloc` 失败**继续砍 `blocks` 重试**，只剩 1 块还失败才 Fail |
| 2 | **连续写越过 staging 尾部** → 设备越界 → ACL 报错 → `CheckAcl` → `abort` | 只有 `m % baseM != 0` 的用例才会让"补齐尺寸步进"的写超出（`validM < baseM`） | staging 每块多留**一个分片**余量：`baseM*(n+baseN)`；并把非活动 worker 的槽位下标**夹到 0**，保证任何 worker 算出的地址都合法 |
| 3 | 分片阶梯全败 → `Fail()` | 需要 tiler 连 `(16, 128/64/…/8)` 全拒 | 保留（构造上几乎不可达） |

**下一步的判定方法**（若还崩）：把崩的那个 case 的 `b/m/n/k` 发我。因为
staging 是"每 worker 放一个行块的整张 C"，真正的根治手段是**按 N 窗口分块**
（这样每块只放 `baseM × N窗口`）—— 但 B 只有在 `TRANSPOSE_B=true` 时列窗口才是连续地址，
所以那是个带条件分支的设计，我不想在没拿到 shape 前先猜。

## 九、第二次 launch 数超标（Expected 75, got 110）的修复：回到"不比基线更严"

真机又报同一条规则，且**数字与 v2 那次完全一样**（110 = 75 + 35 = 7 个 case × 5 次迭代）。
中间换过完全不同的内核结构、也堵过内存/越界，数字没动 ⇒ 病根是**各个版本共享的 host 侧逻辑**，
而且与"内核快慢"无关。逐条比对队友那版真机 15/15 的基线：

| 项 | 基线（15/15） | 我中间几版 | 后果 |
| --- | --- | --- | --- |
| baseN 上界 | **256** | **128** | 比基线更严：tiler 对某些 shape 只肯给更大 baseN 时 → 我方 `Fail()` → abort → `ReplayOnce` → **每次重放多算一个 launch** |
| 回读校验 | **没有**（拿到什么用什么） | 有 3~4 条 `Fail()` | 凭空多出 abort 面 |
| 分片阶梯 | 单次请求，失败即失败 | 多档退让（但档位写法有下限/上界问题） | 阶梯本身不该成为 abort 来源 |

**本版改法**（原则：**永不比基线更严**，回读值只当几何用）：

1. `MAX_TILE_N` 回到 **256**（= 基线口径），UB 按它分配：
   `cUb = 1×128×256×4 = 128KB`（`READ_TILES` 由 2 降到 1，否则超 192KB）。
2. baseN 请求阶梯的**第一档永远是 `splitN` 本身**（= 基线用的口径），
   然后"方阵优先带 128→64→32→16→8"（`baseM≈baseN`、`dbL0C=2`，我们想要的），
   最后"大 baseN 兜底带 256→192→160→96"（只有 tiler 只肯给大 baseN 时才用）。
   ⇒ stage-1 的第一档 = **基线的精确配置** ⇒ 基线能出的 tiling 我们一定能出 ⇒ `chosenN == 0` 不可达。
3. 回读校验只留三条必要条件：`baseM ≤ 128`、`baseN ≤ 256`、`baseN % 8 == 0`。
4. 归约对任意 8 对齐的 `baseN` 都对（完整 64 列组 + 余列组，第二级元素级 `Max` 折组）——
   本地 Oracle 已覆盖 `baseN=256`（组数 4）与 `baseN<64`（组数 1）。

**若修订后仍是 110**：说明这 7 个 case 不是 abort 而是别的（例如某 shape 上设备侧 hang，
被平台按超时重跑）。这时请给我**崩的那个 case 的 `b/m/n/k`**，或日志里 `<ReplayOnce>` 前后
那一行 kernel 名 + `Block Dim` —— 有了它就能直接定位到具体形状，而不是再猜。

## 十、文件

| 文件 | 说明 |
| --- | --- |
| **`kernel.asc`** | ★ 交付物 |
| `teammate-latest.asc` | 基线原文（真机 15/15）—— 退路 |
| `tools/launch_check.py` | 判题硬规则门禁（单 kernel / 无同块并联 / 列出 Fail 点） |
| `tools/syntax_check.py` + `tools/stubs/` | 语法门禁 |
| `tools/reduce_oracle_gm.py` | 搬运/归约的离线逐位复算 + 反向验证 |
| `tools/api_delta.py` | 相对基线的 API 差集 |

```bash
python tools/launch_check.py kernel.asc      # 判题硬规则（必须 0 退出）
python tools/syntax_check.py kernel.asc      # 语法门禁
python tools/reduce_oracle_gm.py             # 逐位复算（退出码 0 = 全过）
python tools/api_delta.py                    # 新 API 差集
```
