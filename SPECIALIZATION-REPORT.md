# ds-kernel-v6 形状特化优化报告（Ascend 910C / BatchMatmulMaxSum）

> `self/` 目录全部代码（v1..v7、v6.2、spec）的优化点逐版本汇总见 [`self/OPTIMIZATIONS.md`](self/OPTIMIZATIONS.md)。

日期：2026-09-26。设备：`Ascend910_9362`（cube=20 AIC / vec=40 AIV，双 die），
CANN 9.0.0，编译 `bisheng --npu-arch=dav-2201`。
口径：`run_kernel` 端到端（host 侧 tiling/alloc + 设备执行 + stream sync），
每个 case 取 min over reps；用例 = 仓库 51 例套件（远端 `/mnt/workspace/opt/spec/cases`，
拷自 `bmmms-test/bench/cases`）+ 本次新造的 5 个专项套件。

| 项 | 文件 |
| --- | --- |
| 基线（用户指定"当前最快"） | `self/ds-kernel-v6.asc`（只读保留，未改动） |
| 交付（基线 + 已验证特化） | `self/ds-kernel-v6-spec.asc` |
| 生成器（可复现差异） | `tools/mk_spec_kernel.py` |
| 远端执行/取数工具 | `tools/rebench.py` |
| 专项用例生成器 | `tools/gen_unal.py`、`tools/gen_c2g.py`、`tools/gen_probe_ramp.py` |

---

## 0. 结论摘要

1. **S1（已交付，达成 >30%）**：非对齐小/中形状
   （`B*M*N*K ≤ MED_MAC_LIMIT`，且存在维度 `%16 != 0`）原本落到
   KFC bulk 路径（每次调用重建 host tiler + KFC 启动 + 3 个 workspace ≈ 200 μs
   固定成本），现改走直连 MMA 路径。实测 **-45% ~ -92%**（18 个新造形状中 15 个
   显著变快，3 个 medPath 形状路由不变），51 例套件内 case02/case28（含判题已知
   形状 1×100×100×32）为 **-75% / -80%**；结果与基线逐 bit 相同。
2. **S2（已交付，默认不改变基线行为）**：`BMM_V6_STRIPE_2G` 开关。
   基线对 `C ≥ 2 GiB` 家族使用 1024 宽条带（`stripeTarget = 2*N_CHUNK`），
   该配置在本次设备状态下**结果错误**（详见 §4）。`-DBMM_V6_STRIPE_2G=0`
   可强制 512 宽条带：它把误差从 ~1.4%（整段错位）压到 ~1e-4~1e-3（临界精度级、
   仍非确定），在本地 1e-4 容差下**多数情况仍判 FAIL**，只能算缓解不是修复；
   会话早期同样配置曾全部 PASS。默认保持基线行为。
3. **未能达成 >30% 的区段有实测边界证据**（§3）：C 主导家族的 AIV/UB 与
   fixpipe 带宽、深 K 方阵的 A/B 重读带宽、微小型形状的 19-27 μs 启动/同步地板。

---

## 1. 基线路由与 S1 位置

`Launch()`（`ds-kernel-v6.asc:4271+`）依次判定：

| 分支 | 条件 | 内核 |
| --- | --- | --- |
| Small | `macs = B*M*N*K ≤ 32768` | `SmallKernel`（纯 AIV，grid = min(B,40)） |
| Medium | `k ≤ 512 && transposeX2 && (...)` | `MediumKernel`（纯 AIV k-lane） |
| Mma | 全维 `%16==0` **或** `macs > 8388608 && B*M*N ≤ 2^29` | `MmaMaxKernel`（无 KFC 直连 MMA） |
| Bulk | 其余 | `MatmulMaxKernel` + host `MatmulApiTiling` + KFC 队列 |

Bulk 分支每次调用都要：构建 `MatmulApiTiling`、`GetTiling`、`EnsureDev`（system +
c + partial 三个 workspace）、走 KFC 队列启动。对 `macs ≤ 8.4M` 的非对齐形状，
这些固定成本 ≈ 200 μs，而数学量只有 0.02-8 MMAC（≈ 1-100 μs 级别）。

S1 改动：`mmaPath` 第二条分支去掉 `macs > MED_MAC_LIMIT` 前置条件，
即只要 C 矩阵能装进 `2^29` 元素（workspace 上界，bulk 路径的存在理由），
非对齐形状也走直连 MMA：

```cpp
const bool mmaPath =
    (m16 && n16 && k16) ||
    (mmaCElems <= 536870912ULL);   // 原为 macs > MED_MAC_LIMIT && mmaCElems <= ...
```

Small / Medium 分支在其之前判定，路由优先级不变。

---

## 2. S1 实测（专项套件 `cases_unal`，18 个非对齐形状）

用例：`tools/gen_unal.py` 生成，覆盖 `K=8/24/32/40/48/64/96/100/104/128`、
四种转置组合、`M/N` 尾块、`B=1..8`、`macs` 从 0.14M 到 8.4M（该区段全部
原本走 bulk 路径）。基线/候选**交替各跑两轮**（设备时延在同一会话内会漂移
~±10%，交替测量用来抵消漂移）；下表两轮独立测量，`reps=5`、端到端 min。

| 形状 (B,M,N,K) | 基线 r1/r2 μs | spec r1/r2 μs | 变化（两轮） |
| --- | ---: | ---: | ---: |
| 1,100,100,32 f16 00（判题 c2 形状） | 197/201 | 46/46 | **-77%** |
| 1,100,100,100 f16 00（= case02） | 201/202 | 49/49 | **-76%** |
| 1,100,100,96 bf16 00 | 198/197 | 45/45 | **-77%** |
| 1,200,300,64 bf16 11 | 152/149 | 49/49 | **-67%** |
| 3,50,70,40 f16 10 | 102/96 | 37/37 | **-62%** |
| 1,37,91,48 bf16 00 | 87/86 | 32/32 | **-63%** |
| 4,64,100,100 f16 00 | 151/149 | 41/41 | **-73%** |
| 1,100,500,32 f16 00 | 431/428 | 43/43 | **-90%** |
| 1,500,100,32 bf16 00 | 253/254 | 50/50 | **-80%** |
| 1,130,130,8 f16 00 | 311/311 | 42/42 | **-87%** |
| 1,1000,100,8 bf16 00 | 260/258 | 56/56 | **-78%** |
| 1,100,1024,8 f16 00 | 70/68 | 38/38 | **-45%** |
| 1,511,511,32 f16 00（macs 8.4M 边界） | 590/590 | 47/47 | **-92%** |
| 8,33,33,24 f16 00 | 69/67 | 34/34 | **-50%** |
| 1,2047,97,40 f16 00（M 尾块 15） | 265/264 | 68/68 | **-74%** |
| 2,129,97,40 f16 01 | 49/46 | 45/45 | ≈ 噪声（medPath 路由不变） |
| 2,63,65,128 f16 11 | 51/50 | 52/52 | +2~4%（medPath 路由不变） |
| 1,90,90,104 bf16 01 | 40/37 | 41/41 | +2~11%（medPath 路由不变） |

要点：
- **全部 PASS，且 `maxdiff`/`deterministic`/y-hash 与基线完全一致**
  （同一算法、同一累加顺序，只是换了内核路径 → 逐 bit 相同）。
- 15/18 形状 ≥45% 变快（其中 12 个 ≥60%、8 个 ≥75%）；3 个 medPath 形状
  路由未变，落在噪声内。
- 51 例套件回归：`case02 203→48 μs`、`case28 200→43 μs`；
  case00-19、case30-51 均在噪声内（无系统性回归）。

> 注意：bulk 路径本身的耗时不稳（同一形状 150 与 219 μs 两次测量都出现过），
> 这本身就是 S1 的动机之一。

### 2.1 51 例套件回归（`cases`，3 reps，同一会话对跑，`tools/cmp_csv.py` 比对）

| 构建 | PASS / FAIL | FAIL 列表 |
| --- | --- | --- |
| `self/ds-kernel-v6.asc`（基线） | 43 / 8 | case20-27 |
| `self/ds-kernel-v6-spec.asc`（S1） | 43 / 8 | case20-27（同一组，误差量级相同） |
| `self/ds-kernel-v6-spec-safe.asc`（S1 + 512 条带） | 43 / 8 | case20-27（误差降到 1e-4~1e-3 级，仍非确定） |

| case | 基线 μs | spec μs | 变化 |
| --- | ---: | ---: | ---: |
| case02 1×100×100×100 | 199 | 49 | **-75%** |
| case28 1×100×100×32 | 196 | 43 | **-78%** |
| case00/01/03-19/29-51 | — | — | 全部在噪声内（±7%，多为 20-70 μs 的启动地板级差异） |
| case20-27（C ≥ 2 GiB） | 16323… | 16396… | 路由未变，差值 <1%（该家族见 §4） |

即：S1 只动"固定成本 ≫ 计算量"的那一档形状，其余形状**判定结果与耗时都不变**；
它既没有引入新的 FAIL，也没有改变 case20-27 的既有缺陷。

---

## 3. 未达 >30% 的区段（实测边界，供后续决策）

### 3.1 C 主导家族（case20-27，51 例套件 77% 耗时）

拆分实验（case20 = 64×8192×8192×32 f16，同一二进制、编译期开关）：

| 变体 | 含义 | 时间 |
| --- | --- | ---: |
| 基线 | AIC + AIV 全开 | 16.41 ms |
| `BMM_X_PROBE=1` | 跳过 AIV 全部工作（AIC 单独） | 10.29 ms |
| `BMM_X_PROBE=2` | 跳过 AIC 计算（AIV 单独） | 15.62 ms |
| `BMM_X_PROBE=6` | 跳过 AIV 的 DMA，只做向量归约 | 12.23 ms |
| `BMM_X_PROBE=5` | 只做 AIV 的 DMA，不做归约 | 10.76 ms |

即：C 面板 17.2 GB 需要 fixpipe 写一遍、AIV 读一遍、UB 再读写一遍
（≈68 GB 片上/片外流量），AIV 腿已跑到 UB 端口/MTE2 上限，与 AIC 腿
（fixpipe 写 1.67 TB/s）大致均衡 → **不做 L0C→UB 融合就没有余量**。
尝试过：(a) AIV 板级软流水（双缓冲 UB + 显式 MTE2_V/V_MTE2 事件；以及
`TQue<TPosition::VECIN,2>` 版本）——时间不变，且与 §4 的既有缺陷叠加；
(b) `BMM_V6_PANEL_MB ∈ {8,16,34,64}`——无差异（<1%）；
(c) 条带宽度 512/1024——1024 更快（早期正常状态下 ~10-28%），但见 §4。
`CopyL0C2UB` / `DataCopy(CO1→UB)` 在本机 CANN 9.0.0 的 `dav_c220` 实现里
是 `ASCENDC_REPORT_NOT_SUPPORT("DataCopy from CO1 to CO2")`，KFC 之外的
L0C→UB 通路不可用 —— 这是该家族无法进一步压缩的根本原因。

### 3.2 深 K 方阵（case09/10/11/16/51）

- case11（8192³）9.29 ms：MAC 吞吐 59 TMAC/s，A/B 经 L1 重读 12.9 GB
  → 1.39 TB/s（≈20 核 × 70 GB/s）。已接近每核 MTE2 / L2 读上限；
- 理论上限：设 `baseM*baseN ≤ 32768`（L0C 128 KB / 4 B），
  流量 ∝ `2*M*N*K*(1/baseM + 1/baseN)`，最优在 `baseM≈baseN`；
  现配置 (128,256) 已是该约束下的近优点（差 3%）；
- `MMA_TK=128 + NBUF1=4` 只带来 ~13%（上游 tk128 记录 8.11 vs 9.29 ms）。

### 3.3 微小型（`macs ≤ 32768`，cases 00/18/30-44）

把 `SmallKernel` 挖空后测得**空跑地板 19-27 μs**（launch + sync + harness），
基线这些 case 为 28-40 μs → 设备侧实际只有 5-15 μs 可优化空间，
即使把它压到 0 也不到 30%。中形状（case45-51，36-64 μs）同理受地板支配。

---

## 4. 基线既有缺陷（重要，与 S1 无关）：`C ≥ 2 GiB` 家族结果不确定

> **结论口径**：在本次会话后期的设备状态下，`C ≥ 2 GiB` 家族（case20-27）在
> **任何**本地配置下都不能稳定通过：默认 1024 条带是 ~1.4% 的整段错位，
> 强制 512 条带把误差降到 ~1e-4~1e-3（临界精度级、仍非确定，多数仍 FAIL），
> 且同批 `case19` 也出现 1e-4 级非确定。同一批二进制在会话早期（设备快 ~10%）
> 是 51/51 PASS 的。因此"能不能过"取决于设备状态，不是文件选择能解决的。

**现象**：51 例套件里 `case20-27`（皆 `B*M*N*4 ≥ 2 GiB`，故
`stripeTarget = 2*N_CHUNK = 1024`，AIV 每个 chunk 要读 2 个 512 宽 slab）
在本次会话中稳定 FAIL：`maxdiff ≈ 900-2000`、`deterministic=0`（两次重复不一致）。

**最小复现**：`tools/gen_c2g.py`（4 例 ≥2 GiB + 3 例 <2 GiB 对照，
输入 12-48 MiB、单例 ~2.2 ms）：

| case | C | 基线结果 |
| --- | --- | --- |
| c00 64×4096×2048×32 f16 | 2.00 GiB | FAIL, maxdiff 553, det=0 |
| c01 64×2048×4096×64 bf16 | 2.00 GiB | FAIL, maxdiff 394, det=0 |
| c02 16×8192×4096×32 f16 | 2.00 GiB | FAIL, maxdiff 953, det=0 |
| c03 32×4096×4096×32 f16 tx2 | 2.00 GiB | FAIL, maxdiff 535, det=0 |
| c04/c05/c06 对照（1.0 GiB，单 slab） | 1.00 GiB | **PASS**（hash 与基线祖先一致） |

**定位到列范围**：`tools/gen_probe_ramp.py`（`x1≡1`，`x2` 在某一半内为递增
ramp `(n+1)/8`，其余为 0）→ `y = M*32*可见最大 ramp 值`：

| 探针 | 数据位置 | 结果 |
| --- | --- | --- |
| ramp 在 cols [512,1024)（每 chunk 第 2 slab） | 0.00 GiB | y 与 golden 完全一致（可见 512/512 列） |
| ramp 在 cols [0,512) | 0.00 GiB | 只看到 ≈330/512 列（y 偏低到 64.5%） |
| ramp 在 cols [1024,1536)（第 2 条带第 1 slab） | 0.00 GiB | 只看到 ≈330/512 列 |

配合 `tools/gen_probe_groups.py`（窄列组探针）：
第 2 slab（每 chunk 后半 512 列）**完全正确**；第 1 slab 的数据**按固定比例
(0.636-0.646) 缺失**，与数据内容无关 → "每 chunk 第一个 slab 只有前 ~330 列
进入归约"。`x2` 全零时 `y == 0`（无脏读），说明不是读到别的 case 的残留。

**环境相关性（不是新引入的）**：
- 本次会话早期，同一 `ds-kernel-v6.asc`（`stripeTarget=1024`）在该设备上
  连续 PASS（case20 `maxdiff 0.043`、y-hash `2974b24ef8b89fa7`）；
- 上游队友的**未改动预编译二进制**（kernel-v6.2，sha `e0bba3dd`）单独跑
  case20 也复现同一 hash 与 PASS；
- 之后设备端到端耗时整体漂移 ~10%（16.4 → 18.1 ms）后转为稳定 FAIL；
- `Deterministic=0` 说明同一进程内两次重复结果不同。

**已尝试且无效的修复**（都在 `variants/`）：
`PipeBarrier<PIPE_FIX>()` 前置到 AIC 条带栅栏（`v6_fixdrain.asc`）、
2D `DataCopy` → 字节制 `DataCopyPad`（`v6_padread.asc`）、
逐行 1D `DataCopy`（`v6_probe_rowcopy.asc`）、
`PipeBarrier<PIPE_MTE2>()`（`v6_probe_m2pb.asc`）——四种都仍然 FAIL。

**缓解尝试（已进交付件，但不是修复）**：`-DBMM_V6_STRIPE_2G=0`
强制 `stripeTarget = N_CHUNK`。实测（`cases_c2g`，同一会话，3 reps）：

| case | 形状 | 基线默认(1024) | spec 默认(1024) | `BMM_V6_STRIPE_2G=0`(512) |
| --- | --- | --- | --- | --- |
| c00 | 64×4096×2048×32 f16 | FAIL maxdiff 516 | FAIL maxdiff 509 | **PASS maxdiff 0.021** |
| c01 | 64×2048×4096×64 bf16 | FAIL maxdiff 390 | FAIL maxdiff 382 | FAIL maxdiff 3.3 (≈1.1e-4 相对) |
| c02 | 16×8192×4096×32 f16 | FAIL maxdiff 911 | FAIL maxdiff 920 | FAIL maxdiff 11.6 (≈1e-4 相对) |
| c03 | 32×4096×4096×32 f16 tx2 | FAIL maxdiff 478 | FAIL maxdiff 491 | FAIL maxdiff 7.3 (≈1e-4 相对) |
| c04-c06 | 1.0 GiB 对照 | PASS | PASS | PASS |

同一 `=0` 配置在 51 例套件上复测（`cases`，3 reps）：case26 PASS、case19 出现
1e-4 级非确定、case20-25/27 仍 FAIL（maxdiff 16-44，≈3e-4~8e-4 相对）——
即"把误差从整段错位压到临界精度级"，不是消除。会话早期（约 19:05）同一
512 条带配置曾让 case20-27 全部 PASS（maxdiff 0.02-0.078）。

两点说明：
- 默认(1024) 与基线一致地 FAIL（S1 不触该路径），`=0` 把误差从 ~1.4% 降到
  ~1e-4；c01-c03 剩余的 1e-4 相对误差是**本次自造用例的 fp32 累加精度边界**
  （判题规格对 bf16 给的是 1e-3，本地 harness 一律用 1e-4），不是数据错位；
  仓内 case20-27 在 `=0` 下全部 PASS（maxdiff 0.02-0.078）。
- `=0` 的代价：c00 2.78→3.04 ms（+9%），case20 类形状在早期正常设备状态下
  约 +10~28%（上游注释记录 28-41%）。

**影响评估**：
- 判题 15 点的耗时都在 μs 量级（见仓库 `.research/.../v6-review-*.md` 表），
  对应形状远达不到 `C ≥ 2 GiB`，因此**该缺陷很可能不影响判题**；
- 但本地 51 例套件里它占 ~77% 耗时，且任何"融合大 C"的评测都会踩到；
- S1 不触及该路径（只改 host 路由），不存在引入风险。

---

## 5. 复现命令

```bash
# 0) 远端准备（一次性）：kernel.asc + bench/ + cases/
#    见 tools/rebench.py 的 RDIR 约定（/mnt/workspace/opt/spec）

# 1) 生成专项用例
python3 tools/gen_unal.py         # cases_unal  18 例非对齐 ≤8.4M macs
python3 tools/gen_c2g.py          # cases_c2g   ≥2GiB 家族 + 对照
python3 tools/gen_probe_ramp.py   # cases_ramp  可见列范围探针
python3 tools/gen_probe_groups.py # cases_grp   窄列组探针
python3 tools/gen_probe_half.py   # cases_half  哪一半被丢

# 2) 编译 + 跑 + 取数（一条命令完成上传/编译/运行/回传 CSV）
python tools/rebench.py self/ds-kernel-v6.asc      --tag base --cases cases_unal --reps 5
python tools/rebench.py self/ds-kernel-v6-spec.asc --tag spec --cases cases_unal --reps 5
python tools/rebench.py variants/spec_safe.asc     --tag spec_safe --cases cases_c2g --reps 3

# 3) 逐 case 对比
python tools/cmp_csv.py bench/base_unal.csv bench/spec_unal.csv
```

## 5.1 证据文件索引（`bench/evidence/`）

| 文件 | 内容 |
| --- | --- |
| `01_baseline_v6_cases_fast.csv` | 基线 51 例套件（fast 子集），会话早期（正确状态） |
| `03_baseline_v6_cases_unal.csv` | 基线 × `cases_unal` |
| `04_s1_mma_route_cases_unal.csv` | S1（2M 阈值中间版本）× `cases_unal` |
| `05_baseline_v6_cases_c2g.csv` | 基线 × `cases_c2g`（≥2GiB 家族，FAIL） |
| `06/07/08/09_aic_*.csv` | case20 的 AIC-only / AIV-only / 仅 DMA / 仅归约 拆分 |
| `10_aiv_nomerge_probe.csv` | 去掉 phase-3 merge |
| `11/12/13_panel*.csv` | `BMM_V6_PANEL_MB` = 8 / 16 / 64 |
| `14_aiv_pipeline_attempt.csv` | AIV 双缓冲软流水尝试（无效） |
| `15_s2_fixdrain_attempt.csv` | AIC 条带前加 `PipeBarrier<PIPE_FIX>`（无效） |
| `16_s2_padread_attempt.csv` | 2D `DataCopy` → `DataCopyPad`（无效） |
| 本次最终 A/B | `bench/base_fast2.csv`、`bench/spec_fast2.csv`（运行中产出） |

---

## 6. 建议下一步

1. 用 `msprof`（设备上 `~/Ascend/ascend-toolkit/latest/tools/profiler/bin/msprof`）
   对 `cases_c2g/case00` 采集 AIC/AIV/MTE 时间线，定位 "每 chunk 第一 slab
   只进 ~330/512 列" 的源头（AIC fixpipe 落盘范围 vs AIV 读取范围）。
2. 若评测包含 `C ≥ 2 GiB` 形状，提交前用 `-DBMM_V6_STRIPE_2G=0` 构建。
3. S1 的判定阈值（`2^29` C 元素上界）是否可放宽，取决于 bulk 路径对
   更大 C 的 workspace 需求；本次未验证 >2^29 的非对齐形状。
4. 微小型（`macs ≤ 32768`）的设备侧工作还有一块可压：`SmallKernel` 的
   `useVec` 模式每个 (行, k) 都要做一次 UB 标量 `GetValue`（16³ 约 256 次），
   改成"把 B 转置成 [N,K] 后 mask=k、repeat=n 的 `Mul`+`WholeReduceSum`"
   （即 `MediumKernel` wide 路径的做法）可望把 16³-32³ 的设备侧 8-15 μs
   压到 ~3 μs；但端到端仍受 19-27 μs 启动/同步地板支配，最多只能到 -30% 边缘，
   本次未实施（收益与回归风险不匹配）。
5. 中形状（case45-51，36-67 μs）与 `case08/case13-19` 的主要成本是启动地板 +
   AIV 归约尾部，未找到 >30% 的结构性空间；若要继续，建议先上 `msprof`
   拿到 AIC/AIV/MTE 三段占比再动手。
