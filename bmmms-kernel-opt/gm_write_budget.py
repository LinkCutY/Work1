#!/usr/bin/env python3
"""gm_write_budget.py — 方向3「少写 GM」的量化对照表（离线，纯公式）

对照对象：
  基线 = 仓库 123b2b8（15/15）：任务=per batch，baseM=16，baseN=ceil16(N)≤256，
         C 走 GM（enSequentialWrite=true，尾块按 validN 紧凑写），y 每 batch 写 1 次，
         host 还有一次 aclrtMemset(y)。
  v2   = 本工作区的候选：baseM=min(128,ceil16(M))、baseN=min(128,ceil16(N))，
         任务=(batch,mwin[,group])，C 同样走 GM，partial 每 task 写 1 个 float
         （groups>1 时写 baseM 个），nmwin==1&&groups==1 时直接写 y、跳过 phase 2。

不可压缩的部分（先说清）：2201 上不存在 L0C→UB 直通（Fixpipe 的 CO→UB 原型仅
Atlas 200I/500 A2 可用），AIV 与 AIC 通过 GM 传递数据（NPU架构版本220x.md:22），
所以 **C 矩阵必须写 GM 一次、再读回一次**，字节量 = B*M_pad*N_pad*4 与 tiling 无关。
本脚本要证明的是：**写的"次数"（事务数/同步点数）被压下来了**，附带写也被压到极小。
"""
from __future__ import annotations

import argparse


def cd(a, b):
    return (a + b - 1) // b


def al(x, a):
    return cd(x, a) * a


def baseline(b, m, n, k, base_m=16, base_n_cap=256):
    bm, bn = base_m, min(base_n_cap, al(n, 16))
    tiles = b * cd(m, bm) * cd(n, bn)          # = GetTensorC 次数 = C 的 GM 写次数
    pad_m, pad_n = cd(m, bm) * bm, cd(n, bn) * bn
    c_bytes = b * pad_m * pad_n * 4            # C 写字节
    extra = b * 4                              # y：每 batch 1 个 float
    memset = b * 4                             # aclrtMemset(y)
    return dict(tiles=tiles, c_bytes=c_bytes, extra_writes=b, extra_bytes=extra + memset)


def crossings2side(b, m, n, k, base_m=128, base_n=128, chunk_n=0, stage_kb=128):
    """**两侧都算**的 GM 往返次数（这才是"经 GM 的次数"）：
       AIC 侧：每 chunk 一次写（宽块 = IterateAll 一次；窄块 = 每 tile 一次）
       AIV 侧：每 band 一次读（band 行数 = 暂存字节 / (ubPitch*4)）"""
    bm, bn = min(base_m, al(m, 16)), min(base_n, al(n, 16))
    cw = (min(chunk_n, n) if chunk_n else bn)
    nmwin = cd(m, bm)
    tasks = b * nmwin
    chunks = cd(n, cw)
    ubpitch = al(min(cw, n), 8)
    band_rows = max(1, (stage_kb * 1024) // (ubpitch * 4))
    row_tiles = cd(m, bm)          # 每个 task 的行数上限
    rows_per_task = min(bm, m)
    bands = cd(rows_per_task, band_rows)
    write = tasks * chunks
    read = tasks * chunks * bands
    return dict(write=write, read=read, total=write + read,
                chunks=chunks, bands=bands, band_rows=band_rows)


def crossings(b, m, n, k, base_m=128, base_n=128, chunk_n=0):
    """AIC↔AIV 经 GM 的**往返次数**（本次优化目标）：
       每个 task 每 chunk 一次 AIC→GM 写 + 一次 GM→UB 读。
       chunk_n=0 → 窄块（每 chunk = 一个 baseN tile）= 已验证模式。"""
    bm, bn = min(base_m, al(m, 16)), min(base_n, al(n, 16))
    cw = (min(chunk_n, n) if chunk_n else bn)
    nmwin = cd(m, bm)
    tasks = b * nmwin
    per_task = cd(n, cw)
    return dict(tasks=tasks, chunks_per_task=per_task,
                crossings=tasks * per_task, chunk_width=cw)


def v2(b, m, n, k, base_m=128, base_n=128, workers=48, mult=1):
    bm, bn = min(base_m, al(m, 16)), min(base_n, al(n, 16))
    nmwin, nchunk = cd(m, bm), cd(n, bn)
    m_tasks = b * nmwin
    want = mult * workers
    gtarget = (want // m_tasks) if want > m_tasks else 1
    groups = max(1, cd(nchunk, max(1, cd(nchunk, gtarget))))
    tasks = b * nmwin * groups
    tiles = b * nmwin * nchunk                 # = GetTensorC 次数 = C 的 GM 写次数
    c_bytes = b * cd(m, bm) * bm * cd(n, bn) * bn * 4
    direct = (nmwin == 1 and groups == 1)
    if direct:
        extra_writes, extra_bytes = b, b * 4   # 直接写 y
        phase2 = 0
    elif groups == 1:
        extra_writes, extra_bytes = b * nmwin, b * nmwin * 4      # 每 task 1 float
        phase2 = b * nmwin * 4
    else:
        extra_writes = tasks
        extra_bytes = tasks * bm * 4                              # 每 task baseM 个 float
        phase2 = b * nmwin * groups * bm * 4
    return dict(tiles=tiles, c_bytes=c_bytes, extra_writes=extra_writes,
                extra_bytes=extra_bytes, groups=groups, nmwin=nmwin,
                direct=direct, phase2_bytes=phase2)


SHAPES = [(1, 8192, 8192, 8192), (1, 8192, 8192, 1024), (1, 4096, 4096, 4096),
          (1, 1024, 1024, 1024), (1, 512, 512, 512), (8, 512, 512, 512),
          (64, 256, 256, 256), (1, 8192, 64, 8192), (1, 128, 8192, 1024),
          (1, 1000, 1000, 1000)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-m", type=int, default=128)
    ap.add_argument("--base-n", type=int, default=128)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--mult", type=int, default=1)
    ap.add_argument("--ub-stage-kb", type=int, default=128)
    a = ap.parse_args()
    a.ub_stage_kb = a.ub_stage_kb
    print("GM 写：基线(123b2b8: baseM=16,baseN<=256) vs v2 "
          f"(baseM={a.base_m},baseN={a.base_n},workers={a.workers},mult={a.mult})")
    print(f"{'shape(B,M,N,K)':>26} | {'基线 tiles':>10} {'v2 tiles':>9} {'倍数':>6} "
          f"| {'基线 附带写':>11} {'v2 附带写':>10} | 模式")
    tot_ratio = []
    for (b, m, n, k) in SHAPES:
        x, y = baseline(b, m, n, k), v2(b, m, n, k, a.base_m, a.base_n, a.workers, a.mult)
        r = x["tiles"] / max(1, y["tiles"])
        tot_ratio.append(r)
        mode = ("直接写y" if y["direct"] else f"phase2(groups={y['groups']})")
        print(f"{f'{b},{m},{n},{k}':>26} | {x['tiles']:>10} {y['tiles']:>9} "
              f"{r:>5.1f}x | {x['extra_writes']:>11} {y['extra_writes']:>10} | {mode}")
    print("\n=== AIC/AIV 经 GM 的往返次数（两侧都算：AIC 写 + AIV 读）===")
    print(f"{'shape':>20} | {'基线(16x256)':>12} | {'窄块128²':>9} | {'宽块896':>8} "
          f"| {'宽块2048':>8} | 宽块 vs 基线")
    for (b, m, n, k) in SHAPES[:6] + [(1, 128, 8192, 1024), (1, 8192, 8192, 8192)]:
        bl = baseline(b, m, n, k)
        base2 = 2 * bl["tiles"]                      # 基线与 v2 都是每 tile 一写一读
        n0 = crossings2side(b, m, n, k, a.base_m, a.base_n, 0)["total"]
        w896 = crossings2side(b, m, n, k, a.base_m, a.base_n, 896)["total"]
        w2k = crossings2side(b, m, n, k, a.base_m, a.base_n, 2048)["total"]
        print(f"{f'{b},{m},{n},{k}':>20} | {base2:>12} | {n0:>9} | {w896:>8} "
              f"| {w2k:>8} | {base2/max(1,w896):>5.1f}x")
    d = crossings2side(1, 8192, 8192, 8192, a.base_m, a.base_n, 896)
    print(f"  （宽块896 明细：每 task {d['chunks']} 次写 + {d['chunks']}×{d['bands']} 次读"
          f"；每 band {d['band_rows']} 行 = {a.ub_stage_kb}KB 暂存）")

    print("\n=== 宽块的代价：GM 窗口分配（blocks*(ratio+1) 个槽 × baseM×min(CHUNK,N)）===")
    print(f"{'CHUNK_N':>8} | {'单槽':>9} | {'总分配(blocks=24,ratio=2)':>22} | N=8192 的往返/task")
    for cw in (0, 512, 1024, 2048, 4096):
        eff = cw if cw else a.base_n
        slot = a.base_m * min(eff, 8192) * 4
        total = slot * 24 * 3
        per_task = cd(8192, min(eff, 8192))
        print(f"{cw if cw else 'baseN':>8} | {slot>>10:>7}KB | {total>>20:>19}MB "
              f"| {per_task:>8}")

    print(f"\nC 写字节（两版相同，架构强制）：见 c_bytes —— 不可压缩项，只降次数")
    print(f"tiles 平均降低 {sum(tot_ratio)/len(tot_ratio):.2f}x；"
          f"基线还额外做 1 次 aclrtMemset(y)，v2 已删除")


if __name__ == "__main__":
    main()
