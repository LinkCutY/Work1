#!/usr/bin/env python3
"""kernel.asc v2 的离线 oracle：验证"任务网格 + 两阶段归约"的语义与覆盖性。

不依赖 NPU/CANN，只用 numpy。验证三件事：
  A. 任务覆盖：所有 task 的 (行窗口 × 列 chunk) 恰好覆盖 [0,M) x [0,N) 一次，
     且 partial 的写入下标与 phase 1 的 task→(batch,mwin,group) 解码一致。
  B. 归约语义：按 v2 的实现路径（chunk 内 64 列一块取最大 → 并入 per-row max；
     groups==1 时 task 内行序求和；groups>1 时 phase 2 先跨 group 取 max 再按
     mwin/row 升序求和）复算 y，与官方 golden（FP64 计算后转 FP32）比，
     判据用官方 verify_result.py 的 np.isclose(rtol=1e-4, atol=1e-4)。
  C. 误差余量：报出 |a-b| / (atol + rtol*|b|)，即离判据还有多少倍余量。

模型口径（与 kernel 对应）：
  - Cube 段：FP64 算相似度后 **cast 到 FP32**（模拟 fixpipe 输出 fp32）。
    fp32 累加本身的影响已由 .research/work/precision_budget.py 单独验证
    （全链路 fp32 相对误差 ≤1.2e-7，余量 ~1000×），此处不重复。
  - Vec 段：max 精确（与顺序无关）；求和按 kernel 的实际结合顺序，用 float32。
"""

from __future__ import annotations

import argparse
import numpy as np

RTOL = 1e-4
ATOL = 1e-4


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def golden(x1, x2, tx1=False, tx2=False) -> np.ndarray:
    """官方语义：FP64 计算 → FP32 输出（与 BatchMatmulMaxSum.py:impl 同）。"""
    a = np.asarray(x1).astype(np.float64)
    b = np.asarray(x2).astype(np.float64)
    a = np.swapaxes(a, -1, -2) if tx1 else a
    b = np.swapaxes(b, -1, -2) if tx2 else b
    sim = np.matmul(a, b)
    return np.sum(np.max(sim, axis=-1), axis=-1).astype(np.float32)


def plan(shape, base_m, base_n, workers, gtarget_ratio=1.0):
    """复算 host 侧的任务网格（与 Launch 中的公式一一对应）。"""
    B, M, N, K = shape
    nmwin = ceil_div(M, base_m)
    nchunk = ceil_div(N, base_n)
    m_tasks = B * nmwin
    groups_target = max(1, int(workers // m_tasks)) if workers > m_tasks else 1
    groups = max(1, ceil_div(nchunk, max(1, ceil_div(nchunk, groups_target))))
    chunks_per_task = ceil_div(nchunk, groups)
    tasks = B * nmwin * groups
    blocks = max(1, min(workers // 2, tasks))
    return dict(B=B, M=M, N=N, K=K, base_m=base_m, base_n=base_n, nmwin=nmwin,
                nchunk=nchunk, groups=groups, chunks_per_task=chunks_per_task,
                tasks=tasks, blocks=blocks, workers=2 * blocks,
                chunk_width=base_n)


# 与 kernel.asc 的 host 逻辑同源：期望宽度 → 门槛 → GM 预算 → baseN 对齐
BMM_CHUNK_N = 2048
BMM_WIDE_MIN_TILES = 8
BMM_WINDOW_BUDGET_MB = 32
BMM_TASK_RATIO = 2


def pick_chunk_width(n, base_m, base_n, blocks):
    slots = (BMM_TASK_RATIO + 1) * max(1, blocks)
    narrow_tiles = ceil_div(n, base_n)
    cw = BMM_CHUNK_N if (BMM_CHUNK_N > 0 and narrow_tiles >= BMM_WIDE_MIN_TILES) \
        else base_n
    per_col = slots * base_m * 4
    cap = (BMM_WINDOW_BUDGET_MB * 1024 * 1024) // per_col if per_col else cw
    cw = min(cw, max(cap, base_n))
    cw = max(base_n, (cw // base_n) * base_n)
    return cw


def check_coverage(p):
    """A 段：任务覆盖互不重叠且完整。

    每个 (batch,row) 恰好被 `groups` 个 task 覆盖（每 group 一个）；
    每一列恰好被 B*nmwin 个 task 覆盖（groups==1 时每个 task 扫全部 N）。
    """
    covered_rows = np.zeros(p["M"], dtype=np.int64)
    covered_cols = np.zeros(p["N"], dtype=np.int64)
    seen = set()
    for task in range(p["tasks"]):
        group = task % p["groups"]
        rest = task // p["groups"]
        mwin = rest % p["nmwin"]
        batch = rest // p["nmwin"]
        key = (batch, mwin, group)
        assert key not in seen, ("duplicate task decode", key)
        seen.add(key)
        row0 = mwin * p["base_m"]
        valid_m = min(p["base_m"], p["M"] - row0)
        covered_rows[row0:row0 + valid_m] += 1
        cw = p.get("chunk_width", p["base_n"])
        for c in range(group * p["chunks_per_task"],
                       min(p["nchunk"], (group + 1) * p["chunks_per_task"])):
            n0 = c * cw
            valid_n = min(cw, p["N"] - n0)
            if valid_n <= 0:
                continue
            covered_cols[n0:n0 + valid_n] += 1
    want_rows = p["B"] * p["groups"]
    want_cols = p["B"] * p["nmwin"]
    ok = (covered_rows.min() == covered_rows.max() == want_rows and
          covered_cols.min() == covered_cols.max() == want_cols)
    return ok, (int(covered_rows.min()), int(covered_rows.max()),
                want_rows), (int(covered_cols.min()), int(covered_cols.max()),
                             want_cols)


def model(x1, x2, p, tx1=False, tx2=False):
    """B 段：按 v2 的实现路径复算 y（float32 求和）。"""
    B, M, N = p["B"], p["M"], p["N"]
    a = np.asarray(x1).astype(np.float64)
    b = np.asarray(x2).astype(np.float64)
    a = np.swapaxes(a, -1, -2) if tx1 else a
    b = np.swapaxes(b, -1, -2) if tx2 else b
    sim = np.matmul(a, b).astype(np.float32)          # Cube 输出 fp32

    partial = {}      # groups==1 -> {(b,mwin): float} ; else {(b,mwin,g): [rows]}
    for task in range(p["tasks"]):
        group = task % p["groups"]
        rest = task // p["groups"]
        mwin = rest % p["nmwin"]
        batch = rest // p["nmwin"]
        row0 = mwin * p["base_m"]
        rows = min(p["base_m"], M - row0)
        row_max = np.full(rows, -3.402823466e38, dtype=np.float32)
        chunk_w = p.get("chunk_width", p["base_n"])
        for c in range(group * p["chunks_per_task"],
                       min(p["nchunk"], (group + 1) * p["chunks_per_task"])):
            n0 = c * chunk_w
            cols = min(chunk_w, N - n0)
            if cols <= 0:
                continue
            # kernel 侧按**行带**扫：每带 rows_band 行、每带内 64 列一块取最大、
            # 立即并入 bestVec[r0:r0+rows_band]（max 精确、与带划分无关）
            # 与 kernel.asc 的自适应暂存同源：
            #   stageBytes = min(BMM_UB_STAGE_KB*1024, baseM*maxPitch*4)，下限 maxPitch*4
            #   maxPitch = align8(min(chunkWidth, N))；band = stageBytes/(ubPitch*4)
            max_pitch = ((min(chunk_w, p["N"]) + 7) // 8) * 8
            stage = min(128 * 1024, p.get("base_m", rows) * max_pitch * 4)
            stage = max(stage, max_pitch * 4)
            ubpitch = ((cols + 7) // 8) * 8
            band = max(1, stage // (ubpitch * 4))
            band = min(band, rows)
            for r0 in range(0, rows, band):
                rb = min(band, rows - r0)
                tile = sim[batch, row0 + r0:row0 + r0 + rb, n0:n0 + cols]
                for sub in range(0, cols, 64):
                    row_max[r0:r0 + rb] = np.maximum(
                        row_max[r0:r0 + rb],
                        tile[:, sub:sub + 64].max(axis=1))
        if p["groups"] == 1:
            s = np.float32(0.0)
            for r in range(rows):
                s = np.float32(s + row_max[r])        # task 内行序求和
            partial[(batch, mwin)] = s
        else:
            partial[(batch, mwin, group)] = row_max

    y = np.zeros(B, dtype=np.float32)
    for batch in range(B):
        total = np.float32(0.0)
        for mwin in range(p["nmwin"]):
            row0 = mwin * p["base_m"]
            rows = min(p["base_m"], M - row0)
            if p["groups"] == 1:
                total = np.float32(total + partial[(batch, mwin)])
            else:
                rmax = np.full(rows, -3.402823466e38, dtype=np.float32)
                for g in range(p["groups"]):
                    rmax = np.maximum(rmax, partial[(batch, mwin, g)])
                for r in range(rows):
                    total = np.float32(total + rmax[r])
        y[batch] = total
    return y


def apply_config(p):
    """把 config 链应用到 plan 结果（覆盖检查与模型都用同一份）。"""
    cw = pick_chunk_width(p["N"], p["base_m"], p["base_n"], p["blocks"])
    p = dict(p)
    p["chunk_width"] = cw
    if cw != p["base_n"]:
        p["nchunk"] = ceil_div(p["N"], cw)
        p["groups"] = 1
        p["chunks_per_task"] = p["nchunk"]
        p["tasks"] = p["B"] * p["nmwin"]
        p["blocks"] = max(1, min(48 // 2, p["tasks"]))
    return p


def run_case(B, M, N, K, base_m, base_n, workers, dtype, tx1, tx2, seed,
             chunk_width=None):
    rng = np.random.default_rng(seed)
    shape = (B, M, N, K)
    if tx1:
        x1 = rng.uniform(-1, 1, (B, K, M))
    else:
        x1 = rng.uniform(-1, 1, (B, M, K))
    if tx2:
        x2 = rng.uniform(-1, 1, (B, N, K))
    else:
        x2 = rng.uniform(-1, 1, (B, K, N))
    if dtype == "fp16":
        x1, x2 = x1.astype(np.float16), x2.astype(np.float16)
    else:                                            # bf16 用尾数截断近似
        x1 = (x1.astype(np.float32).view(np.uint32) & 0xFFFF0000).view(
            np.float32)
        x2 = (x2.astype(np.float32).view(np.uint32) & 0xFFFF0000).view(
            np.float32)

    g = golden(x1, x2, tx1, tx2)
    p = apply_config(plan(shape, base_m, base_n, workers))
    if chunk_width:
        p["chunk_width"] = chunk_width
        p["nchunk"] = ceil_div(N, chunk_width)
        p["groups"] = 1
        p["chunks_per_task"] = p["nchunk"]
        p["tasks"] = B * p["nmwin"]
    y = model(x1, x2, p, tx1, tx2)

    cov_rows, cov_lo, cov_hi = check_coverage(p)
    tol = ATOL + RTOL * np.abs(g)
    err = np.abs(y - g)
    margin = float(np.min(tol / np.maximum(err, 1e-30)))
    ok = bool(np.all(err <= tol))
    return ok, margin, p, cov_rows, (cov_lo, cov_hi), float(err.max())


def atomic_probe(B, M, N, K, base_m, base_n, dtype, tx1, tx2, seed, trials=16):
    """单 launch 兜底模式（BMM_SINGLE_LAUNCH）：groups 强制 1，每个 (batch,mwin)
    的行和用浮点原子加累进 y[batch] → 累加顺序不确定。这里量化"顺序不确定"的
    后果：同一输入下 16 种随机顺序的 y 离散度，以及与 golden 的距离。
    """
    rng = np.random.default_rng(seed)
    if tx1:
        x1 = rng.uniform(-1, 1, (B, K, M))
    else:
        x1 = rng.uniform(-1, 1, (B, M, K))
    if tx2:
        x2 = rng.uniform(-1, 1, (B, N, K))
    else:
        x2 = rng.uniform(-1, 1, (B, K, N))
    if dtype == "fp16":
        x1, x2 = x1.astype(np.float16), x2.astype(np.float16)
    else:
        x1 = (x1.astype(np.float32).view(np.uint32) & 0xFFFF0000).view(np.float32)
        x2 = (x2.astype(np.float32).view(np.uint32) & 0xFFFF0000).view(np.float32)

    g = golden(x1, x2, tx1, tx2)
    p = plan((B, M, N, K), base_m, base_n, 48)
    p["groups"] = 1
    p["chunks_per_task"] = p["nchunk"]
    p["tasks"] = B * p["nmwin"]

    a = np.swapaxes(x1.astype(np.float64), -1, -2) if tx1 else x1.astype(np.float64)
    b = np.swapaxes(x2.astype(np.float64), -1, -2) if tx2 else x2.astype(np.float64)
    sim = np.matmul(a, b).astype(np.float32)

    sums = {}                      # (batch, mwin) -> float32 行和
    for mwin in range(p["nmwin"]):
        row0 = mwin * p["base_m"]
        rows = min(p["base_m"], M - row0)
        for batch in range(B):
            rmax = sim[batch, row0:row0 + rows, :].max(axis=1)
            s = np.float32(0.0)
            for r in range(rows):
                s = np.float32(s + rmax[r])
            sums[(batch, mwin)] = s

    ys = []
    for t in range(trials):
        order = np.random.default_rng(1000 + t).permutation(p["nmwin"])
        y = np.zeros(B, dtype=np.float32)
        for batch in range(B):
            acc = np.float32(0.0)
            for mwin in order:
                acc = np.float32(acc + sums[(batch, mwin)])
            y[batch] = acc
        ys.append(y)
    ys = np.stack(ys)
    spread = float(np.max(np.ptp(ys, axis=0)))
    worst = float(np.max(np.abs(ys - g)))
    tol = float(np.min(ATOL + RTOL * np.abs(g)))
    return spread, worst, tol, float(np.max(np.abs(ys.mean(axis=0) - g)))


def check_phase2_coverage(p):
    """phase 2 的 worker 覆盖检查（评审 Finding 1）。

    phase 2 的 stride 传的是 blocks；实际 AIV 索引范围有两种可能读法：
      AIV_ONLY  → [0, blocks)      （纯 Vector 核被推断为 AIV_ONLY）
      MIX 1:2   → [0, 2*blocks)
    要求：两种读法下每个 batch 都被至少一个 worker 处理（重复处理是幂等的，
    因为两个 worker 算出的和逐位相同、写同一地址同一值）。
    """
    batches, blocks = p["B"], p["blocks"]
    res = {}
    for id_range in (blocks, 2 * blocks):
        covered = set()
        for w in range(id_range):
            for b in range(w, batches, blocks):
                covered.add(b)
        res[id_range] = len(covered) == batches
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()

    cases = []
    for M, N in [(1, 1), (1, 7), (7, 1), (8, 8), (16, 16), (17, 31), (63, 65),
                 (100, 100), (127, 129), (128, 128), (129, 127), (200, 300),
                 (255, 257)]:
        for K in (8, 64):
            cases.append((1, M, N, K))
    cases += [(2, 128, 128, 64), (4, 65, 33, 32), (8, 128, 256, 128),
              (16, 256, 128, 64)]
    cases += [(1, 128, 8192, 256), (1, 8192, 128, 256), (1, 8192, 8192, 64)]

    print(f"{'B':>3} {'M':>5} {'N':>5} {'K':>5} {'dtype':>5} {'tx':>4} "
          f"{'baseM':>5} {'baseN':>5} {'nmw':>4} {'nch':>5} {'grp':>4} "
          f"{'tasks':>6} {'blk':>3} {'cov':>4} {'maxerr':>10} {'margin':>10}  ok")
    bad = 0
    for (B, M, N, K) in cases:
        for dtype in ("fp16", "bf16"):
            for tx1, tx2 in ((False, False), (True, True)):
                base_m = min(128, ((M + 15) // 16) * 16)
                base_n = min(128, ((N + 15) // 16) * 16)
                ok, margin, p, cov_rows, cov_cols, maxerr = run_case(
                    B, M, N, K, base_m, base_n, a.workers, dtype, tx1, tx2,
                    seed=hash((B, M, N, K, dtype, tx1)) & 0xFFFF,
                    chunk_width=min(2048, N))
                cov_ok = bool(cov_rows) and bool(cov_cols[0])
                if not ok or not cov_ok:
                    bad += 1
                print(f"{B:>3} {M:>5} {N:>5} {K:>5} {dtype:>5} "
                      f"{str(tx1)[0]:>4} {base_m:>5} {base_n:>5} "
                      f"{p['nmwin']:>4} {p['nchunk']:>5} {p['groups']:>4} "
                      f"{p['tasks']:>6} {p['blocks']:>3} "
                      f"{'OK' if cov_ok else 'BAD':>4} {maxerr:>10.3e} "
                      f"{margin:>10.0f}x  {'pass' if ok else 'FAIL'}")
    print(f"\n失败数 = {bad} / {len(cases) * 4}")

    # ---- phase 2 的 worker 覆盖（两种 AIV 索引读法都要完整） ----
    print("\n=== phase 2 worker 覆盖（stride=blocks，AIV_ONLY 与 MIX 1:2 两种读法）===")
    cov_bad = 0
    for (B, M, N, K) in cases[:8] + [(64, 256, 256, 256)]:
        base_m = min(128, ((M + 15) // 16) * 16)
        base_n = min(128, ((N + 15) // 16) * 16)
        p = plan((B, M, N, K), base_m, base_n, a.workers)
        r = check_phase2_coverage(p)
        ok = all(r.values())
        cov_bad += 0 if ok else 1
        if not ok:
            print(f"  B={B} M={M} N={N} blocks={p['blocks']} -> {r}")
    print(f"  覆盖失败数 = {cov_bad}（0 = 两种读法下每个 batch 都被处理）")

    # ---- 单 launch 兜底模式（原子加）的离散度 ----
    print("\n=== BMM_SINGLE_LAUNCH=1（groups=1 + 浮点原子加）的累加顺序离散度 ===")
    print(f"{'M':>6} {'N':>6} {'K':>6} {'16 种顺序的极差':>16} "
          f"{'与 golden 最大偏差':>18} {'容差下限':>10} 判定")
    for (B, M, N, K) in [(1, 512, 4096, 256), (1, 1024, 1024, 512),
                         (1, 8192, 8192, 64), (2, 4096, 512, 256)]:
        base_m = min(128, ((M + 15) // 16) * 16)
        base_n = min(128, ((N + 15) // 16) * 16)
        spread, worst, tol, mean_err = atomic_probe(
            B, M, N, K, base_m, base_n, "fp16", False, False, seed=M * K + N)
        print(f"{M:>6} {N:>6} {K:>6} {spread:>16.3e} {worst:>18.3e} "
              f"{tol:>10.3e} {'pass' if worst <= tol else 'FAIL'}")


if __name__ == "__main__":
    main()
