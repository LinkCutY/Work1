#!/usr/bin/env python3
"""reduce_oracle_gm.py — kernel.asc 归约/路由逻辑的**离线逻辑复算**（含反向验证）

本机没有 CANN、没有 NPU，所以不可能"跑 kernel"。但新版的正确性依赖一串**下标、
布局与路由推理**，这些可以逐位复算：

  1. Fixpipe 连续写把整分片打包成 [baseM, baseN]、把 N 尾块打包成 [validM, tailN]；
  2. 一个 M 窗口内的分片按**先 M 轴再 N 轴**（`Iterate.md:21` 的默认顺序）产出
     ⇒ mBlk = tileIdx % mBlocks、nBlk = tileIdx / mBlocks；
  3. 整分片按 mBlk 路由到该行块的归并区，nBlk == 0 的那一片是普通写（初值），
     其余 AtomicMax 就地取大（max 可交换可结合 ⇒ 与顺序无关）；
  4. AIV 在窗口末尾对每个行块读回归并区一次（blockCount = validM 行、每行 baseN 个 float）；
  5. WholeReduceMax 每 pass 覆盖 64 列，dst 下标 = 行号；
  6. 尾块走标量逐行读；
  7. 行和只在 row < validM 上做。

`simulate()` 就是这 7 条的直接翻译；`golden()` 是定义本身（先按行取 N 维最大值，
再按行求和）。fp32 的 max 是精确操作，所以用 `np.array_equal` 而不是 `isclose`。

反向验证：`simulate(variant=...)` 故意做错，Oracle 必须报出不一致，证明它不是空转：
  * tail_in_acc  ：把 N 尾块也混进原子归并缓冲（跨行串列）
  * sum_all_rows ：行和多算到窗口末尾行块的全部 baseM 行（把上一块的残留当有效数据）
  * last_wins    ：关掉 AtomicMax（只剩最后一个 N 分片的值）
  * no_tail      ：漏掉 N 尾块
  * order_n      ：分片实际按"先 N 再 M"产出，而 kernel 仍按 M 优先解码
                   —— 这是**窗口 > 1 时的路由假设**，必须能被检出

用法： python reduce_oracle_gm.py           # 全部通过则退出码 0
"""
from __future__ import annotations

import sys

import numpy as np

MAX_BASE_M = 128
MAX_TILE_N = 128
MAX_WINDOW_BLOCKS = 4
NEG_INF = np.float32(-3.402823466e38)


def golden(c: np.ndarray, m: int, n: int) -> np.ndarray:
    """定义：每个 batch 先按行取 N 维最大值，再把 m 行加起来。"""
    return c[:, :m, :n].max(axis=2).sum(axis=1, dtype=np.float32)


def simulate(
    c: np.ndarray,
    m: int,
    n: int,
    baseM: int,
    baseN: int,
    windowBlocks: int = 1,
    variant: str | None = None,
) -> np.ndarray:
    """按 kernel.asc 的搬运/路由/归约顺序复算 y。c 形状 [B, M, N]。"""
    b = c.shape[0]
    out = np.zeros(b, dtype=np.float32)

    passes = -(-baseN // 64)          # ceil(baseN/64)
    mWindow = windowBlocks * baseM

    for batch in range(b):
        batch_sum = np.float32(0.0)

        # 槽位**跨窗口复用**（真实 GM 槽位也是），残留必须保留在模型里
        acc = np.zeros((MAX_WINDOW_BLOCKS, MAX_BASE_M, MAX_TILE_N), dtype=np.float32)
        tail = np.zeros((MAX_WINDOW_BLOCKS, MAX_BASE_M, MAX_TILE_N), dtype=np.float32)

        for win_start in range(0, m, mWindow):
            window_rows = min(mWindow, m - win_start)
            wb = -(-window_rows // baseM)          # 本窗口的行块数 = ceil()
            n_full = n // baseN
            tail_n = n - n_full * baseN
            if variant == "no_tail":
                tail_n = 0

            # 分片产出顺序：默认"先 M 再 N"（M 优先）；order_n 变体模拟"先 N 再 M"
            tiles = []
            if variant == "order_n":
                for mb in range(wb):
                    for nb in range(n_full + (1 if tail_n else 0)):
                        tiles.append((mb, nb))
            else:
                for nb in range(n_full + (1 if tail_n else 0)):
                    for mb in range(wb):
                        tiles.append((mb, nb))

            tail_col = n_full
            for tile_idx, (mb_true, nb_true) in enumerate(tiles):
                # kernel 侧的解码（恒为 M 优先）
                mb = tile_idx % wb
                nb = tile_idx // wb
                rows = min(baseM, window_rows - mb_true * baseM)

                if nb_true < n_full or variant == "tail_in_acc":
                    col0 = 0 if nb_true >= n_full else nb_true * baseN
                    width = tail_n if nb_true >= n_full else baseN
                    blk = c[batch, win_start + mb_true * baseM:
                            win_start + mb_true * baseM + rows, col0:col0 + width]
                    flat = blk.reshape(-1).astype(np.float32)
                    dst = acc[mb].reshape(-1)[: flat.size]
                    if variant == "last_wins" or nb_true == 0:
                        dst[:] = flat                        # 普通写（=初值）
                    else:
                        dst[:] = np.maximum(dst, flat)       # AtomicMax 就地取大
                else:
                    blk = c[batch, win_start + mb_true * baseM:
                            win_start + mb_true * baseM + rows,
                            tail_col * baseN: tail_col * baseN + tail_n]
                    flat = blk.reshape(-1).astype(np.float32)
                    tail[mb].reshape(-1)[: flat.size] = flat  # 尾块：普通写

            # 窗口末尾：逐行块读回归并区并归约
            for w in range(wb):
                valid_m = min(baseM, window_rows - w * baseM)
                if valid_m <= 0:
                    continue
                row_max = np.full(MAX_BASE_M, NEG_INF, dtype=np.float32)

                if n_full > 0:
                    a_blk = acc[w].reshape(-1)[: valid_m * baseN].reshape(valid_m, baseN)
                    for p in range(passes):
                        cols = min(64, baseN - p * 64)
                        if cols <= 0:
                            break
                        part = a_blk[:, p * 64: p * 64 + cols].max(axis=1)
                        row_max[:valid_m] = np.maximum(row_max[:valid_m], part)

                if tail_n and variant != "no_tail":
                    t_blk = tail[w].reshape(-1)[: valid_m * tail_n].reshape(valid_m, tail_n)
                    row_max[:valid_m] = np.maximum(row_max[:valid_m], t_blk.max(axis=1))

                sum_rows = valid_m
                if variant == "sum_all_rows" and w == wb - 1:
                    sum_rows = min(baseM, MAX_BASE_M)
                with np.errstate(over="ignore"):
                    batch_sum = np.float32(
                        batch_sum + row_max[:sum_rows].sum(dtype=np.float32))

        out[batch] = batch_sum

    return out


def shapes():
    yield from [
        (1, 16, 16), (1, 128, 128), (1, 129, 129), (1, 200, 300),
        (1, 64, 1000), (1, 1000, 64), (4, 100, 2000), (1, 1, 8192),
        (3, 300, 300), (1, 128, 4096), (2, 257, 1000), (1, 17, 65),
        (1, 128, 512), (1, 128, 500), (1, 300, 1024), (1, 64, 8192),
        (2, 256, 4096), (1, 129, 256), (1, 128, 255), (1, 4000, 260),
    ]


def tile_shape(m: int, n: int) -> tuple[int, int]:
    baseM = min(MAX_BASE_M, -(-m // 16) * 16)
    baseN = min(MAX_TILE_N, -(-n // 16) * 16)
    return baseM, baseN


def main() -> int:
    np.seterr(over="ignore")  # 反向变体会溢出（它就是要算错），不是被测代码的问题
    rng = np.random.default_rng(20260925)
    bad = 0

    print("== 正向：simulate 必须与 golden 逐位一致（窗口 1 / 2 / 4 与兜底档）==")
    for b, m, n in shapes():
        baseM, baseN = tile_shape(m, n)
        c = rng.integers(-1000, 1000, size=(b, m, n)).astype(np.float16).astype(np.float32)
        neg = rng.random((b, m, n)) < 0.1
        c[neg] = rng.integers(-30000, -1, size=int(neg.sum()))
        exp = golden(c, m, n)
        ok = True
        detail = []
        for wb in (1, 2, 4):
            got = simulate(c, m, n, baseM, baseN, windowBlocks=wb)
            good = np.array_equal(got, exp)
            ok = ok and good
            detail.append(f"w{wb}{'✓' if good else '✗'}")
        # 兜底档（host 分片阶梯的最后一档 = 基线配置 baseM=16）
        fbM, fbN = 16, min(MAX_TILE_N, -(-n // 16) * 16)
        for wb in (1, 4):
            good = np.array_equal(simulate(c, m, n, fbM, fbN, windowBlocks=wb), exp)
            ok = ok and good
            detail.append(f"fb{wb}{'✓' if good else '✗'}")
        bad += 0 if ok else 1
        print(
            f"  B={b:2d} M={m:5d} N={n:5d} baseM={baseM:3d} baseN={baseN:3d} "
            f"nFull={n // baseN:3d} tailN={n % baseN:3d}  "
            f"{'BITWISE-OK' if ok else 'MISMATCH'}  {' '.join(detail)}"
        )

    print("== 反向：故意做错必须被检出（窗口 4）==")
    for variant in ("tail_in_acc", "sum_all_rows", "last_wins", "no_tail", "order_n"):
        caught = 0
        tested = 0
        for b, m, n in shapes():
            baseM, baseN = tile_shape(m, n)
            wb = 4
            if min(4 * baseM, m) < baseM * 2:
                continue                                   # 窗口放不下 2 个行块
            if variant == "last_wins" and n // baseN < 2:
                continue
            if variant in ("tail_in_acc", "no_tail") and n % baseN == 0:
                continue
            if variant == "sum_all_rows" and (m % baseM == 0 or m <= baseM):
                continue
            if variant == "order_n":
                # 两种迭代顺序在这些情形下**数学上完全一致**，本就无法（也不需要）检出
                n_tiles = n // baseN + (1 if n % baseN else 0)
                wb_here = -(-min(4 * baseM, m) // baseM)
                if n_tiles <= 1 or wb_here <= 1:
                    continue
            tested += 1
            c = rng.integers(-1000, 1000, size=(b, m, n)).astype(np.float16).astype(np.float32)
            if variant in ("no_tail", "order_n"):
                # 整块压平 + 每行最大值唯一（只出现在第 0 列）—— 确定性反向验证
                c[:] = np.float32(-5000.0)
                c[:, :, 0] = (
                    np.float32(1000.0) +
                    np.arange(m, dtype=np.float32)[None, :])
            if variant == "no_tail":
                c[:, :, (n // baseN) * baseN :] = np.float32(20000.0)
            if variant == "last_wins":
                c[:, :, :baseN] = np.float32(20000.0)
            got = simulate(c, m, n, baseM, baseN, windowBlocks=wb, variant=variant)
            if not np.array_equal(got, golden(c, m, n)):
                caught += 1
        flag = "OK" if caught == tested else "MISS"
        if caught != tested:
            bad += 1
        print(f"  {variant:12s} 检出 {caught}/{tested}   {flag}")

    print("PASS" if bad == 0 else f"FAIL ({bad})")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
