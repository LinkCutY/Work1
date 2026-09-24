#!/usr/bin/env python3
"""reduce_oracle_gm.py — kernel.asc 的搬运/归约逻辑**离线逐位复算**（含反向验证）

本机没有 CANN、没有 NPU（队友那台 910C 我们这边也连不上），所以正确性只能这样验：
把 kernel 里的下标/布局推理**逐字翻译成 numpy**，再与定义（FP64 golden：max over N、
sum over M、转 fp32）逐位比对。模型严格照抄 kernel 的表达式：

  1. 一个行块（validM = min(baseM, m-rowStart) 行）由**一次 IterateAll** 算完；
     validM < baseM 时（M 尾块）framework 会多吐 ghost tile，所以 kernel 退回
     逐 tile 的 Iterate/GetTensorC —— 两者在本模型里都归到"slot 网格"上：
  2. slot 网格（队友 910C 真机探明）：**slot 步进 = baseM*baseN**，
     slot 内**行的步进 = 该 tile 自己的 validN**，只写有效行；
     逐 tile 路径每次写回 slot 0。
  3. 读回：`DataCopyPad(cUb, cTile[slot*baseM*baseN], blockLen = validM*validN*4)`。
  4. 归约：整片（validN == baseN）逐行 `ReduceMax(..., count=validN)`，行距 = baseN；
     N 尾片行距 = validN，走标量逐行读。
  5. 片间只做 Max（跨 N 取最大，可交换可结合，与顺序无关）。
  6. 行和只在 row < validM 上做；每 batch 只写一个 float。

反向验证（simulate(variant=...) 故意做错，Oracle 必须报出不一致）：
  * valid_stride ：slot 偏移按 validM*baseN 算（= 我上一版的错；真机是 baseM*baseN）
  * tail_pitch   ：N 尾片读时把行距当成 baseN（真机是 validN）
  * no_tail      ：漏掉 N 尾片
  * sum_all_rows ：行和多算到行块末尾的全部 baseM 行

用法： python reduce_oracle_gm.py           # 全部通过则退出码 0
"""
from __future__ import annotations

import sys

import numpy as np

MAX_BASE_M = 128
MAX_TILE_N = 256
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
    variant: str | None = None,
) -> np.ndarray:
    """按 kernel.asc 的搬运/归约顺序复算 y。c 形状 [B, M, N]。"""
    b = c.shape[0]
    out = np.zeros(b, dtype=np.float32)

    n_tiles = -(-n // baseN)

    for batch in range(b):
        batch_sum = np.float32(0.0)

        for row_start in range(0, m, baseM):
            valid_m = min(baseM, m - row_start)
            strip_at_once = valid_m >= baseM

            # staging：slot 网格（slot 步进 = baseM*baseN，行距 = 该 tile 的 validN）
            stage = np.zeros((n_tiles + 1) * baseM * baseN, dtype=np.float32)


            def write_slot(t: int) -> None:
                """把第 t 个 tile 的有效数据写进它该在的 slot（= framework 的连续写）。"""
                valid_n = min(baseN, n - t * baseN)
                slot = t if strip_at_once else 0
                base = slot * baseM * baseN
                blk = c[batch, row_start:row_start + valid_m,
                        t * baseN:t * baseN + valid_n]
                stage[base:base + valid_m * valid_n] = blk.reshape(-1)


            if strip_at_once:
                # 一次性搬出：网格里每个 slot 都就位了才开始读
                for t in range(n_tiles):
                    write_slot(t)

            row_max = np.full(MAX_BASE_M, NEG_INF, dtype=np.float32)

            for t in range(n_tiles):
                valid_n = min(baseN, n - t * baseN)
                if variant == "no_tail" and valid_n < baseN:
                    continue
                if not strip_at_once:
                    # 逐 tile 路径：写一个 slot 立刻读一个（同一个 slot 被反复覆盖）
                    write_slot(t)
                slot = t if strip_at_once else 0
                if variant == "valid_stride":
                    # 我上一版的错：偏移按有效尺寸算
                    base = t * valid_m * baseN
                else:
                    base = slot * baseM * baseN

                if variant == "tail_pitch" and valid_n < baseN:
                    # 错：尾片还按 baseN 行距读（会读到下一个 slot 的数据）
                    raw = stage[base:base + valid_m * baseN]
                    tile = raw.reshape(valid_m, baseN)[:, :valid_n]
                else:
                    # 对：读 valid_m*valid_n 个 float，行距 = valid_n
                    raw = stage[base:base + valid_m * valid_n]
                    tile = raw.reshape(valid_m, valid_n)

                row_max[:valid_m] = np.maximum(row_max[:valid_m], tile.max(axis=1))

            sum_rows = valid_m
            if variant == "sum_all_rows" and row_start + baseM >= m:
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
        (1, 64, 48), (1, 128, 40), (1, 300, 24), (1, 200, 96), (1, 200, 112),
        (1, 128, 256), (1, 256, 256), (1, 300, 256), (1, 200, 300), (1, 128, 512),
        # 队友 bench 里点名的真机大 shape（尾部/窄 M/窄 N 都在）
        (1, 2048, 8192), (1, 8192, 8192), (1, 8191, 8191), (1, 33, 8192),
        (1, 4096, 100), (1, 5000, 5000), (32, 256, 256), (1, 16, 16),
    ]


def tile_shape(m: int, n: int) -> tuple[int, int]:
    """host 侧 fixM/fixN 的口径：fixM = align16(min(256, m))；fixN = min(tileN, 32768/fixM)。"""
    fixM = max(16, min(MAX_BASE_M, -(-m // 16) * 16))
    tileN = min(MAX_TILE_N, -(-n // 16) * 16)
    fixN = min(tileN, 32768 // fixM)
    return fixM, max(8, fixN - (fixN % 8))


def main() -> int:
    np.seterr(over="ignore")  # 反向变体会溢出（它就是要算错），不是被测代码的问题
    rng = np.random.default_rng(20260925)
    bad = 0

    print("== 正向：simulate 必须与 golden 逐位一致 ==")
    for b, m, n in shapes():
        baseM, baseN = tile_shape(m, n)
        c = rng.integers(-1000, 1000, size=(b, m, n)).astype(np.float16).astype(np.float32)
        neg = rng.random((b, m, n)) < 0.1
        c[neg] = rng.integers(-30000, -1, size=int(neg.sum()))
        exp = golden(c, m, n)
        got = simulate(c, m, n, baseM, baseN)
        ok = np.array_equal(got, exp)
        bad += 0 if ok else 1
        print(
            f"  B={b:2d} M={m:5d} N={n:5d} baseM={baseM:3d} baseN={baseN:3d} "
            f"nTiles={-(-n // baseN):4d} tailN={n % baseN:3d}  "
            f"{'BITWISE-OK' if ok else 'MISMATCH'}"
        )

    print("== 反向：故意做错必须被检出 ==")
    for variant in ("valid_stride", "tail_pitch", "no_tail", "sum_all_rows"):
        caught = 0
        tested = 0
        for b, m, n in shapes():
            baseM, baseN = tile_shape(m, n)
            if variant == "valid_stride" and (m % baseM == 0 or n // baseN < 2):
                continue
            if variant in ("tail_pitch", "no_tail") and n % baseN == 0:
                continue
            if variant == "sum_all_rows" and (m % baseM == 0 or m <= baseM):
                continue
            tested += 1
            c = rng.integers(-1000, 1000, size=(b, m, n)).astype(np.float16).astype(np.float32)
            # 数据要"对着变体的错误方向"造：这三种错都会把 0（未写的 slot 区）或邻片数据
            # 捞进来 ⇒ 让真实行最大值是随行号递增的负数，比 0 小；no_tail 让尾片独占最大。
            if variant in ("valid_stride", "tail_pitch"):
                rows_i = np.arange(m, dtype=np.float32)[None, :, None]
                cols_i = np.arange(n, dtype=np.float32)[None, None, :]
                c[:] = np.float32(-10000.0) + rows_i + np.float32(0.001) * cols_i
            elif variant == "no_tail":
                c[:] = np.float32(-5000.0)
                c[:, :, (n // baseN) * baseN:] = np.float32(20000.0)
            got = simulate(c, m, n, baseM, baseN, variant=variant)
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
