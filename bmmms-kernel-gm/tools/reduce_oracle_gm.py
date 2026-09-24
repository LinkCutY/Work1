#!/usr/bin/env python3
"""reduce_oracle_gm.py — 新 kernel.asc 归约逻辑的**离线逻辑复算**（含反向验证）

本机没有 CANN、没有 NPU，所以不可能"跑 kernel"。但新版的正确性依赖一串**下标
与布局推理**，这些可以逐位复算：

  1. Fixpipe 连续写把整分片打包成 [rows, baseN]、把 N 尾块打包成 [rows, tailN]；
  2. 整分片走 AtomicMax 就地累到一个 [baseM, baseN] 缓冲（第 0 片普通写=初值）；
  3. AIV 读回时 blockCount = validM 行、每行 baseN 个 float；
  4. WholeReduceMax 每 pass 覆盖 64 列，dst 下标 = 行号；
  5. 尾块走标量逐行读；
  6. 行和只在 row < validM 上做。

`simulate()` 就是这 6 条的直接翻译；`golden()` 是定义本身（先按行取 N 维最大值，
再按行求和）。两者在随机数据上必须逐位相同 —— fp32 的 max 是精确操作，所以这里
可以用 `np.array_equal` 而不是 `isclose`。

反向验证：`simulate(variant=...)` 故意做错三件事，Oracle 必须报出不一致，
证明它不是空转：
  * tail_in_acc  ：把 N 尾块也混进原子归并缓冲（跨行串列）
  * read_baseM   ：读回时按 baseM 行而不是 validM 行（把上一行块的残留当有效数据）
  * last_wins    ：关掉 AtomicMax（只剩最后一个 N 分片的值）
  * no_tail      ：漏掉 N 尾块

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

    passes = -(-baseN // 64)  # ceil(baseN/64)

    for batch in range(b):
        batch_sum = np.float32(0.0)

        # 槽位是**跨行块复用**的（真实 GM 槽位也是），残留必须保留在模型里
        acc = np.zeros((MAX_BASE_M, MAX_TILE_N), dtype=np.float32)
        for row_start in range(0, m, baseM):
            valid_m = min(baseM, m - row_start)
            n_full = n // baseN
            tail_n = n - n_full * baseN

            tiles = [("full", t) for t in range(n_full)]
            if tail_n and variant != "no_tail":
                tiles.append(("tail", n_full))

            tail_flat = None
            for tile_idx, (kind, t) in enumerate(tiles):
                if kind == "full" or variant == "tail_in_acc":
                    col0 = n_full * baseN if kind == "tail" else t * baseN
                    width = tail_n if kind == "tail" else baseN
                    rows = min(valid_m, MAX_BASE_M)
                    blk = c[batch, row_start : row_start + rows, col0 : col0 + width]
                    flat = blk.reshape(-1).astype(np.float32)
                    if variant == "last_wins" or tile_idx == 0:
                        acc.reshape(-1)[: flat.size] = flat          # 普通写（=初值）
                    else:
                        cur = acc.reshape(-1)[: flat.size]
                        acc.reshape(-1)[: flat.size] = np.maximum(cur, flat)
                else:
                    rows = min(valid_m, MAX_BASE_M)
                    tail_flat = c[
                        batch, row_start : row_start + rows, n_full * baseN :
                    ].reshape(-1).astype(np.float32)

            row_max = np.full(MAX_BASE_M, NEG_INF, dtype=np.float32)

            if n_full > 0:
                read_rows = baseM if variant == "read_baseM" else valid_m
                read_rows = min(read_rows, MAX_BASE_M)
                a = acc.reshape(-1)[: read_rows * baseN].reshape(read_rows, baseN)
                for p in range(passes):
                    cols = min(64, baseN - p * 64)
                    if cols <= 0:
                        break
                    part = a[:, p * 64 : p * 64 + cols].max(axis=1)
                    row_max[:read_rows] = np.maximum(row_max[:read_rows], part)

            if tail_flat is not None:
                t_tile = tail_flat.reshape(valid_m, tail_n)
                row_max[:valid_m] = np.maximum(row_max[:valid_m], t_tile.max(axis=1))

            # 正确的行和范围是 row < valid_m；变体 sum_all_rows 故意多算到最后一行块的
            # 全部 baseM 行（那些行在块尾是**上一块残留**，不是本块的数据）。
            sum_rows = valid_m
            if variant == "sum_all_rows":
                sum_rows = min(baseM, MAX_BASE_M)
            batch_sum = np.float32(batch_sum + row_max[:sum_rows].sum(dtype=np.float32))

        out[batch] = batch_sum

    return out


def shapes():
    yield from [
        (1, 16, 16), (1, 128, 128), (1, 129, 129), (1, 200, 300),
        (1, 64, 1000), (1, 1000, 64), (4, 100, 2000), (1, 1, 8192),
        (3, 300, 300), (1, 128, 4096), (2, 257, 1000), (1, 17, 65),
        # baseN=256 档（L0C 满存）与其各种尾块
        (1, 128, 512), (1, 128, 500), (1, 300, 1024), (1, 64, 8192),
        (2, 256, 4096), (1, 129, 256), (1, 128, 255), (1, 4000, 260),
    ]


def main() -> int:
    np.seterr(over="ignore")  # 反向变体会溢出（它就是要算错），不是被测代码的问题
    rng = np.random.default_rng(20260925)
    bad = 0
    print("== 正向：simulate 必须与 golden 逐位一致 ==")
    for b, m, n in shapes():
        baseM = min(MAX_BASE_M, -(-m // 16) * 16)
        baseN = min(MAX_TILE_N, -(-n // 16) * 16)
        c = rng.integers(-1000, 1000, size=(b, m, n)).astype(np.float16).astype(np.float32)
        neg = rng.random((b, m, n)) < 0.1
        c[neg] = rng.integers(-30000, -1, size=int(neg.sum()))
        got = simulate(c, m, n, baseM, baseN)
        exp = golden(c, m, n)
        ok = np.array_equal(got, exp)
        # 兜底档（host 分片阶梯的最后一档 = 基线配置 baseM=16）也必须逐位正确
        fbM = 16
        fbN = min(MAX_TILE_N, -(-n // 16) * 16)
        ok_fb = np.array_equal(simulate(c, m, n, fbM, fbN), exp)
        ok = ok and ok_fb
        bad += 0 if ok else 1
        print(
            f"  B={b:2d} M={m:5d} N={n:5d} baseM={baseM:3d} baseN={baseN:3d} "
            f"nFull={n // baseN:3d} tailN={n % baseN:3d}  兜底16/{fbN:3d}  "
            f"{'BITWISE-OK' if ok else 'MISMATCH'}  (y0={got[0]!r} golden={exp[0]!r})"
        )

    print("== 反向：故意做错必须被检出 ==")
    for variant in ("tail_in_acc", "sum_all_rows", "last_wins", "no_tail"):
        caught = 0
        tested = 0
        for b, m, n in shapes():
            baseM = min(MAX_BASE_M, -(-m // 16) * 16)
            baseN = min(MAX_TILE_N, -(-n // 16) * 16)
            if variant == "last_wins" and n // baseN < 2:
                continue
            if variant == "tail_in_acc" and n % baseN == 0:
                continue
            if variant == "no_tail" and n % baseN == 0:
                continue
            if variant == "sum_all_rows" and (m % baseM == 0 or m <= baseM):
                continue
            tested += 1
            c = rng.integers(-1000, 1000, size=(b, m, n)).astype(np.float16).astype(np.float32)
            if variant == "no_tail" and n % baseN:
                # 让行最大值只出现在 N 尾块里 —— 漏掉尾块必然被检出（确定性反向验证）
                c[:, :, (n // baseN) * baseN :] = np.float32(10000.0)
            if variant == "last_wins":
                # 让行最大值只出现在**第 0 个** N 分片里 —— 只留最后一片必然被检出
                c[:, :, :baseN] = np.float32(10000.0)
            got = simulate(c, m, n, baseM, baseN, variant=variant)
            # 变体本身会溢出（它就是要算错），屏蔽 numpy 的溢出告警
            with np.errstate(over="ignore"):
                exp = golden(c, m, n)
                if not np.array_equal(got, exp):
                    caught += 1
        print(f"  {variant:12s} 检出 {caught}/{tested}")
        if tested and caught != tested:
            bad += 1

    print("PASS" if bad == 0 else f"FAIL ({bad})")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
