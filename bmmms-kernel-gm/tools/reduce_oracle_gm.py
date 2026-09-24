#!/usr/bin/env python3
"""reduce_oracle_gm.py — kernel.asc 的搬运/归约逻辑**离线逐位复算**（含反向验证）

本机没有 CANN、没有 NPU，所以不可能"跑 kernel"。但正确性依赖一串下标与布局推理，
这些可以逐位复算。模型严格照抄 kernel 里的表达式：

  1. 一个行块（validM = min(baseM, m-rowStart) 行）由**一次 IterateAll** 算完；
  2. **框架**的连续写把第 t 个 N 分片铺在 `stage[t * validM * baseN]`（这是"有效尺寸
     步进"，依据是 123b2b8 实测的"尾片按 validN 紧凑打包"），尾片紧跟其后；
  3. AIV **按自己假定的步进**去读（`read_stride`）：假定错了就读到错位的数据
     —— 反向验证里的 pad_stride 就是模拟"假定成补齐尺寸"；
  4. 一次读 READ_TILES 片：`rows = tiles*validM` 行、每行 baseN 个 float；
  5. 第一级 WholeReduceMax 逐个"列组"做：第 g 组取每行的 [colBase, colBase+columns) 列，
     行距 = baseN/8 个 datablock，结果**连续**落在 groupMax[g*MAX_ROWS + r]；
  6. 第二级用元素级 Max 把各列组折起来（组值之间隔 8 个 float，靠归约 mask 读不到）；
  7. 每个分片的行最大值再 Max 进行最大值向量（跨 N 取 max，与分片顺序无关）；
  8. N 尾块：行距 = tailN 的紧凑布局，逐行标量取最大；
  9. 行和只在 row < validM 上做。

fp32 的 max 是精确操作，所以用 `np.array_equal` 而不是 `isclose`。

反向验证（simulate(variant=...) 故意做错，Oracle 必须报出不一致）：
  * pad_stride  ：读的时候按**补齐**尺寸 baseM*baseN 找分片（真机是有效尺寸）
  * tail_pitch  ：尾片行距读成 baseN（真机是 tailN）
  * no_tail     ：漏掉 N 尾块
  * sum_all_rows：行和多算到行块末尾的全部 baseM 行
  * group_fold  ：把第二级的元素级 Max 换成"再调一次归约、mask=64"（读串行）

用法： python reduce_oracle_gm.py           # 全部通过则退出码 0
"""
from __future__ import annotations

import sys

import numpy as np

MAX_BASE_M = 128
MAX_TILE_N = 128
READ_TILES = 2
MAX_ROWS = READ_TILES * MAX_BASE_M
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

    full_groups = baseN // 64
    rem_cols = baseN - full_groups * 64
    group_count = full_groups + (1 if rem_cols else 0)
    n_full = n // baseN
    tail_n_true = n - n_full * baseN

    for batch in range(b):
        batch_sum = np.float32(0.0)

        for row_start in range(0, m, baseM):
            valid_m = min(baseM, m - row_start)

            # ---- staging：框架按有效尺寸 [validM, baseN] 逐片紧挨着铺 ----
            write_stride = valid_m * baseN
            tail_off = n_full * write_stride
            # 预留 slack：反向变体的错误步进会读到更远的地址，模型里给足空间
            stage = np.zeros(
                max(tail_off + valid_m * tail_n_true + baseN,
                    (n_full + 1) * baseM * baseN + baseN),
                dtype=np.float32)
            for t in range(n_full):
                blk = c[batch, row_start:row_start + valid_m,
                        t * baseN:(t + 1) * baseN].reshape(-1)
                stage[t * write_stride:t * write_stride + blk.size] = blk
            if tail_n_true:
                blk = c[batch, row_start:row_start + valid_m,
                        n_full * baseN:n].reshape(-1)
                stage[tail_off:tail_off + blk.size] = blk

            # ---- AIV 侧：按自己假定的步进去读 ----
            read_stride = baseM * baseN if variant == "pad_stride" else write_stride

            row_max = np.full(MAX_BASE_M, NEG_INF, dtype=np.float32)

            t = 0
            while t < n_full:
                tiles = min(READ_TILES, n_full - t)
                rows = tiles * valid_m

                buf = np.zeros((rows, baseN), dtype=np.float32)
                for j in range(tiles):
                    off = (t + j) * read_stride
                    piece = stage[off:off + valid_m * baseN]
                    buf[j * valid_m:(j + 1) * valid_m, :] = piece.reshape(valid_m, baseN)

                # 第一级：逐个列组，结果连续放在 groupMax[g*MAX_ROWS + r]
                group_max = np.full(group_count * MAX_ROWS, NEG_INF, dtype=np.float32)
                for g in range(group_count):
                    cols = 64 if g < full_groups else rem_cols
                    col0 = g * 64 if g < full_groups else full_groups * 64
                    group_max[g * MAX_ROWS:g * MAX_ROWS + rows] = (
                        buf[:, col0:col0 + cols].max(axis=1))

                # 第二级：元素级 Max 折组
                if variant == "group_fold":
                    # 错误做法：再调一次归约、mask=64 —— 连续读 64 个槽，读到别的行
                    for r in range(rows):
                        row_pick = group_max[r * group_count: r * group_count + 64].max()
                        group_max[r] = row_pick
                else:
                    for g in range(1, group_count):
                        group_max[:rows] = np.maximum(
                            group_max[:rows], group_max[g * MAX_ROWS:g * MAX_ROWS + rows])

                # 跨分片 Max
                for j in range(tiles):
                    chunk = group_max[j * valid_m:(j + 1) * valid_m]
                    row_max[:valid_m] = np.maximum(row_max[:valid_m], chunk)

                t += tiles

            # ---- N 尾块：紧凑布局，行距 = tailN ----
            tail_n = 0 if variant == "no_tail" else tail_n_true
            if tail_n:
                pitch = baseN if variant == "tail_pitch" else tail_n
                raw = stage[tail_off:tail_off + valid_m * tail_n_true]
                view = np.zeros(valid_m * pitch, dtype=np.float32)
                view[:raw.size] = raw
                row_max[:valid_m] = np.maximum(
                    row_max[:valid_m], view.reshape(valid_m, pitch)[:, :tail_n].max(axis=1))

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
    ]


def tile_shape(m: int, n: int) -> tuple[int, int]:
    baseM = min(MAX_BASE_M, -(-m // 16) * 16)
    baseN = min(MAX_TILE_N, -(-n // 16) * 16)
    return baseM, baseN


def main() -> int:
    np.seterr(over="ignore")  # 反向变体会溢出（它就是要算错），不是被测代码的问题
    rng = np.random.default_rng(20260925)
    bad = 0

    print("== 正向：simulate 必须与 golden 逐位一致（方阵档 + 基线兜底档）==")
    for b, m, n in shapes():
        baseM, baseN = tile_shape(m, n)
        c = rng.integers(-1000, 1000, size=(b, m, n)).astype(np.float16).astype(np.float32)
        neg = rng.random((b, m, n)) < 0.1
        c[neg] = rng.integers(-30000, -1, size=int(neg.sum()))
        exp = golden(c, m, n)
        got = simulate(c, m, n, baseM, baseN)
        ok = np.array_equal(got, exp)
        fbM, fbN = 16, min(MAX_TILE_N, -(-n // 16) * 16)
        ok_fb = np.array_equal(simulate(c, m, n, fbM, fbN), exp)
        ok = ok and ok_fb
        bad += 0 if ok else 1
        print(
            f"  B={b:2d} M={m:5d} N={n:5d} baseM={baseM:3d} baseN={baseN:3d} "
            f"nFull={n // baseN:3d} tailN={n % baseN:3d}  兜底16/{fbN:3d}  "
            f"{'BITWISE-OK' if ok else 'MISMATCH'}"
        )

    print("== 反向：故意做错必须被检出 ==")
    for variant in ("pad_stride", "tail_pitch", "no_tail", "sum_all_rows", "group_fold"):
        caught = 0
        tested = 0
        for b, m, n in shapes():
            baseM, baseN = tile_shape(m, n)
            if variant == "pad_stride" and (m % baseM == 0 or n // baseN < 2):
                continue
            if variant in ("tail_pitch", "no_tail") and n % baseN == 0:
                continue
            if variant == "sum_all_rows" and (m % baseM == 0 or m <= baseM):
                continue
            if variant == "group_fold" and (baseN // 64) < 1:
                continue
            if variant == "group_fold" and n // baseN < 1:
                continue
            tested += 1
            c = rng.integers(-1000, 1000, size=(b, m, n)).astype(np.float16).astype(np.float32)
            # 反向验证的数据要"对着变体的错误方向"造，否则会被别的项盖住：
            #   pad_stride / tail_pitch / group_fold 的错都会**把 0 或邻行/邻片的值捞进来**
            #   ⇒ 让每个元素都是"随行号、随列号递增的负数"：误读必然得到一个更大的数
            #     （槽位错位会把后面更大的值捞进来，捞到 slack 则是 0，都比真实值大）；
            #   no_tail 是**漏掉**尾片（变小）⇒ 让尾片独占最大值且为正。
            if variant in ("pad_stride", "tail_pitch", "group_fold"):
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
