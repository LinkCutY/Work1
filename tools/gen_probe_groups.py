#!/usr/bin/env python3
"""Narrow-group probe for the chunkW=1024 (2-slab) path.

B=64 M=4096 N=2048 K=32 (C=2.0GiB -> chunkW=1024).  Each case puts random data
only in one 8/64-column group and zeros elsewhere; y==gold means the group is
read and reduced correctly, y==0 means it is dropped, anything in between means
a partial/aliased read.
"""
import os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases_grp")
GROUPS = [(0, 8), (504, 512), (512, 520), (1016, 1024), (1024, 1032),
          (0, 64), (256, 320), (448, 512)]


def build(cid, lo, hi):
    B, M, N, K = 64, 4096, 2048, 32
    rng = np.random.default_rng(9000 + cid)
    x1 = rng.uniform(-1, 1, (B, M, K)).astype(np.float32).astype(np.float16)
    x2 = np.zeros((B, K, N), dtype=np.float16)
    x2[:, :, lo:hi] = rng.uniform(-1, 1, (B, K, hi - lo)).astype(np.float32).astype(np.float16)
    gold = np.empty(B, dtype=np.float32)
    for b in range(B):
        sim = x1[b].astype(np.float64) @ x2[b].astype(np.float64)
        gold[b] = np.sum(np.max(sim, axis=-1), dtype=np.float64)
    d = os.path.join(OUT, f"case{cid:02d}")
    os.makedirs(d, exist_ok=True)
    x1.tofile(os.path.join(d, "x1.bin"))
    x2.tofile(os.path.join(d, "x2.bin"))
    gold.astype(np.float32).tofile(os.path.join(d, "golden_y.bin"))
    open(os.path.join(d, "meta.txt"), "w").write(f"{B} {M} {N} {K} 1 0 0\n")
    print(f"case{cid:02d}: cols[{lo},{hi}) gold0={gold[0]:.2f}")


for i, (lo, hi) in enumerate(GROUPS):
    build(i, lo, hi)
print("done")
