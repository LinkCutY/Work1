#!/usr/bin/env python3
"""Ramp probe: which column extent of a 512-wide chunk does the AIV actually see?

B=32 M=8192 N=2048 K=32 (C=2.0GiB -> chunkW=1024 -> two 512-wide slabs/chunk).
x1 = all ones; x2[b,k,n] = (n - base + 1)/8 inside the probed half, 0 outside.
Then C[m,n] = 32*x2[.,.,n] and the row max = 32*max(x2) over the columns the
kernel can see => y = M * 32 * visible_max.  y pins the visible column extent.
"""
import os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases_ramp")
B, M, N, K = 32, 8192, 2048, 32


def build(cid, base, cnt):
    x1 = np.ones((B, M, K), dtype=np.float16)
    x2 = np.zeros((B, K, N), dtype=np.float16)
    ramp = (np.arange(1, cnt + 1, dtype=np.float32) / 8.0).astype(np.float16)
    x2[:, :, base:base + cnt] = ramp[None, None, :]
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
    print(f"case{cid:02d}: ramp at cols [{base},{base+cnt}) visible_max={cnt} "
          f"expected y={M*32*ramp.max():.0f}")


build(0, 0, 512)      # first half of strip 0 (known broken)
build(1, 512, 512)    # second half of strip 0 (known good)
build(2, 1024, 512)   # first half of strip 1
print("done")
