#!/usr/bin/env python3
"""Probe: is the second 512-column half of each 1024-wide strip written/read?

B=64 M=4096 N=2048 K=32 (C=2.0 GiB -> stripeTarget=1024 -> chunkW=1024).
  case00: x2 nonzero only in columns [512,1024)   -> tests the 2nd half only
  case01: x2 nonzero only in columns [0,512)      -> tests the 1st half only
If a half is dropped the corresponding y collapses to ~0.
"""
import os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases_half")


def build(cid, lo, hi):
    B, M, N, K = 64, 4096, 2048, 32
    rng = np.random.default_rng(31337 + cid)
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
    print(f"case{cid:02d}: x2 nonzero cols [{lo},{hi})  gold y=[{gold.min():.2f},{gold.max():.2f}]")


build(0, 512, 1024)
build(1, 0, 512)
print("done")
