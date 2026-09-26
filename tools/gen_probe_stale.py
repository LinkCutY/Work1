#!/usr/bin/env python3
"""Stale-read probe for the chunkW=1024 (2-slab) path.

B=64 M=4096 N=2048 K=32 (C=2.0GiB -> chunkW=1024).
  case00: x2 nonzero only cols [0,512)     (strip 0, 1st half)
  case01: x2 ALL ZERO                      (gold y == 0 exactly)
  case02: x2 nonzero only cols [1024,1536) (strip 1, 1st half)
case01 must print y==0; any other value is stale/garbage data read by the AIV.
"""
import os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases_stale")


def build(cid, lo, hi):
    B, M, N, K = 64, 4096, 2048, 32
    rng = np.random.default_rng(4242 + cid)
    x1 = rng.uniform(-1, 1, (B, M, K)).astype(np.float32).astype(np.float16)
    x2 = np.zeros((B, K, N), dtype=np.float16)
    if hi > lo:
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
    print(f"case{cid:02d}: cols[{lo},{hi}) gold=[{gold.min():.3f},{gold.max():.3f}]")


build(0, 0, 512)
build(1, 0, 0)
build(2, 1024, 1536)
print("done")
