#!/usr/bin/env python3
"""gen_c2g.py - small reproducers for the C >= 2 GiB (stripeTarget=1024) family.

Every case has B*M*N*4 >= 2 GiB so the host picks stripeTarget = 2*N_CHUNK =
1024 (chunkW = 1024 -> the AIV drains TWO 512-wide slabs per chunk task),
while keeping B*M*K and B*N*K small so the inputs are ~20-60 MB and a run takes
milliseconds.  c00..c03 are the 2-slab (failing) class; c10..c12 are the same
shapes just under the 2 GiB line (stripeTarget = 512, one slab) as controls.
"""
import os

import numpy as np

try:
    from ml_dtypes import bfloat16
except ImportError:  # pragma: no cover
    bfloat16 = None

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cases_c2g")

# (B, M, N, K, dtype, tx1, tx2, note)
CASES = [
    (64, 4096, 2048, 32, "float16", False, False, "C=2.15G 2-slab"),
    (64, 2048, 4096, 64, "bfloat16", False, False, "C=2.15G 2-slab"),
    (16, 8192, 4096, 32, "float16", False, False, "C=2.15G 2-slab"),
    (32, 4096, 4096, 32, "float16", False, True, "C=2.15G 2-slab tx2"),
    (32, 4096, 2048, 32, "float16", False, False, "control C=1.07G 1-slab"),
    (16, 8192, 2048, 32, "float16", False, False, "control C=1.07G 1-slab"),
    (64, 2048, 2048, 32, "bfloat16", False, False, "control C=1.07G 1-slab"),
]


def impl(x1, x2, tx1, tx2):
    a = np.asarray(x1).astype(np.float64)
    b = np.asarray(x2).astype(np.float64)
    if tx1:
        a = np.swapaxes(a, -1, -2)
    if tx2:
        b = np.swapaxes(b, -1, -2)
    out = np.empty(x1.shape[0], dtype=np.float32)
    for i in range(x1.shape[0]):
        sim = a[i] @ b[i]
        out[i] = np.sum(np.max(sim, axis=-1), dtype=np.float64)
    return out.astype(np.float32)


def main():
    os.makedirs(OUT, exist_ok=True)
    for cid, (B, M, N, K, dt, tx1, tx2, note) in enumerate(CASES):
        rng = np.random.default_rng(777 + cid)
        x1s = (B, K, M) if tx1 else (B, M, K)
        x2s = (B, N, K) if tx2 else (B, K, N)
        v1 = rng.uniform(-1, 1, x1s).astype(np.float32)
        v2 = rng.uniform(-1, 1, x2s).astype(np.float32)
        x1 = v1.astype(np.float16) if dt == "float16" else v1.astype(bfloat16)
        x2 = v2.astype(np.float16) if dt == "float16" else v2.astype(bfloat16)
        y = impl(x1, x2, tx1, tx2)
        d = os.path.join(OUT, f"case{cid:02d}")
        os.makedirs(d, exist_ok=True)
        x1.tofile(os.path.join(d, "x1.bin"))
        x2.tofile(os.path.join(d, "x2.bin"))
        y.tofile(os.path.join(d, "golden_y.bin"))
        with open(os.path.join(d, "meta.txt"), "w") as f:
            f.write(f"{B} {M} {N} {K} {1 if dt == 'float16' else 2}"
                    f" {1 if tx1 else 0} {1 if tx2 else 0}\n")
        cgb = B * M * N * 4 / 2**30
        print(f"case{cid:02d} B={B} M={M} N={N} K={K} {dt} C={cgb:.2f}GiB"
              f" inp={(x1.nbytes + x2.nbytes)/2**20:.0f}MiB {note}", flush=True)
    print("done")


if __name__ == "__main__":
    raise SystemExit(main())
