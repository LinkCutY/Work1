#!/usr/bin/env python3
"""gen_unal.py - synthesize an "unaligned small/mid" regime suite.

Target regime (the one currently routed to the bulk KFC MatmulMax path):
  B*M*N*K <= MED_MAC_LIMIT (8 388 608) AND at least one of M/N/K not a
  multiple of 16.  case02 (1x100x100x100) and case28 (1x100x100x32, the
  known judge shape c2) live here.

Each case is written as cases_unal/caseNN/{x1.bin,x2.bin,golden_y.bin,meta.txt}
with the same contract as the repo's gen_cases.py (FP64 golden, fp32 y).
Run on the evaluator host (needs numpy + ml_dtypes).
"""
import os

import numpy as np

try:
    from ml_dtypes import bfloat16
except ImportError:  # pragma: no cover
    bfloat16 = None

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cases_unal")

# (B, M, N, K, dtype, tx1, tx2, note)
CASES = [
    (1, 100, 100, 32, "float16", False, False, "judge c2 shape"),
    (1, 100, 100, 100, "float16", False, False, "= case02"),
    (1, 100, 100, 96, "bfloat16", False, False, "bf16 k%16=0"),
    (2, 129, 97, 40, "float16", False, True, "tx2 only, m/n/k tails"),
    (1, 200, 300, 64, "bfloat16", True, True, "double transpose"),
    (3, 50, 70, 40, "float16", True, False, "tx1 only"),
    (1, 37, 91, 48, "bfloat16", False, False, "odd m/n"),
    (4, 64, 100, 100, "float16", False, False, "b=4 same tails"),
    (1, 100, 500, 32, "float16", False, False, "wide n"),
    (1, 500, 100, 32, "bfloat16", False, False, "tall m"),
    (2, 63, 65, 128, "float16", True, True, "k=128 double transpose"),
    (1, 130, 130, 8, "float16", False, False, "k=8 min"),
    (1, 1000, 100, 8, "bfloat16", False, False, "k=8 long m"),
    (1, 100, 1024, 8, "float16", False, False, "k=8 wide n"),
    (1, 511, 511, 32, "float16", False, False, "macs 8.5M boundary"),
    (1, 90, 90, 104, "bfloat16", False, True, "k=104%16=8"),
    (8, 33, 33, 24, "float16", False, False, "small b, 24 k"),
    (1, 2047, 97, 40, "float16", False, False, "m tail 15"),
]


def impl(x1, x2, tx1, tx2):
    a = np.asarray(x1).astype(np.float64)
    b = np.asarray(x2).astype(np.float64)
    if tx1:
        a = np.swapaxes(a, -1, -2)
    if tx2:
        b = np.swapaxes(b, -1, -2)
    sim = np.matmul(a, b)
    return np.sum(np.max(sim, axis=-1), axis=-1).astype(np.float32)


def main():
    os.makedirs(OUT, exist_ok=True)
    for cid, (B, M, N, K, dt, tx1, tx2, note) in enumerate(CASES):
        if K % 8:
            print(f"  [warn] case{cid}: K={K} is not a multiple of 8")
        macs = B * M * N * K
        if macs > 8388608:
            raise SystemExit(f"case{cid}: macs {macs} outside the regime")
        rng = np.random.default_rng(20260926 + cid)
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
        print(f"case{cid:02d} B={B} M={M} N={N} K={K} {dt} tx={int(tx1)}{int(tx2)}"
              f" macs={macs/1e6:.2f}M y=[{y.min():.4f},{y.max():.4f}] {note}", flush=True)
    print("done")


if __name__ == "__main__":
    raise SystemExit(main())
