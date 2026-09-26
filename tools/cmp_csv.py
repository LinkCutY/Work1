#!/usr/bin/env python3
"""cmp_csv.py - per-case baseline vs candidate comparison from two bench CSVs."""
import pathlib
import sys


def load(p):
    d = {}
    for ln in pathlib.Path(p).read_text(errors="replace").splitlines():
        if not ln.startswith("case"):
            continue
        f = ln.split(",")
        if len(f) < 13:
            continue
        try:
            d[f[0]] = (float(f[8]), f[10], float(f[11]), f[12],
                       [int(x) for x in f[1:5]])
        except ValueError:
            pass
    return d


a = load(sys.argv[1])
b = load(sys.argv[2])
print(f"{'case':6} {'B,M,N,K':>20} {'base_us':>9} {'cand_us':>9} {'delta':>8}  flags")
for k in sorted(a):
    if k not in b:
        continue
    ba, pa, ma, da, sh = a[k]
    bb, pb, mb, db, _ = b[k]
    flag = ""
    if pa != pb:
        flag += f" PASS:{pa}->{pb}"
    if da != db:
        flag += f" det:{da}->{db}"
    if ma != mb:
        flag += f" maxdiff:{ma:.4g}->{mb:.4g}"
    print(f"{k:6} {str(sh):>20} {ba*1000:9.1f} {bb*1000:9.1f}"
          f" {(bb/ba-1)*100:+7.1f}% {flag}")
