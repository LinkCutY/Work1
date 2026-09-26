#!/usr/bin/env python3
"""rebench.py - upload a kernel to the 910C devspace, build, run the bench, pull CSV.

Usage:
  python tools/rebench.py <kernel.asc> [--tag NAME] [--cases DIR] [--reps N] [--no-build]

Remote layout (created on first use):
  /mnt/workspace/opt/spec/kernel.asc        <- uploaded kernel
  /mnt/workspace/opt/spec/bench/bench_main_hash.asc
  /mnt/workspace/opt/spec/cases             <- 51-case suite (full)
  /mnt/workspace/opt/spec/cases_fast        <- symlink subset for quick loops
  /mnt/workspace/opt/spec/res/<tag>.csv
Local layout:
  bench/<tag>.csv                           <- pulled result
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

HOST = "aorx7"
RDIR = "/mnt/workspace/opt/spec"
LOCAL_BENCH = pathlib.Path(__file__).resolve().parent.parent / "bench"
LOCAL_BENCH.mkdir(exist_ok=True)

# representative subset: host-floor small, unaligned small, medium, deep-K big,
# K-minimal C-bound, bf16 variants, tail-heavy
FAST_CASES = ["00", "02", "03", "05", "07", "09", "11", "12", "15", "17",
              "20", "22", "25", "26", "28", "45", "51"]


def sh(args, check: bool = True) -> subprocess.CompletedProcess:
    p = subprocess.run(args, shell=isinstance(args, str), capture_output=True, text=True)
    if check and p.returncode != 0:
        sys.stderr.write(p.stdout + p.stderr)
        raise SystemExit(f"command failed rc={p.returncode}: {args}")
    return p


ENVPRE = (
    "source /home/developer/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 && "
    "export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:"
    "/usr/local/Ascend/driver/lib64/common:"
    "/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH"
)


def rsh(script: str) -> subprocess.CompletedProcess:
    """Run a bash script on the remote host via ssh (single argv element)."""
    return sh(["ssh", HOST, script], check=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--cases", default="cases_fast")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--no-build", action="store_true")
    a = ap.parse_args()

    kp = pathlib.Path(a.kernel).resolve()
    tag = a.tag or kp.stem
    if not kp.exists():
        raise SystemExit(f"no such kernel: {kp}")

    sh(["scp", "-q", str(kp), f"{HOST}:{RDIR}/kernel.asc"])
    if not a.no_build:
        build = f"""cd {RDIR} && {ENVPRE} && \
bisheng -I. -fPIC --npu-arch=dav-2201 -o bench/build/bench_main.o -c --asc-aicore-lang bench/bench_main_hash.asc 2>bench/build/compile.err && \
bisheng bench/build/bench_main.o -o bench/build/bench_main -ltiling_api -lregister -lplatform -lunified_dlog -ldl -lm -lgraph_base && \
grep -c 'error:' bench/build/compile.err; true"""
        p = rsh(build)
        out = p.stdout + p.stderr
        errors = [ln for ln in out.splitlines() if "error:" in ln]
        if p.returncode != 0 or errors:
            print(out[-6000:])
            raise SystemExit("BUILD FAILED")
        print(f"[build ok] tag={tag}")
    run = f"""cd {RDIR} && {ENVPRE} && timeout 900 ./bench/build/bench_main {a.cases} {a.reps} > res/{tag}.csv 2>res/{tag}.err; echo RC=$?"""
    p = rsh(run)
    if "RC=0" not in p.stdout:
        print(p.stdout + p.stderr)
        print(rsh(f"tail -20 {RDIR}/res/{tag}.err").stdout)
        raise SystemExit("RUN FAILED")
    sh(["scp", "-q", f"{HOST}:{RDIR}/res/{tag}.csv", str(LOCAL_BENCH / (tag + ".csv"))])
    print(f"[run ok] {LOCAL_BENCH / (tag + '.csv')}")
    print(rsh(f"cat {RDIR}/res/{tag}.csv").stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
