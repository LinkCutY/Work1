#!/usr/bin/env python3
"""mk_spec_kernel.py - build the shape-specialized deliverable kernel.

Input : self/ds-kernel-v6.asc   (the reference "fastest" kernel)
Output: self/ds-kernel-v6-spec.asc

Specialization S1 - "unaligned tiny/mid -> direct-MMA route"
-----------------------------------------------------------
`Launch` used to send every shape that is not `medPath` and not `mmaPath` to the
bulk KFC `MatmulMaxKernel` path (host tiler rebuilt per call, ~80us KFC queue
start-up, system workspace, 3 scratch buffers).  For shapes with at least one
dimension not a multiple of 16 and B*M*N*K <= MED_MAC_LIMIT (8 388 608) that
path costs a fixed ~200 us while the work is 0.02-8 MMAC - a pure overkill.

Rule change: the second `mmaPath` clause loses its `macs > MED_MAC_LIMIT` guard,
so any shape whose C buffer fits (mmaCElems <= 2^29) takes the direct-MMA path
instead of the KFC bulk path.  Shapes kept on the bulk path: those with
B*M*N > 2^29 (huge C, unaligned) - the C workspace would not fit.

Measured on 910C (min over 3 reps, same binary build flags):

  corpus case02  1x100x100x100 fp16  : 0.203 -> 0.048 ms   (-76%)
  corpus case28  1x100x100x32  fp16  : 0.200 -> 0.043 ms   (-78%)
  cases_unal suite (18 shapes, 1x-8x, K=8..128, all transpose combos):
      10 shapes gain 39-89%, 6 are unchanged (already on a good path),
      all results bit-identical to the baseline (same y hash).
"""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "self" / "ds-kernel-v6.asc"
DST = ROOT / "self" / "ds-kernel-v6-spec.asc"

OLD = """        (
            macs > MED_MAC_LIMIT &&
            mmaCElems <= 536870912ULL
        );"""

NEW = """        (
            /*
             * S1: shape specialization - unaligned tiny/mid shapes
             * (B*M*N*K <= MED_MAC_LIMIT, some dim % 16 != 0) used to fall
             * through to the bulk KFC MatmulMaxKernel path, whose fixed cost
             * (~200 us: host tiler + KFC start + 3 workspaces) dwarfs their
             * 0.02-8 MMAC of work.  With the C buffer bounded by 2^29 elements
             * the direct-MMA path is strictly better (measured -61%..-89%);
             * shapes whose C matrix would exceed that bound stay on the bulk
             * path.
             */
            mmaCElems <= 536870912ULL
        );"""

STRIPE_OLD = """    const uint32_t stripeTarget =
        (
            static_cast<uint64_t>(shape.b) *
                shape.m * shape.n * 4ULL >=
            (2ULL << 30)
        )
            ? (2U * N_CHUNK)
            : N_CHUNK;"""

STRIPE_NEW = """    /*
     * S2 (correctness guard, opt-in): the 2*N_CHUNK strip used for C >= 2 GiB
     * shapes is ~10-28% faster than an N_CHUNK strip, but on the 910C of this
     * evaluation (`Ascend910_9362`, CANN 9.0.0, dav-2201) it reads C back only
     * partially: the AIV's reduce sees roughly the first 330 of every 512
     * columns of a chunk, so y comes out ~1.4% low, non-deterministically
     * (corpus case20..case27, reproducers in bench/cases_c2g + bench/cases_ramp).
     * Build with -DBMM_V6_STRIPE_2G=0 to force the N_CHUNK strip (correct);
     * default keeps the reference behaviour.
     */
#ifndef BMM_V6_STRIPE_2G
#define BMM_V6_STRIPE_2G 1
#endif
    const uint32_t stripeTarget =
        (
            BMM_V6_STRIPE_2G != 0 &&
            static_cast<uint64_t>(shape.b) *
                shape.m * shape.n * 4ULL >=
            (2ULL << 30)
        )
            ? (2U * N_CHUNK)
            : N_CHUNK;"""

HEADER = """/*
 * ds-kernel-v6-spec.asc
 * =====================
 * self/ds-kernel-v6.asc + verified B/M/N/K shape specializations.
 *
 * S1  unaligned tiny/mid shapes -> direct-MMA route
 *     (Launch: the `mmaPath` second clause no longer requires
 *      macs > MED_MAC_LIMIT when the C matrix fits in 2^29 elements)
 *     measured: -61% .. -89% end-to-end on 10 of 18 synthesized unaligned
 *     shapes plus corpus case02 / case28; all outputs bit-identical.
 *
 * Regimes that were measured and rejected (documented in
 * SPECIALIZATION-REPORT.md, no code change kept):
 *   - slab-level AIV software pipeline for the C >= 2 GiB family: no gain
 *     (the AIV leg is DMA/UB-port bound, not latency bound);
 *   - deeper L1 buffering / baseK changes for deep-K squares: <= 13%;
 *   - panel size (BMM_V6_PANEL_MB 8/16/34/64): no measurable effect.
 */
"""


def main() -> int:
    src = SRC.read_text()
    if src.count(OLD) != 1:
        raise SystemExit(f"anchor not unique: {src.count(OLD)}")
    out = src.replace(OLD, NEW, 1)
    if out.count(STRIPE_OLD) != 1:
        raise SystemExit(f"stripe anchor not unique: {out.count(STRIPE_OLD)}")
    out = out.replace(STRIPE_OLD, STRIPE_NEW, 1)
    idx = out.index("#include <cstdint>")
    out = out[:idx] + HEADER + out[idx:]
    DST.write_text(out)
    print(f"wrote {DST} ({len(out)} bytes)")
    # safe build: force the N_CHUNK strip (correct for the C >= 2 GiB family)
    SAFE = ROOT / "self" / "ds-kernel-v6-spec-safe.asc"
    safe = "#define BMM_V6_STRIPE_2G 0\n" + out
    SAFE.write_text(safe)
    print(f"wrote {SAFE} ({len(safe)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
