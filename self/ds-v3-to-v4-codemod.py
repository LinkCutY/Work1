#!/usr/bin/env python3
"""v3 -> v4 codemod: bound the C workspace to one N strip and ping-pong the
AIC fill / AIV drain per strip (two all-core barriers per strip).

Base: ds-kernel-v3.asc (itself v1-derived + the on-device final merge).
Output: ds-kernel-v4.asc. Every edit is anchored on unique source text and
asserted before it is applied.
"""
import sys

SRC = 'ds-kernel-v3.asc'
DST = 'ds-kernel-v4.asc'

content = open(SRC, encoding='utf-8', newline='').read()
crlf = content.count('\r\n')
content = content.replace('\r\n', '\n')
lines = content.split('\n')
N = len(lines)


def L(i):
    return lines[i - 1].rstrip('\r')


def find(text, nth=1):
    hits = [i + 1 for i, l in enumerate(lines) if l.rstrip('\r') == text]
    if len(hits) < nth:
        sys.exit('anchor not found (%d hits): %r' % (len(hits), text))
    return hits[nth - 1]


def find_in(text, lo, hi, nth=1):
    hits = [i + 1 for i, l in enumerate(lines) if l.rstrip('\r') == text and lo <= i + 1 <= hi]
    if len(hits) < nth:
        sys.exit('anchor not found in [%d,%d]: %r (%d hits)' % (lo, hi, text, len(hits)))
    return hits[nth - 1]


def rep(old, new, count=1):
    global raw
    if raw.count(old) != count:
        sys.exit('replace count %d != %d for:\n%r' % (raw.count(old), count, old[:120]))
    raw = raw.replace(old, new)


raw = '\n'.join(lines)

# ---------------------------------------------------------------- 1. host sizing
rep('''    /* full fp32 C matrix in GM workspace
       (kSplits partial copies when split) */
    static DevScratch mmaCWs;
    static DevScratch mmaPWs;

    const size_t mmaCBytes =
        static_cast<size_t>(kSplits) *
        shape.b *
        shape.m * shape.n * sizeof(float);''',
'''    /*
     * v4: the C workspace holds TWO N strips (2 x N_CHUNK columns) instead of
     * the whole [B,M,N] matrix, so the AIC can fill the next strip while the
     * AIV drains the previous one. C residency drops from B*M*N floats to
     * 2*B*M*N_CHUNK floats. Striping requires each chunk to own exactly one
     * partial slot column (realChunks == nChunksTask); otherwise a single
     * full-width strip is used and the kernel degenerates to the previous bulk
     * behaviour.
     */
    const uint32_t mmaStrips =
        (realChunks == nChunksTask) ? realChunks : 1U;
    const uint32_t mmaStripW =
        (mmaStrips == 1U) ? shape.n : N_CHUNK;
    static DevScratch mmaCWs;
    static DevScratch mmaPWs;

    const size_t mmaCBytes =
        2U * static_cast<size_t>(kSplits) *
        shape.b *
        shape.m * mmaStripW * sizeof(float);''')

# ------------------------------------------- 2. kernel-scope strip constants
i_sig = find('__global__ __aicore__ void MmaMaxKernel(')
i_pipe = find_in('    TPipe pipe;', i_sig, i_sig + 20)
insert2 = '''
    /*
     * v4: N-axis striping with double buffering. Two strip buffers of
     * `stripW` columns each hold consecutive strips; the AIC leg fills strip
     * c+1 while the AIV leg drains strip c, kept in step by one all-core
     * barrier per strip. When a chunk does not own exactly one partial slot
     * column (realChunks != nChunksTask) a single full-width strip is used,
     * which degenerates to the bulk behaviour.
     */
    const uint32_t stripCount =
        (realChunks == nChunksTask) ? realChunks : 1U;
    const uint32_t stripW =
        (stripCount == 1U) ? shape.n : N_CHUNK;
'''.strip('\n').split('\n')
ins2 = {}
ins2[i_pipe + 1] = list(insert2)

# ------------------------------------------------- 3. AIC: drop totalTasks
rep('''        const uint64_t totalTasks =
            static_cast<uint64_t>(shape.b) * mTiles *
                nTiles * kSplits;
''', '')

# ---------------------------------------- 4. AIC: strip loop + barriers
rep('''        uint32_t tcount = 0;
        for (uint64_t task = worker;
             task < totalTasks;
             task += workers) {''',
'''        uint32_t tcount = 0;
        for (uint32_t strip = 0; strip < stripCount; ++strip) {
        const uint32_t stripStart = strip * stripW;
        const uint32_t stripN =
            MinU32(stripW, shape.n - stripStart);
        const uint32_t stripNTiles =
            MinU32(nTiles, (stripN + MMA_TN - 1) / MMA_TN);
        const uint32_t nTileBegin = stripStart / MMA_TN;
        const uint64_t stripTasks =
            static_cast<uint64_t>(shape.b) * mTiles *
                stripNTiles * kSplits;
        for (uint64_t task = worker;
             task < stripTasks;
             task += workers) {''')

# ------------------------------------------- 5. AIC: N-tile decode per strip
rep('''            const uint32_t nT =
                static_cast<uint32_t>(mnt % nTiles);
            const uint64_t mb = mnt / nTiles;''',
'''            const uint32_t nT =
                nTileBegin +
                static_cast<uint32_t>(mnt % stripNTiles);
            const uint64_t mb = mnt / stripNTiles;''')

# ------------------------------------------------- 6. AIC: C address + pitch
rep('''            fp.dstStride = shape.n;''', '''            fp.dstStride = stripW;''')
rep('''            const uint64_t cOff =
                static_cast<uint64_t>(ks) * shape.b *
                    shape.m * shape.n +
                static_cast<uint64_t>(batch) * shape.m * shape.n +
                static_cast<uint64_t>(mStart) * shape.n + nStart;''',
'''            const uint64_t cOff =
                static_cast<uint64_t>(strip & 1U) *
                    kSplits * shape.b * shape.m * stripW +
                static_cast<uint64_t>(ks) * shape.b *
                    shape.m * stripW +
                static_cast<uint64_t>(batch) * shape.m * stripW +
                static_cast<uint64_t>(mStart) * stripW +
                (nStart - stripStart);''')

# --------------------------------- 7. both legs: C workspace extent = stripW
rep('''        cGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ CDtype *>(cWorkspace),
            static_cast<uint64_t>(kSplits) * shape.b *
                shape.m * shape.n);''',
'''        cGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ CDtype *>(cWorkspace),
            static_cast<uint64_t>(kSplits) * shape.b *
                shape.m * stripW);''', count=2)

# ------------------------- 8. AIC: close the strip loop with the two barriers
rep('''            const uint32_t tail0 =
                mySteps < MMA_NBUF0 ? mySteps : MMA_NBUF0;
            for (uint32_t t = 0; t < tail0; ++t) {
                const uint32_t tbuf =
                    (mySteps - 1 - t) & (MMA_NBUF0 - 1);
                WaitFlag<HardEvent::M_MTE1>(evL0Free[tbuf]);
            }
        }
''',
'''            const uint32_t tail0 =
                mySteps < MMA_NBUF0 ? mySteps : MMA_NBUF0;
            for (uint32_t t = 0; t < tail0; ++t) {
                const uint32_t tbuf =
                    (mySteps - 1 - t) & (MMA_NBUF0 - 1);
                WaitFlag<HardEvent::M_MTE1>(evL0Free[tbuf]);
            }
        }

        /*
         * One all-core barrier per strip. This arrival is the AIC side's
         * "strip c is complete in GM" signal (the AIC leg is gated on
         * PIPE_FIX, so every fixpipe write has landed); the AIV side's arrival
         * at the same round is its "strip c-1 drained" signal. Double
         * buffering makes that legal: the AIC then fills strip c+1 into the
         * other half while the AIV drains strip c, so the two legs overlap
         * instead of ping-ponging.
         */
        SyncAll<false>();
        }
''')

# ------------------------------------- 9. AIV: task space shrinks to one chunk
rep('''        const uint64_t totalRed =
            static_cast<uint64_t>(shape.b) * rowTiles *
                mSub * nChunksTask;''',
'''        const uint64_t totalRed =
            static_cast<uint64_t>(shape.b) * rowTiles *
                mSub;''')

# ------------------------------- 10. AIV: strip loop, barrier 1, chunk=strip
rep('''        for (uint64_t task = worker;
             task < totalRed;
             task += workers) {
            const uint32_t chunk =
                static_cast<uint32_t>(task % nChunksTask);
            uint64_t mw = task / nChunksTask;''',
'''        for (uint32_t strip = 0; strip < stripCount; ++strip) {
        const uint32_t stripStart = strip * stripW;
        /*
         * Drain this leg's own pipes first: the FFTS arrival is gated on the
         * MTE3 pipe, so without this the AIC leg could start refilling this
         * half of the buffer while an MTE2 read of it is still in flight
         * (measured as non-deterministic results on small-K shapes). Then the
         * barrier pairs with the AIC's "strip c complete" arrival.
         */
        PipeBarrier<PIPE_ALL>();
        SyncAll<false>();
        for (uint64_t task = worker;
             task < totalRed;
             task += workers) {
            const uint32_t chunk = strip;
            uint64_t mw = task;''')

# ------------------------------------------------ 11. AIV: strip-local C read
rep('''                cp.srcStride =
                    static_cast<uint16_t>(
                        (shape.n - validN) / cElemsBlk);''',
'''                cp.srcStride =
                    static_cast<uint16_t>(
                        (stripW - validN) / cElemsBlk);''')
rep('''                const uint64_t tileOff =
                    static_cast<uint64_t>(batch) *
                        shape.m * shape.n +
                    static_cast<uint64_t>(rowStart) *
                        shape.n +
                    nStart;
                const uint64_t splitStride =
                    static_cast<uint64_t>(shape.b) *
                        shape.m * shape.n;''',
'''                const uint64_t tileOff =
                    static_cast<uint64_t>(batch) *
                        shape.m * stripW +
                    static_cast<uint64_t>(rowStart) *
                        stripW +
                    (nStart - stripStart);
                const uint64_t splitStride =
                    static_cast<uint64_t>(shape.b) *
                        shape.m * stripW;''')

# ------------------------------- 12. AIV: close the strip loop, barrier 2
rep('''            DataCopyPad(
                partialGm[pOff * TILE_M + msub * subM],
                bestVec,
                ocp);

            Fence<HardEvent::MTE3_S>(pipe);
        }
''',
'''            DataCopyPad(
                partialGm[pOff * TILE_M + msub * subM],
                bestVec,
                ocp);

            Fence<HardEvent::MTE3_S>(pipe);
        }
        }
''')

# ---------------------------- 12b. AIV: C reads from the strip's own half
rep("""                DataCopy(cUb, cGm[tileOff], cp);""",
"""                const uint64_t bufBase =
                    static_cast<uint64_t>(strip & 1U) *
                    kSplits * splitStride;

                DataCopy(cUb, cGm[bufBase + tileOff], cp);""")

rep("""                    DataCopy(
                        cTmpUb,
                        cGm[ks * splitStride + tileOff],
                        cp);""",
"""                    DataCopy(
                        cTmpUb,
                        cGm[bufBase + ks * splitStride + tileOff],
                        cp);""")

# ------------- 13. drop the old phase1->2 barrier (barrier-round alignment)
rep("""    /*
     * All-core FFTS barrier: releases the AIV
     * workers only after every AIC has drained
     * its FIX pipe, so all C tiles are in GM.
     */
    SyncAll<false>();

""",
"""    /*
     * v4: the phase1->2 barrier that used to sit here is gone. With striping
     * each leg runs its own strip loop with two barriers per strip, and the
     * AIC leg reaches those barriers *before* the AIV leg does (the AIV leg
     * still has to enter its block). Keeping this barrier would shift one
     * leg's arrival sequence by one round: the AIC would read the AIV's
     * arrival here as "strip drained" and overwrite a strip that is still
     * being read (measured: non-deterministic results on small-K shapes).
     * Barrier rounds must line up per leg.
     */

""")

# ------------------------------------------------------------------ rebuild
out = []
for i, line in enumerate(raw.split('\n'), start=1):
    if i in ins2:
        out.extend(ins2[i])
    out.append(line)
open(DST, 'w', encoding='utf-8', newline='').write(
    '\n'.join(out).replace('\n', '\r\n'))
print('wrote %s: %d lines (was %d), crlf lines normalised: %d' % (DST, len(out), N, crlf))
