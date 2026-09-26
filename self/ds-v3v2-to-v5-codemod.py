#!/usr/bin/env python3
"""v5 = v2 base (fastest measured) + v3's on-device final merge + v4's striped
double-buffered AIC/AIV overlap, gated by a K threshold.

Source: ds-kernel-v3-from-v2.asc   (v2 + device-side merge, validated)
Output: ds-kernel-v5.asc
"""
import sys

SRC = 'ds-kernel-v3-from-v2.asc'
DST = 'ds-kernel-v5.asc'

raw = open(SRC, encoding='utf-8', newline='').read().replace('\r\n', '\n')
lines = raw.split('\n')
N = len(lines)


def find(text, nth=1):
    hits = [i + 1 for i, l in enumerate(lines) if l == text]
    if len(hits) < nth:
        sys.exit('anchor not found (%d hits): %r' % (len(hits), text))
    return hits[nth - 1]


def rep(old, new, count=1):
    global raw
    if raw.count(old) != count:
        sys.exit('replace count %d != %d for:\n%r' % (raw.count(old), count, old[:140]))
    raw = raw.replace(old, new)


J = '\n'.join


# ======================================================= 0. stripe policy knob
rep('static constexpr uint64_t MED_MAC_LIMIT = 8388608;',
    J(['static constexpr uint64_t MED_MAC_LIMIT = 8388608;',
       '/*',
       ' * v5 stripe policy. Striping pays when the AIV leg C traffic is a large',
       ' * part of the kernel (small K) and costs when the AIC task pipeline has',
       ' * to drain at every strip boundary (large K). Measured on 910C: K=512 ->',
       ' * -30%, K=32..128 -> -2..-7%, K=4096/8192 -> +11% (loss). Stripe only up',
       ' * to this K; above it the kernel takes the bulk path.',
       ' */',
       '#define BMM_V5_STRIPE_MAX_K 1024']))

# ================================================== 1. kernel signature: chunkW
rep(J(['    GM_ADDR cWorkspace,',
       '    Shape shape,',
       '    uint32_t nChunksTask,',
       '    uint32_t realChunks,',
       '    uint32_t aivWorkers,',
       '    uint32_t mSub,',
       '    uint32_t kSplits)']),
    J(['    GM_ADDR cWorkspace,',
       '    Shape shape,',
       '    uint32_t nChunksTask,',
       '    uint32_t realChunks,',
       '    uint32_t chunkW,',
       '    uint32_t aivWorkers,',
       '    uint32_t mSub,',
       '    uint32_t kSplits)']))

# ========================================== 2. kernel scope: strip constants
rep(J(['    uint32_t aivWorkers,',
       '    uint32_t mSub,',
       '    uint32_t kSplits)',
       '{',
       '    TPipe pipe;']),
    J(['    uint32_t aivWorkers,',
       '    uint32_t mSub,',
       '    uint32_t kSplits)',
       '{',
       '    TPipe pipe;',
       '    /*',
       '     * v5 striping. The C workspace holds two N strips of `stripW` columns;',
       '     * the AIC leg fills strip c+1 while the AIV leg drains strip c, kept in',
       '     * step by one all-core barrier per strip. stripW is the AIV chunk width',
       '     * (a multiple of 2*MMA_TN, so a task n-pair tiles never straddle a',
       '     * strip boundary). Striping needs realChunks == nChunksTask (one slot',
       '     * column per chunk) and shape.k <= BMM_V5_STRIPE_MAX_K; otherwise one',
       '     * full-width strip is used and the kernel keeps the v3 behaviour.',
       '     */',
       '    const uint32_t stripCount =',
       '        ((realChunks == nChunksTask) &&',
       '         (shape.k <= BMM_V5_STRIPE_MAX_K))',
       '            ? realChunks',
       '            : 1U;',
       '    const uint32_t stripW =',
       '        (stripCount == 1U) ? ((shape.n + 7U) & ~7U) : chunkW;']))

# ============================================= 3. AIC: drop the global count
rep(J(['        const uint64_t totalTasks =',
       '            static_cast<uint64_t>(shape.b) * mTiles *',
       '                nPairs * kSplits;',
       '']), '')

# =================================== 4. AIC: strip loop + per-strip task space
rep(J(['        uint32_t subCnt = 0;',
       '        uint32_t tcount = 0;',
       '        for (uint64_t task = worker;',
       '             task < totalTasks;',
       '             task += workers) {']),
    J(['        uint32_t subCnt = 0;',
       '        uint32_t tcount = 0;',
       '        for (uint32_t strip = 0; strip < stripCount; ++strip) {',
       '        const uint32_t stripStart = strip * stripW;',
       '        const uint32_t stripN =',
       '            MinU32(stripW, shape.n - stripStart);',
       '        const uint32_t stripNTiles =',
       '            MinU32(nTiles, (stripN + MMA_TN - 1) / MMA_TN);',
       '        const uint32_t stripPairs = (stripNTiles + 1U) / 2U;',
       '        const uint32_t nPairBegin = stripStart / (2U * MMA_TN);',
       '        const uint64_t stripTasks =',
       '            static_cast<uint64_t>(shape.b) * mTiles *',
       '                stripPairs * kSplits;',
       '        for (uint64_t task = worker;',
       '             task < stripTasks;',
       '             task += workers) {']))

# ==================================================== 5. AIC: n-pair decode
rep(J(['            const uint32_t nP =',
       '                static_cast<uint32_t>(nb % nPairs);',
       '            const uint32_t batch =',
       '                static_cast<uint32_t>(nb / nPairs);']),
    J(['            const uint32_t nP =',
       '                nPairBegin +',
       '                static_cast<uint32_t>(nb % stripPairs);',
       '            const uint32_t batch =',
       '                static_cast<uint32_t>(nb / stripPairs);']))

# ============================ 6. both legs: C row pitch becomes the strip pitch
rep(J(['        const uint32_t cPitch =',
       '            (shape.n + 7U) & ~7U;']),
    '        const uint32_t cPitch = stripW;  /* v5: strip pitch */', count=2)

# ======================== 7. both legs: C workspace extent = two strip halves
rep(J(['            static_cast<uint64_t>(kSplits) * shape.b *',
       '                shape.m * cPitch);']),
    J(['            2U * static_cast<uint64_t>(kSplits) * shape.b *',
       '                shape.m * cPitch);']), count=2)

# ============================= 8. AIC: strip-local write, own buffer half
rep(J(['            const uint64_t cOff =',
       '                static_cast<uint64_t>(ks) * shape.b *',
       '                    shape.m * cPitch +',
       '                static_cast<uint64_t>(batch) * shape.m *',
       '                    cPitch +',
       '                static_cast<uint64_t>(mStart) * cPitch +',
       '                nStart0;']),
    J(['            const uint64_t cOff =',
       '                static_cast<uint64_t>(strip & 1U) *',
       '                    kSplits * shape.b * shape.m * stripW +',
       '                static_cast<uint64_t>(ks) * shape.b *',
       '                    shape.m * stripW +',
       '                static_cast<uint64_t>(batch) * shape.m *',
       '                    stripW +',
       '                static_cast<uint64_t>(mStart) * stripW +',
       '                (nStart0 - stripStart);']))
rep('                fp.dstStride = cPitch;', '                fp.dstStride = stripW;')

# ===================================== 9. AIC: one barrier per strip
rep(J(['            const uint32_t tail1 =',
       '                mySteps < MMA_NBUF1 ? mySteps : MMA_NBUF1;',
       '            for (uint32_t t = 0; t < tail1; ++t) {',
       '                const uint32_t tbuf =',
       '                    (mySteps - 1 - t) & (MMA_NBUF1 - 1);',
       '                WaitFlag<HardEvent::MTE1_MTE2>(evL1Free[tbuf]);',
       '            }',
       '        }',
       '']),
    J(['            const uint32_t tail1 =',
       '                mySteps < MMA_NBUF1 ? mySteps : MMA_NBUF1;',
       '            for (uint32_t t = 0; t < tail1; ++t) {',
       '                const uint32_t tbuf =',
       '                    (mySteps - 1 - t) & (MMA_NBUF1 - 1);',
       '                WaitFlag<HardEvent::MTE1_MTE2>(evL1Free[tbuf]);',
       '            }',
       '        }',
       '',
       '        /*',
       '         * One barrier per strip. This arrival is the AIC leg "strip c is',
       '         * complete in GM" signal (the AIC leg is gated on PIPE_FIX, so every',
       '         * fixpipe write has landed); the AIV leg arrival at the same round is',
       '         * its "strip c-1 drained" signal. Double buffering makes that legal:',
       '         * the AIC then fills strip c+1 into the other half while the AIV',
       '         * drains strip c, so the two legs overlap instead of ping-ponging.',
       '         */',
       '        SyncAll<false>();',
       '        }',
       '']))

# ============================== 10. drop the old phase1->2 barrier
rep(J(['    /*',
       '     * All-core FFTS barrier: releases the AIV',
       '     * workers only after every AIC has drained',
       '     * its FIX pipe, so all C tiles are in GM.',
       '     */',
       '    SyncAll<false>();',
       '',
       '']),
    J(['    /*',
       '     * v5: the phase1->2 barrier that used to sit here is gone. Each leg now',
       '     * runs its own strip loop, and the AIC leg reaches its strip barriers',
       '     * before the AIV leg reaches them (the AIV leg still has to enter its',
       '     * block). Keeping this barrier would shift one leg arrival sequence by',
       '     * one round: the AIC would read the AIV arrival here as "strip drained"',
       '     * and overwrite a strip that is still being read (measured as',
       '     * non-deterministic results on small-K shapes). Rounds must line up.',
       '     */',
       '',
       '']))

# ======================= 11. AIV: chunkW now arrives from the host
rep(J(['        /*',
       '         * Balanced chunk width: instead of always',
       '         * striding by N_CHUNK and leaving a small',
       '         * tail chunk (e.g. N=768 -> 512+256), split',
       '         * N evenly across realChunks at 64-element',
       '         * granularity so every AIV task carries the',
       '         * same reduce width. chunkW*realChunks >= N',
       '         * keeps the partition exact.',
       '         */',
       '        const uint32_t chunkW =',
       '            MinU32(',
       '                ((shape.n + realChunks - 1) /',
       '                     realChunks +',
       '                 63U) & ~63U,',
       '                N_CHUNK);',
       '']),
    J(['        /* v5: chunkW is computed by the host (Launch) and passed in, so the',
       '           strip width and the C workspace size cannot drift apart. */',
       '']))

# ===================== 12. AIV: task space (chunk dimension only when bulk)
rep(J(['        const uint64_t totalRed =',
       '            static_cast<uint64_t>(shape.b) * rowTiles *',
       '                mSub * nChunksTask;']),
    J(['        /*',
       '         * Striped: one chunk per strip, so the task space is the row windows',
       '         * only. Bulk (stripCount == 1): keep the v3 task space, which carries',
       '         * the chunk index and folds every chunk congruent to it into the same',
       '         * partial slot.',
       '         */',
       '        const uint64_t totalRed =',
       '            (stripCount > 1U)',
       '                ? static_cast<uint64_t>(shape.b) * rowTiles * mSub',
       '                : static_cast<uint64_t>(shape.b) * rowTiles *',
       '                      mSub * nChunksTask;']))

# ==================== 13. AIV: strip loop, pipe drain, barrier, chunk decode
rep(J(['        for (uint64_t task = worker;',
       '             task < totalRed;',
       '             task += workers) {',
       '            const uint32_t chunk =',
       '                static_cast<uint32_t>(task % nChunksTask);',
       '            uint64_t mw = task / nChunksTask;']),
    J(['        for (uint32_t strip = 0; strip < stripCount; ++strip) {',
       '        const uint32_t stripStart = strip * stripW;',
       '        /*',
       '         * Drain this leg own pipes first: the FFTS arrival is gated on the',
       '         * MTE3 pipe, so without it the AIC leg could start refilling this',
       '         * half of the buffer while an MTE2 read of it is still in flight',
       '         * (measured as non-deterministic results on small-K shapes). Then the',
       '         * barrier pairs with the AIC leg "strip c complete" arrival.',
       '         */',
       '        PipeBarrier<PIPE_ALL>();',
       '        SyncAll<false>();',
       '        for (uint64_t task = worker;',
       '             task < totalRed;',
       '             task += workers) {',
       '            const uint32_t chunk =',
       '                (stripCount > 1U)',
       '                    ? strip',
       '                    : static_cast<uint32_t>(task % nChunksTask);',
       '            uint64_t mw =',
       '                (stripCount > 1U) ? task : task / nChunksTask;']))

# ===================== 14. AIV: strip-local address + own buffer half
rep(J(['                const uint64_t tileOff =',
       '                    static_cast<uint64_t>(batch) *',
       '                        shape.m * cPitch +',
       '                    static_cast<uint64_t>(rowStart) *',
       '                        cPitch +',
       '                    nStart;']),
    J(['                const uint64_t tileOff =',
       '                    static_cast<uint64_t>(batch) *',
       '                        shape.m * stripW +',
       '                    static_cast<uint64_t>(rowStart) *',
       '                        stripW +',
       '                    (nStart - stripStart);']))
rep(J(['                const uint64_t splitStride =',
       '                    static_cast<uint64_t>(shape.b) *',
       '                        shape.m * cPitch;']),
    J(['                const uint64_t splitStride =',
       '                    static_cast<uint64_t>(shape.b) *',
       '                        shape.m * stripW;',
       '                const uint64_t bufBase =',
       '                    static_cast<uint64_t>(strip & 1U) *',
       '                    kSplits * splitStride;']))
rep(J(['                if (nByteAligned) {',
       '                    DataCopy(cUb, cGm[tileOff], cp);']),
    J(['                if (nByteAligned) {',
       '                    DataCopy(cUb, cGm[bufBase + tileOff], cp);']))
rep(J(['                        DataCopyPad(',
       '                            cUb[r * rowPitch],',
       '                            cGm[tileOff +']),
    J(['                        DataCopyPad(',
       '                            cUb[r * rowPitch],',
       '                            cGm[bufBase + tileOff +']))
rep(J(['                        DataCopy(',
       '                            cTmpUb,',
       '                            cGm[ks * splitStride + tileOff],',
       '                            cp);']),
    J(['                        DataCopy(',
       '                            cTmpUb,',
       '                            cGm[bufBase + ks * splitStride + tileOff],',
       '                            cp);']))
rep(J(['                            DataCopyPad(',
       '                                cTmpUb[r * rowPitch],',
       '                                cGm[ks * splitStride +',
       '                                    tileOff +']),
    J(['                            DataCopyPad(',
       '                                cTmpUb[r * rowPitch],',
       '                                cGm[bufBase + ks * splitStride +',
       '                                    tileOff +']))

# ========================================== 15. AIV: close the strip loop
rep(J(['            Fence<HardEvent::MTE3_S>(pipe);',
       '        }',
       '    }',
       '',
       '    /*',
       '     * Phase 3: on-device final merge']),
    J(['            Fence<HardEvent::MTE3_S>(pipe);',
       '        }',
       '        }',
       '    }',
       '',
       '    /*',
       '     * Phase 3: on-device final merge']))

# ==================================== 16. LaunchMmaPath: pass chunkW through
rep(J(['    uint32_t blocks,',
       '    uint32_t nChunksTask,',
       '    uint32_t realChunks,',
       '    uint32_t mSub,',
       '    uint32_t kSplits,',
       '    aclrtStream stream,',
       '    bool transposeX1,',
       '    bool transposeX2)']),
    J(['    uint32_t blocks,',
       '    uint32_t nChunksTask,',
       '    uint32_t realChunks,',
       '    uint32_t chunkW,',
       '    uint32_t mSub,',
       '    uint32_t kSplits,',
       '    aclrtStream stream,',
       '    bool transposeX1,',
       '    bool transposeX2)']))
rep(J(['                nChunksTask, realChunks, 2U * blocks,',
       '                mSub, kSplits);']),
    J(['                nChunksTask, realChunks, chunkW, 2U * blocks,',
       '                mSub, kSplits);']), count=4)
rep(J(['            mmaBlocks,',
       '            nChunksTask,',
       '            realChunks,',
       '            mSub,',
       '            kSplits,',
       '            stream,']),
    J(['            mmaBlocks,',
       '            nChunksTask,',
       '            realChunks,',
       '            mmaChunkW,',
       '            mSub,',
       '            kSplits,',
       '            stream,']), count=2)

# ================================ 17. Launch: strip sizing + C bytes
rep(J(['    /* full fp32 C matrix in GM workspace',
       '       (kSplits partial copies when split) */',
       '    static DevScratch mmaCWs;',
       '    static DevScratch mmaPWs;',
       '',
       '    const size_t mmaCBytes =',
       '        static_cast<size_t>(kSplits) *',
       '        shape.b *',
       '        shape.m *',
       '        ((static_cast<size_t>(shape.n) + 7) &',
       '         ~static_cast<size_t>(7)) *',
       '        sizeof(float);']),
    J(['    /*',
       '     * v5: the C workspace holds TWO N strips instead of the whole [B,M,N]',
       '     * matrix, so the AIC leg can fill the next strip while the AIV leg drains',
       '     * the previous one. Residency drops from B*M*N floats to',
       '     * 2*B*M*stripW floats. Striping needs one slot column per chunk',
       '     * (realChunks == nChunksTask) and shape.k <= BMM_V5_STRIPE_MAX_K, and the',
       '     * strip width is rounded up to a multiple of 2*MMA_TN so a task n-pair',
       '     * tiles never straddle a strip boundary; otherwise the v2 chunk width and',
       '     * a single full-width strip are used (previous bulk behaviour).',
       '     */',
       '    const uint32_t mmaChunkWBase =',
       '        std::min<uint32_t>(',
       '            ((shape.n + realChunks - 1) / realChunks + 63U) & ~63U,',
       '            N_CHUNK);',
       '    const bool mmaStriped =',
       '        (realChunks == nChunksTask) &&',
       '        (shape.k <= BMM_V5_STRIPE_MAX_K);',
       '    const uint32_t mmaChunkW =',
       '        mmaStriped',
       '            ? std::min<uint32_t>(',
       '                  N_CHUNK,',
       '                  (mmaChunkWBase + 2U * MMA_TN - 1U) &',
       '                      ~(2U * MMA_TN - 1U))',
       '            : mmaChunkWBase;',
       '    const uint32_t mmaStrips = mmaStriped ? realChunks : 1U;',
       '    const uint32_t mmaStripW =',
       '        (mmaStrips == 1U) ? ((shape.n + 7U) & ~7U) : mmaChunkW;',
       '    static DevScratch mmaCWs;',
       '    static DevScratch mmaPWs;',
       '',
       '    const size_t mmaCBytes =',
       '        (mmaStrips == 1U ? 1U : 2U) *',
       '        static_cast<size_t>(kSplits) *',
       '        shape.b *',
       '        shape.m *',
       '        static_cast<size_t>(mmaStripW) *',
       '        sizeof(float);']))

open(DST, 'w', encoding='utf-8', newline='').write(raw.replace('\n', '\r\n'))
print('wrote %s: %d lines (was %d)' % (DST, len(raw.split('\n')), N))
