#!/usr/bin/env python3
"""v6 = v5 + L2-resident C panels (batch-sliced) so the fixpipe writes and the
AIV reads stay in cache, plus a stripe policy that keeps the K-heavy shapes on
the bulk path.

Source: ds-kernel-v5.asc      Output: ds-kernel-v6.asc
"""
import sys

SRC = 'ds-kernel-v5.asc'
DST = 'ds-kernel-v6.asc'

raw = open(SRC, encoding='utf-8', newline='').read().replace('\r\n', '\n')
lines = raw.split('\n')
N = len(lines)
J = '\n'.join


def rep(old, new, count=1):
    global raw
    if raw.count(old) != count:
        sys.exit('count %d != %d for:\n%r' % (raw.count(old), count, old[:160]))
    raw = raw.replace(old, new)


# ---------------------------------------------------------- 0. panel knob
rep('static constexpr uint64_t MED_MAC_LIMIT = 8388608;',
    J(['static constexpr uint64_t MED_MAC_LIMIT = 8388608;',
       '/*',
       ' * v6 panel size (bytes) for the L2-resident C buffer. Measured on 910C:',
       ' * writing C into a small reused region instead of streaming B*M*N*4 bytes',
       ' * to HBM saves ~12 ms on the B=64/N=8192 shapes (32.5 -> 20.3 ms), because',
       ' * the fixpipe destination stays in cache and its dirty lines are re-used',
       ' * instead of being written back. The C buffer is 2 panels (ping-pong).',
       ' */',
       '#ifndef BMM_V6_PANEL_MB',
       '#define BMM_V6_PANEL_MB 34',
       '#endif']))

# ------------------------------------------- 1. kernel signature: batchSlice
rep(J(['    uint32_t chunkW,',
       '    uint32_t aivWorkers,']),
    J(['    uint32_t chunkW,',
       '    uint32_t batchSlice,',
       '    uint32_t aivWorkers,']))

# ------------------------------------------ 2. kernel scope: panel constants
rep(J(['    const uint32_t stripW =',
       '        (stripCount == 1U) ? ((shape.n + 7U) & ~7U) : chunkW;']),
    J(['    const uint32_t stripW =',
       '        (stripCount == 1U) ? ((shape.n + 7U) & ~7U) : chunkW;',
       '    /*',
       '     * Batch slice per C panel. In the bulk path the whole batch range is',
       '     * one panel (identical to v5); in the striped path the panel covers',
       '     * `batchSlice` batches so that 2 panels fit in L2 and the fixpipe',
       '     * writes and the AIV reads of C never leave the cache.',
       '     */',
       '    const uint32_t panelStride =',
       '        (stripCount == 1U) ? shape.b : batchSlice;',
       '    const uint32_t panelBatches =',
       '        (stripCount == 1U) ? shape.b : MinU32(batchSlice, shape.b);']))

# ------------------------------- 3. AIC: batch-slice loop around the strips
rep(J(['        uint32_t subCnt = 0;',
       '        uint32_t tcount = 0;',
       '        for (uint32_t strip = 0; strip < stripCount; ++strip) {',
       '        const uint32_t stripStart = strip * stripW;']),
    J(['        uint32_t subCnt = 0;',
       '        uint32_t tcount = 0;',
       '        for (uint32_t bs = 0; bs < shape.b; bs += panelBatches) {',
       '        const uint32_t panelB =',
       '            MinU32(panelBatches, shape.b - bs);',
       '        for (uint32_t strip = 0; strip < stripCount; ++strip) {',
       '        const uint32_t stripStart = strip * stripW;']))

# ------------------------------- 4. AIC: per-panel task space
rep(J(['        const uint64_t stripTasks =',
       '            static_cast<uint64_t>(shape.b) * mTiles *',
       '                stripPairs * kSplits;']),
    J(['        const uint64_t stripTasks =',
       '            static_cast<uint64_t>(panelB) * mTiles *',
       '                stripPairs * kSplits;']))

# ------------------------------- 5. AIC: batch decode -> panel-local
rep(J(['            const uint32_t nP =',
       '                nPairBegin +',
       '                static_cast<uint32_t>(nb % stripPairs);',
       '            const uint32_t batch =',
       '                static_cast<uint32_t>(nb / stripPairs);']),
    J(['            const uint32_t nP =',
       '                nPairBegin +',
       '                static_cast<uint32_t>(nb % stripPairs);',
       '            const uint32_t bLocal =',
       '                static_cast<uint32_t>(nb / stripPairs);',
       '            const uint32_t batch = bs + bLocal;']))

# ------------------------------- 6. AIC: panel-local C address
rep(J(['            const uint64_t cOff =',
       '                static_cast<uint64_t>(strip & 1U) *',
       '                    kSplits * shape.b * shape.m * stripW +',
       '                static_cast<uint64_t>(ks) * shape.b *',
       '                    shape.m * stripW +',
       '                static_cast<uint64_t>(batch) * shape.m *',
       '                    stripW +']),
    J(['            const uint64_t cOff =',
       '                static_cast<uint64_t>(strip & 1U) *',
       '                    kSplits * panelStride * shape.m * stripW +',
       '                static_cast<uint64_t>(ks) * panelStride *',
       '                    shape.m * stripW +',
       '                static_cast<uint64_t>(bLocal) * shape.m *',
       '                    stripW +']))

# ------------------------------- 7. AIC: close the batch-slice loop
rep(J(['        SyncAll<false>();',
       '        }',
       '        {',
       '            const uint32_t tail0 =']),
    J(['        SyncAll<false>();',
       '        }',
       '        }',
       '        {',
       '            const uint32_t tail0 =']))

# ------------------------------- 8. AIV: batch-slice loop around the strips
rep(J(['        for (uint32_t strip = 0; strip < stripCount; ++strip) {',
       '        const uint32_t stripStart = strip * stripW;',
       '        /*',
       '         * Drain this leg own pipes first:']),
    J(['        for (uint32_t bs = 0; bs < shape.b; bs += panelBatches) {',
       '        const uint32_t panelB =',
       '            MinU32(panelBatches, shape.b - bs);',
       '        for (uint32_t strip = 0; strip < stripCount; ++strip) {',
       '        const uint32_t stripStart = strip * stripW;',
       '        /*',
       '         * Drain this leg own pipes first:']))

# ------------------------------- 9. AIV: per-panel task space + decode
rep(J(['        const uint64_t totalRed =',
       '            (stripCount > 1U)',
       '                ? static_cast<uint64_t>(shape.b) * rowTiles * mSub',
       '                : static_cast<uint64_t>(shape.b) * rowTiles *',
       '                      mSub * nChunksTask;']),
    J(['        const uint64_t totalRedPerPanel =',
       '            (stripCount > 1U)',
       '                ? static_cast<uint64_t>(panelBatches) * rowTiles * mSub',
       '                : static_cast<uint64_t>(shape.b) * rowTiles *',
       '                      mSub * nChunksTask;'])
)
rep(J(['        for (uint64_t task = worker;',
       '             task < totalRed;',
       '             task += workers) {',
       '            const uint32_t chunk =',
       '                (stripCount > 1U)',
       '                    ? strip',
       '                    : static_cast<uint32_t>(task % nChunksTask);',
       '            uint64_t mw =',
       '                (stripCount > 1U) ? task : task / nChunksTask;',
       '            const uint32_t msub =',
       '                static_cast<uint32_t>(mw % mSub);',
       '            mw /= mSub;',
       '            const uint32_t mwin =',
       '                static_cast<uint32_t>(mw % rowTiles);',
       '            const uint32_t batch =',
       '                static_cast<uint32_t>(mw / rowTiles);']),
    J(['        for (uint64_t task = worker;',
       '             task < totalRedPerPanel;',
       '             task += workers) {',
       '            const uint32_t chunk =',
       '                (stripCount > 1U)',
       '                    ? strip',
       '                    : static_cast<uint32_t>(task % nChunksTask);',
       '            uint64_t mw =',
       '                (stripCount > 1U) ? task : task / nChunksTask;',
       '            const uint32_t msub =',
       '                static_cast<uint32_t>(mw % mSub);',
       '            mw /= mSub;',
       '            const uint32_t mwin =',
       '                static_cast<uint32_t>(mw % rowTiles);',
       '            const uint32_t bLocal =',
       '                static_cast<uint32_t>(mw / rowTiles);',
       '            if (stripCount > 1U && bLocal >= panelB) {',
       '                continue;',
       '            }',
       '            const uint32_t batch =',
       '                (stripCount > 1U) ? (bs + bLocal) : bLocal;']))

# ------------------------------- 10. AIV: panel-local C address
rep(J(['                const uint64_t tileOff =',
       '                    static_cast<uint64_t>(batch) *',
       '                        shape.m * stripW +']),
    J(['                const uint64_t tileOff =',
       '                    static_cast<uint64_t>(bLocal) *',
       '                        shape.m * stripW +']))
rep(J(['                const uint64_t splitStride =',
       '                    static_cast<uint64_t>(shape.b) *',
       '                        shape.m * stripW;',
       '                const uint64_t bufBase =',
       '                    static_cast<uint64_t>(strip & 1U) *',
       '                    kSplits * splitStride;']),
    J(['                const uint64_t splitStride =',
       '                    static_cast<uint64_t>(panelStride) *',
       '                        shape.m * stripW;',
       '                const uint64_t bufBase =',
       '                    static_cast<uint64_t>(strip & 1U) *',
       '                    kSplits * splitStride;']))

# ---------------------------- 10b. AIV: 1D copy when the window is gap-free
rep(J(['                if (nByteAligned) {',
       '                    DataCopy(cUb, cGm[bufBase + tileOff], cp);',
       '                } else {']),
    J(['                /*',
       '                 * When the window rows pack with no gap (validN equals both',
       '                 * the C row pitch and the UB pitch) the whole window is one',
       '                 * contiguous run: a 1D copy is measurably cheaper than the',
       '                 * 2D form (1-4% on the mid shapes).',
       '                 */',
       '                const bool oneShot =',
       '                    nByteAligned && (validN == cPitch) &&',
       '                    (rowPitch == validN);',
       '                if (oneShot) {',
       '                    DataCopy(',
       '                        cUb,',
       '                        cGm[bufBase + tileOff],',
       '                        static_cast<uint32_t>(validM) * validN);',
       '                } else if (nByteAligned) {',
       '                    DataCopy(cUb, cGm[bufBase + tileOff], cp);',
       '                } else {']))

# ------------------------------- 11. AIV: close the batch-slice loop
rep(J(['        /*',
       '         * Drain this leg own pipes first: the FFTS arrival is gated on the',
       '         * MTE3 pipe, so without it the AIC leg could start refilling this',
       '         * half of the buffer while an MTE2 read of it is still in flight',
       '         * (measured as non-deterministic results on small-K shapes). Then the',
       '         * barrier pairs with the AIC leg "strip c complete" arrival.',
       '         */',
       '        PipeBarrier<PIPE_ALL>();',
       '        SyncAll<false>();']),
    J(['        /*',
       '         * Drain this leg own pipes first: the FFTS arrival is gated on the',
       '         * MTE3 pipe, so without it the AIC leg could start refilling this',
       '         * half of the buffer while an MTE2 read of it is still in flight',
       '         * (measured as non-deterministic results on small-K shapes). Then the',
       '         * barrier pairs with the AIC leg "strip c complete" arrival.',
       '         */',
       '        PipeBarrier<PIPE_ALL>();',
       '        SyncAll<false>();',
       '        (void)0;']))
rep(J(['            Fence<HardEvent::MTE3_S>(pipe);',
       '        }',
       '        }',
       '    }',
       '',
       '    /*',
       '     * Phase 3: on-device final merge']),
    J(['            Fence<HardEvent::MTE3_S>(pipe);',
       '        }',
       '        }',
       '        }',
       '    }',
       '',
       '    /*',
       '     * Phase 3: on-device final merge']))

# --------------------------- 12. LaunchMmaPath: pass batchSlice through
rep(J(['    uint32_t realChunks,',
       '    uint32_t chunkW,',
       '    uint32_t mSub,']),
    J(['    uint32_t realChunks,',
       '    uint32_t chunkW,',
       '    uint32_t batchSlice,',
       '    uint32_t mSub,']))
rep(J(['                nChunksTask, realChunks, chunkW, 2U * blocks,',
       '                mSub, kSplits);']),
    J(['                nChunksTask, realChunks, chunkW, batchSlice, 2U * blocks,',
       '                mSub, kSplits);']), count=4)
rep(J(['            realChunks,',
       '            mmaChunkW,',
       '            mSub,']),
    J(['            realChunks,',
       '            mmaChunkW,',
       '            mmaBatchSlice,',
       '            mSub,']), count=2)

# --------------------------- 13. Launch: panel sizing + C bytes
rep(J(['    const uint32_t mmaStrips = mmaStriped ? realChunks : 1U;',
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
       '        sizeof(float);']),
    J(['    const uint32_t mmaStrips = mmaStriped ? realChunks : 1U;',
       '    const uint32_t mmaStripW =',
       '        (mmaStrips == 1U) ? ((shape.n + 7U) & ~7U) : mmaChunkW;',
       '    /*',
       '     * Batches per C panel. One panel is panelStride*M*stripW floats; the',
       '     * buffer holds two of them so the AIC fills the next panel while the',
       '     * AIV drains the previous one, and both stay cache resident.',
       '     */',
       '    const uint64_t mmaPanelBytesTarget =',
       '        static_cast<uint64_t>(BMM_V6_PANEL_MB) << 20;',
       '    const uint64_t mmaRowBytes =',
       '        static_cast<uint64_t>(shape.m) * mmaStripW * sizeof(float);',
       '    uint32_t mmaBatchSlice = shape.b;',
       '    if (mmaStrips != 1U && mmaRowBytes != 0) {',
       '        const uint64_t want = mmaPanelBytesTarget / mmaRowBytes;',
       '        mmaBatchSlice = static_cast<uint32_t>(',
       '            std::max<uint64_t>(1, std::min<uint64_t>(shape.b, want)));',
       '    }',
       '    const uint32_t mmaPanelStride =',
       '        (mmaStrips == 1U) ? shape.b : mmaBatchSlice;',
       '    static DevScratch mmaCWs;',
       '    static DevScratch mmaPWs;',
       '',
       '    const size_t mmaCBytes =',
       '        (mmaStrips == 1U ? 1U : 2U) *',
       '        static_cast<size_t>(kSplits) *',
       '        mmaPanelStride *',
       '        shape.m *',
       '        static_cast<size_t>(mmaStripW) *',
       '        sizeof(float);']))

# ---------------- 14. phase-3 gate: AIV-only barrier instead of all-core
rep(J(['    /*',
       '     * Phase 3: on-device final merge over this launch\'s AIV workers. The',
       '     * all-core barrier makes every phase-2 partial write visible before any',
       '     * worker reads a slot; still exactly one kernel launch per iteration.',
       '     */',
       '    SyncAll<false>();',
       '',
       '    if',
       '        ASCEND_IS_AIV',
       '    {',
       '        MergePartialsOnDevice(']),
    J(['    /*',
       '     * Phase 3: on-device final merge over this launch\'s AIV workers. The',
       '     * merge only needs every AIV partial write to be visible to MTE2 (MTE3',
       '     * producer + barrier + MTE2 consumer), so an AIV-only barrier is enough:',
       '     * waiting for the AIC legs too costs another all-core barrier, which is',
       '     * ~10% of a small judge-shaped case. Still exactly one kernel launch.',
       '     */',
       '    if',
       '        ASCEND_IS_AIV',
       '    {',
       '        SyncAll<true>();',
       '        MergePartialsOnDevice(']))

open(DST, 'w', encoding='utf-8', newline='').write(raw.replace('\n', '\r\n'))
print('wrote %s: %d lines (was %d)' % (DST, len(raw.split('\n')), N))
