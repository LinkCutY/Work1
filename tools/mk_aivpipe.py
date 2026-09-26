#!/usr/bin/env python3
"""Build variants/v6_aivpipe.asc: software-pipelined AIV C-drain.

The baseline AIV slab loop is DMA -> Fence(MTE2_V) -> reduce -> Fence(V_MTE2),
so the MTE2 read and the vector max never overlap: measured on 910C
case20 (64x8192x8192x32) the AIV leg costs DMA 10.7 ms + reduce 5.6 ms serially
while the AIC leg is 10.3 ms, i.e. ~6 ms of the 16.4 ms total is unoverlapped.

This variant double-buffers the cUb slab (2 x TILE_M x N_CHUNK fp32 = 64 KiB)
and issues each slab's DMA before consuming the previous one, with explicit
MTE2_V / V_MTE2 event IDs instead of blocking Fence<> pairs.
"""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "self" / "ds-kernel-v6.asc"
DST = ROOT / "variants" / "v6_aivque.asc"

lines = SRC.read_text().split("\n")

# ------------------------------------------------- 1. cUb queue (VECIN, 2)
i = next(i for i, ln in enumerate(lines) if "TBuf<TPosition::VECCALC> cWinBuf;" in ln)
lines[i] = "        TQue<TPosition::VECIN, 2> cUbQue;"
buf_line = next(i for i, ln in enumerate(lines)
                if "TILE_M * N_CHUNK * sizeof(CDtype)" in ln)
print("queue InitBuffer line:", buf_line + 1, repr(lines[buf_line]))
# pipe.InitBuffer(\n cUbQue,\n <size>);   ->  pipe.InitBuffer(cUbQue, 2, <size>);
assert lines[buf_line - 1].strip() == "cWinBuf,"
assert lines[buf_line - 2].strip() == "pipe.InitBuffer("
lines[buf_line - 2] = "            pipe.InitBuffer("
lines[buf_line - 1] = "                cUbQue,"
lines[buf_line] = "                2U, TILE_M * N_CHUNK * sizeof(CDtype));"
i2 = buf_line - 2
lines[i2:i2] = ["            pipe.InitBuffer("]  # keep line count stable: replaced below
del lines[i2]

_ = "no hand-rolled events: TQue<TPosition::VECIN, 2> owns the sync flags"

# --------------------------------------------------------- 3. slab loop swap
start = next(i for i, ln in enumerate(lines)
             if "for (uint32_t so = 0; so < cValid;" in ln)
end = next(i for i, ln in enumerate(lines) if "} /* end slab loop */" in ln)
print("swap lines", start + 1, "..", end + 1)
old = "\n".join(lines[start:end + 1])
assert old.count("for (uint32_t ks = 1;") == 1

NEW = r"""                /*
                 * Double-buffered slab drain. Two passes per iteration:
                 *   produce: issue slab sIdx's DMA into cUb[sIdx&1]
                 *   consume: reduce slab sIdx-1 from cUb[(sIdx-1)&1]
                 * so the MTE2 read of one slab overlaps the vector max of the
                 * previous one. Geometry (validN, rowPitch) is a pure function
                 * of the slab index, so the consumer recomputes its own.
                 */
                const uint32_t nSlabs =
                    (cValid + N_CHUNK - 1) / N_CHUNK;
                constexpr uint32_t cElemsBlk =
                    32U / sizeof(CDtype);
                for (uint32_t sIdx = 0;
                     sIdx <= nSlabs;
                     ++sIdx) {
                if (sIdx < nSlabs) {
                    const uint32_t so = sIdx * N_CHUNK;
                    const uint32_t nStart = cStart + so;
                    const uint32_t validN = MinU32(
                        N_CHUNK, shape.n - nStart);
                    const uint32_t rowPitch =
                        ((validN & 63U) == 0 && validN > 64)
                            ? validN
                            : N_CHUNK;
                    DataCopyParams cp;
                    cp.blockCount =
                        static_cast<uint16_t>(validM);
                    cp.blockLen =
                        static_cast<uint16_t>(
                            validN / cElemsBlk);
                    cp.srcStride =
                        static_cast<uint16_t>(
                            (cPitch - validN) / cElemsBlk);
                    cp.dstStride =
                        static_cast<uint16_t>(
                            (rowPitch - validN) / cElemsBlk);
                    const uint64_t tileOff =
                        static_cast<uint64_t>(bLocal) *
                            shape.m * stripW +
                        static_cast<uint64_t>(rowStart) *
                            stripW +
                        (nStart - stripStart);
                    const uint64_t splitStride =
                        static_cast<uint64_t>(panelStride) *
                            shape.m * stripW;
                    const uint64_t bufBase =
                        static_cast<uint64_t>(strip & 1U) *
                        kSplits * splitStride;
                    const bool nByteAligned =
                        (validN % cElemsBlk) == 0;
                    const bool oneShot =
                        nByteAligned && (validN == cPitch) &&
                        (rowPitch == validN);
                    LocalTensor<CDtype> cUbB =
                        cUbQue.AllocTensor<CDtype>();
                    if (oneShot) {
                        DataCopy(
                            cUbB,
                            cGm[bufBase + tileOff],
                            static_cast<uint32_t>(
                                validM) * validN);
                    } else if (nByteAligned) {
                        DataCopy(
                            cUbB,
                            cGm[bufBase + tileOff],
                            cp);
                    } else {
                        DataCopyExtParams icp;
                        icp.blockCount = 1;
                        icp.blockLen =
                            static_cast<uint32_t>(
                                validN *
                                sizeof(CDtype));
                        icp.srcStride = 0;
                        icp.dstStride = 0;
                        icp.rsv = 0;
                        DataCopyPadExtParams<CDtype> ipp;
                        ipp.isPad = false;
                        ipp.leftPadding = 0;
                        ipp.rightPadding = 0;
                        ipp.paddingValue = 0;
                        for (uint32_t r = 0;
                             r < validM; ++r) {
                            DataCopyPad(
                                cUbB[r * rowPitch],
                                cGm[bufBase + tileOff +
                                    static_cast<
                                        uint64_t>(r) *
                                        cPitch],
                                icp,
                                ipp);
                        }
                    }
                    for (uint32_t ks = 1;
                         ks < kSplits;
                         ++ks) {
                        Fence<HardEvent::V_MTE2>(pipe);
                        if (nByteAligned) {
                            DataCopy(
                                cTmpUb,
                                cGm[bufBase + ks * splitStride + tileOff],
                                cp);
                        } else {
                            DataCopyExtParams icp;
                            icp.blockCount = 1;
                            icp.blockLen =
                                static_cast<uint32_t>(
                                    validN *
                                    sizeof(CDtype));
                            icp.srcStride = 0;
                            icp.dstStride = 0;
                            icp.rsv = 0;
                            DataCopyPadExtParams<CDtype> ipp;
                            ipp.isPad = false;
                            ipp.leftPadding = 0;
                            ipp.rightPadding = 0;
                            ipp.paddingValue = 0;
                            for (uint32_t r = 0;
                                 r < validM; ++r) {
                                DataCopyPad(
                                    cTmpUb[r * rowPitch],
                                    cGm[bufBase + ks * splitStride +
                                        tileOff +
                                        static_cast<
                                            uint64_t>(r) *
                                            cPitch],
                                    icp,
                                    ipp);
                            }
                        }
                        Fence<HardEvent::MTE2_V>(pipe);
                        Add(
                            cUbB,
                            cUbB,
                            cTmpUb,
                            validM * rowPitch);
                    }
                    cUbQue.EnQue(cUbB);
                }
                if (sIdx > 0) {
                    const uint32_t pIdx = sIdx - 1U;
                    const uint32_t so = pIdx * N_CHUNK;
                    const uint32_t nStart = cStart + so;
                    const uint32_t validN = MinU32(
                        N_CHUNK, shape.n - nStart);
                    const uint32_t rowPitch =
                        ((validN & 63U) == 0 && validN > 64)
                            ? validN
                            : N_CHUNK;
                    LocalTensor<CDtype> cUbB =
                        cUbQue.DeQue<CDtype>();

                    Duplicate(
                        tileMaxVec,
                        -3.402823466e38f,
                        TILE_M);

                    if (rowPitch == validN) {
                        const uint32_t wRow = validN >> 6;
                        const uint32_t nPartial =
                            validM * wRow;
                        Duplicate(
                            reduceWork,
                            -3.402823466e38f,
                            nPartial * 8);
                        WholeReduceMax(
                            reduceWork,
                            cUbB,
                            64,
                            static_cast<int32_t>(
                                nPartial),
                            8,
                            1,
                            8,
                            ReduceOrder::ORDER_ONLY_VALUE);
                        WholeReduceMax(
                            tileMaxVec,
                            reduceWork,
                            static_cast<int32_t>(
                                wRow * 8),
                            static_cast<int32_t>(validM),
                            1,
                            1,
                            static_cast<int32_t>(wRow),
                            ReduceOrder::ORDER_ONLY_VALUE);
                    } else if (validN <= 64) {
                        WholeReduceMax(
                            tileMaxVec,
                            cUbB,
                            static_cast<int32_t>(validN),
                            static_cast<int32_t>(validM),
                            1,
                            1,
                            static_cast<int32_t>(
                                rowPitch / 8),
                            ReduceOrder::ORDER_ONLY_VALUE);
                    } else {
                        for (uint32_t r = 0; r < validM;
                             ++r) {
                            ReduceMax(
                                tileMaxVec[r],
                                cUbB[r * rowPitch],
                                reduceWork,
                                static_cast<int32_t>(
                                    validN),
                                false);
                        }
                    }

                    Max(
                        bestVec,
                        bestVec,
                        tileMaxVec,
                        validM);

                    cUbQue.FreeTensor(cUbB);
                }
                }"""

lines[start:end + 1] = NEW.split("\n")
DST.write_text("\n".join(lines))
print("wrote", DST)
