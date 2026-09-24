#!/usr/bin/env python3
"""BatchMatmulMaxSum @ 910C(A3/2201) tiling 搜索（离线，纯标准库）

按官方约束枚举 (baseM, baseN, baseK, dbL0A/B/C, depthA1/B1, stepK)，按
"GM 写次数(tile 数) / L1 装载量 / 尾块浪费" 排序，并报出限制项。

约束出处（本地 CANN 8.5 快照）：
 [C2] baseM*baseK*sizeof(A)*dbL0A        < l0a_size     TCubeTiling结构体.md 表2
 [C3] baseN*baseK*sizeof(B)*dbL0B        < l0b_size     同上
 [C4] baseM*baseN*sizeof(l0c_type)*dbL0C < l0c_size     同上
 [C5] stepM*stepKa*db=depthA1 ; stepN*stepKb*db=depthB1 同上
 [C6] AL1+BL1 <= L1 (转置: A=ceilD16(baseM)*baseK*depthA1*2, B=baseN*baseK*depthB1*2;
                     非转置: A=baseM*baseK*depthA1*2, B=ceilD16(baseN)*baseK*depthB1*2) 同上
 [C7] baseM/baseN/baseK 16 元素对齐（fp16/bf16 C0=16）  TCubeTiling结构体.md 表1
 [C8] SetFixSplit 只能钉 baseM/baseN，baseK 仅 -1        SetFixSplit.md
 [C9] baseM<=ceil16(singleM), baseN<=alignC0(singleN)   SetFixSplit.md
 [C11] baseM*baseN*sizeof(C) <= L0C                     SetFixSplit.md
 [C12] 容量：UB 192KB/L0C 128KB/BT 1KB 官方；L1 512KB 官方示例；L0A=L0B=64KB 第三方
 [K1] C 走 GM（2201 无 L0C→UB 直通）→ 每 tile 一次 GM 写 => 最小化 tile 数
 [K2] UB_used ≈ baseM*baseN*4(中转) + baseM*8*4(行最大值) + 3*baseM*4
 [K3] WholeReduceMax mask<=64 => baseN 分 ceil(baseN/64) 块; repeatTime<=255 => baseM<=255
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, field

C0 = 16      # fp16/bf16 分形内轴
EA = 2       # 输入字节
EC = 4       # fp32 累加字节


def cd(a, b):
    return (a + b - 1) // b


def al(x, a):
    return cd(x, a) * a


@dataclass
class Platform:
    ub: int = 192 * 1024
    l1: int = 512 * 1024
    l0a: int = 64 * 1024
    l0b: int = 64 * 1024
    l0c: int = 128 * 1024
    aic: int = 24
    aiv: int = 48


@dataclass
class Cand:
    bm: int
    bn: int
    bk: int
    dba: int
    dbb: int
    dbc: int
    da1: int
    db1: int
    ska: int
    skb: int
    l1: int
    tiles: int
    load: int
    waste: float
    lim: list = field(default_factory=list)

    @property
    def ok(self):
        return not self.lim


def depths(p, bm, bn, bk, k, db=2):
    """depthA1 = stepM*stepKa*db 等：按 L1 的一半预算给 A / B，尽量让 K 全载。"""
    a_layer = bm * bk * EA
    b_layer = bn * bk * EA
    kt = max(1, cd(k, bk))
    da = min(kt * db, max(db, ((p.l1 // 2) // max(1, a_layer)) // db * db))
    dbb = min(kt * db, max(db, ((p.l1 // 2) // max(1, b_layer)) // db * db))
    return da, dbb, max(1, da // db), max(1, dbb // db)


def ev(p, b, m, n, k, bm, bn, bk, dba=2, dbb=2, ta=False):
    lim = []
    if bm % 16 or bn % 16 or bk % 16:
        lim.append("align16")
    if bm * bk * EA * dba >= p.l0a:
        lim.append("L0A")
    if bn * bk * EA * dbb >= p.l0b:
        lim.append("L0B")
    one = bm * bn * EC
    dbc = 2 if one * 2 < p.l0c else 1
    if one * dbc >= p.l0c:
        lim.append("L0C")
    da, dbb2, ska, skb = depths(p, bm, bn, bk, k)
    if ta:
        a_l1 = cd(bm, C0) * bk * da * EA
        b_l1 = bn * bk * dbb2 * EA
    else:
        a_l1 = bm * bk * da * EA
        b_l1 = cd(bn, C0) * bk * dbb2 * EA
    l1 = a_l1 + b_l1
    if l1 > p.l1:
        lim.append("L1")
    ub_used = bm * bn * EC + bm * 8 * EC + 3 * bm * EC
    if ub_used > p.ub:
        lim.append("UB")
    if bm > 255:
        lim.append("rep255")
    tiles = b * cd(m, bm) * cd(n, bn)
    load = tiles * (bm * k + bn * k) * EA
    pad = cd(m, bm) * bm * cd(n, bn) * bn
    return Cand(bm, bn, bk, dba, dbb, dbc, da, dbb2, ska, skb, l1, tiles, load,
                (pad - m * n) / float(m * n), lim)


def search(p, b, m, n, k, top=6, ta=False):
    """对每个 (baseM, baseN) 取**合法且最大**的 baseK（L0A/L0B 越富余越好），
    再按 L0 装载量 / K-step 数 / tile 数 / 尾块浪费排序。

    排序键的依据：L0A/L0B 的装载字节 = M*K*2*(N/baseN) + N*K*2*(M/baseM)，
    只由 baseM/baseN 决定；baseK 越大，L0 填充的摊销越好、K 循环指令越少。
    """
    out = []
    for bm in (16, 32, 64, 96, 112, 128, 160, 192, 240, 256):
        if bm > 16 and bm > al(m, 16):
            continue                      # [C9] baseM <= ceil16(singleM)
        for bn in (32, 64, 96, 128, 160, 192, 240, 256, 320, 512):
            if bn > al(n, 16):
                continue                      # 超出 N 的实际长度没有意义
            best = None
            for bk in (16, 32, 64, 128, 256):
                if bk > k:
                    continue
                for dba, dbb in ((2, 2), (2, 1), (1, 2), (1, 1)):
                    c = ev(p, b, m, n, k, bm, bn, bk, dba, dbb, ta)
                    if not c.ok:
                        continue
                    key = (-c.bk, -(c.dba + c.dbb))
                    if best is None or key < best[0]:
                        best = (key, c)
            if best:
                out.append(best[1])
    out.sort(key=lambda c: (c.load, c.tiles, c.waste))
    return out[:top]


SHAPES = [(1, 8192, 8192, 8192), (1, 8192, 8192, 1024), (1, 4096, 4096, 4096),
          (1, 1024, 1024, 1024), (1, 512, 512, 512), (8, 512, 512, 512),
          (64, 128, 128, 512), (1, 8192, 64, 8192), (1, 64, 8192, 1024),
          (1, 1000, 1000, 1000), (1, 8191, 8191, 1023)]


def row(c):
    return (f"baseM={c.bm:<4d} baseN={c.bn:<4d} baseK={c.bk:<4d} "
            f"db={c.dba}/{c.dbb}/{c.dbc} depth={c.da1}/{c.db1} stepK={c.ska}/{c.skb} "
            f"L1={c.l1>>10:>4d}KB tiles={c.tiles:<7d} load={c.load>>20:>5d}MB "
            f"waste={c.waste*100:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--transposed-a", action="store_true")
    ap.add_argument("--ub", type=int, default=192 * 1024)
    ap.add_argument("--l1", type=int, default=512 * 1024)
    ap.add_argument("--l0a", type=int, default=64 * 1024)
    ap.add_argument("--l0b", type=int, default=64 * 1024)
    ap.add_argument("--l0c", type=int, default=128 * 1024)
    ap.add_argument("--aic", type=int, default=24)
    ap.add_argument("--aiv", type=int, default=48)
    a = ap.parse_args()
    p = Platform(a.ub, a.l1, a.l0a, a.l0b, a.l0c, a.aic, a.aiv)
    print(f"# 910C(A3/2201) UB={p.ub>>10}KB L1={p.l1>>10}KB L0A={p.l0a>>10}KB "
          f"L0B={p.l0b>>10}KB L0C={p.l0c>>10}KB AIC={p.aic} AIV={p.aiv}")
    shapes = SHAPES if a.sweep else [tuple(int(x) for x in a.shape.split(","))]
    for (b, m, n, k) in shapes:
        print(f"\n=== B={b} M={m} N={n} K={k} macs={b*m*n*k/1e9:.2f}G")
        res = search(p, b, m, n, k, a.top, a.transposed_a)
        if not res:
            print("  !! 无合法候选")
            continue
        for i, c in enumerate(res):
            print(f"  [{i}] {row(c)}")
        print(f"  -> 推荐 {row(res[0])}")


if __name__ == "__main__":
    main()
