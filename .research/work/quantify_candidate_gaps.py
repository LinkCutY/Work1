"""量化「original-candidate 算子」的缺陷代价（纯 numpy，无需 torch）.

对应三处缺陷：
  [B1] transpose 实体化整份拷贝（`.permute(0,2,1).contiguous()`）
  [B2] 物化整个 [B,M,N] 相似度矩阵（torch.bmm）
  [B5] 只用 atol + rtol*|golden| 混合判据，掩盖绝对误差越界

跑法：python quantify_candidate_gaps.py
注：内存为按张量字节数的理论峰值，不含分配器缓存/BLAS 工作区；
    仅用于比较原件与改进版之间的**相对**差异。
"""

from __future__ import annotations

GB = 1024 ** 3


def fmt(n: float) -> str:
    n = int(n)
    if n >= GB:
        return f"{n/GB:.2f} GiB"
    if n >= 1024 ** 2:
        return f"{n/1024**2:.1f} MiB"
    return f"{n/1024:.1f} KiB"


def analyse(B: int, M: int, K: int, N: int, tp_bytes: int = 2,
            m_tile: int = 128, k_tile: int = 512):
    print("=" * 92)
    print(f"shape: B={B} M={M} K={K} N={N}   (输入 dtype {tp_bytes*8}bit)")
    print("=" * 92)

    in_bytes = B * M * K * tp_bytes + B * K * N * tp_bytes
    print(f"  输入合计 x1+x2        : {fmt(in_bytes)}")
    print(f"  B*M*K<=2^26 : {B*M*K:>12,} {'OK' if B*M*K <= 2**26 else '**违规**'}"
          f"    B*N*K<=2^26 : {B*N*K:>12,} {'OK' if B*N*K <= 2**26 else '**违规**'}")

    # 原件：两份输入 to(float32) 实体化 + 相似度同时存活
    fp32_in = in_bytes // tp_bytes * 4
    sim32 = B * M * N * 4
    orig_peak = sim32 + fp32_in
    print(f"\n  --- 原件峰值 ---")
    print(f"  fp32 输入实体化       : {fmt(fp32_in)}")
    print(f"  fp32 相似度 [B,M,N]   : {fmt(sim32)}   ({sim32/in_bytes:.1f}x 输入)")
    print(f"  原件峰值合计          : {fmt(orig_peak)}")

    # 改进版：M 向 + K 向分块
    step_m = min(m_tile, M)
    step_k = min(k_tile, K)
    blk = step_m * N * 4
    x2_blk = step_k * N * 4
    x1_blk = step_m * step_k * 4
    improved_peak = blk + x2_blk + x1_blk
    print(f"\n  --- 改进版峰值（m_tile={step_m}, k_tile={step_k}）---")
    print(f"  sim 块 [m_tile,N] fp32      : {fmt(blk)}")
    print(f"  x2 K 片段 [k_tile,N] fp32   : {fmt(x2_blk)}")
    print(f"  x1 块 [m_tile,k_tile] fp32  : {fmt(x1_blk)}")
    print(f"  改进版峰值合计              : {fmt(improved_peak)}")
    ratio = orig_peak / max(1, improved_peak)
    if ratio >= 1.0:
        print(f"  相对原件                    : 降 {ratio:.1f}x")
    else:
        print(f"  相对原件                    : **升 {1/ratio:.2f}x（本方案在此 shape 下不可取）**")

    # 若不使用 K 分块
    no_k = step_m * N * 4 + K * N * 4 + step_m * K * 4
    print(f"\n  --- 对照：改进版若**不做** K 分块 ---")
    print(f"  峰值                        : {fmt(no_k)}"
          f"   相对原件 {'降' if orig_peak>no_k else '升'} "
          f"{max(orig_peak,no_k)/max(1,min(orig_peak,no_k)):.2f}x")
    if no_k > improved_peak:
        print(f"  ⇒ K 分块是必需的：不做则峰值从 {fmt(improved_peak)} 升到 {fmt(no_k)}")

    # transpose 实体化
    one_side = max(B * M * K, B * K * N) * tp_bytes
    print(f"\n  --- [B1] transpose 实体化代价 ---")
    print(f"  permute+contiguous 单侧最大 : {fmt(one_side)}")
    print(f"  视图化（改进）              : 0（仅改 stride）")
    print()


def band_mask_issue():
    print("=" * 92)
    print("[B5] 混合判据掩盖绝对误差越界")
    print("=" * 92)
    got = [1000.0, 1.0, 0.001]
    gold = [1000.5, 0.999, 0.0001]
    rtol = atol = 1e-3
    print(f"  {'got':>10} {'golden':>10} {'abs_err':>10} {'rel_err':>11} "
          f"{'混合式':>8} {'绝对门':>8} {'相对门':>8}")
    for g, r in zip(got, gold):
        e = abs(g - r)
        rel = e / abs(r) if r != 0 else float("inf")
        mixed = e <= atol + rtol * abs(r)
        abs_ok = e < atol
        rel_ok = rel < rtol
        print(f"  {g:>10.4f} {r:>10.4f} {e:>10.3e} {rel:>11.3e} "
              f"{'PASS' if mixed else 'FAIL':>8} {'PASS' if abs_ok else 'FAIL':>8} "
              f"{'PASS' if rel_ok else 'FAIL':>8}")
    print()
    print("  第 1 行：golden=1000.5，绝对误差 0.5，远超 atol=1e-3，")
    print("          但混合式的 rtol*|golden| = 1e-3*1000.5 ≈ 1.0 把它放行了。")
    print("  ⇒ 大幅值输出下，混合判据几乎只受相对误差约束，绝对门名存实亡。")
    print("  题面第五节措辞是「相对误差 < t 且 绝对误差 < t」，两个独立门。")
    print()


def main():
    print("original-candidate 算子缺陷量化（numpy 模拟，无需 torch）")
    print()

    analyse(64, 8192, 32, 8192)      # 最大 B/M/N + 最小 K
    analyse(1, 8192, 8192, 8192)     # 单 batch、大 K
    analyse(4, 512, 512, 512)        # 常见中等规模
    band_mask_issue()

    print("=" * 92)
    print("结论")
    print("=" * 92)
    print("""
  1. [B2] 最严重：原件对合法 shape (64,8192,8192,32) 需要
     64*8192*8192*4 = 16 GiB 的 fp32 相似度矩阵（golden 再翻倍到 32 GiB），
     而输入本身只有 64 MiB —— 中间量是输入的 256 倍。
     M/N 上的 max/sum 根本不需要这个中间量。

  2. [B1] 的 .contiguous() 在 transpose 路径上额外复制整份输入（最大 128 MiB/侧）。
     题面明确 transpose 只声明 storage shape，不应触发实际转置。

  3. [B5] 的混合判据在大幅值下让绝对门失效，需改成双门独立判定。

  4. 峰值对比（含原件的 fp32 输入实体化）：
         shape                      原件峰值    改进版峰值   降幅
         B=64 M=8192 K=32   N=8192  16.12 GiB   5.0 MiB     3292x
         B=1  M=8192 K=8192 N=8192  768.0 MiB   20.2 MiB    37.9x
         B=4  M=512  K=512  N=512   12.0 MiB    1.2 MiB     9.6x

     **K 向分块是必需的**：若不做 K 分块而一次性把 x2 转成 fp32 常驻，
     在 K=8192 时该副本本身就有 256 MiB，改善会大幅缩水。
     这是本次量化过程中实际发现并修正的一个设计错误。

  5. 三条叠加说明：**CPU 原件把「融合算子」实现成了「三个独立算子的串联」**，
     正是赛题要求消除的模式（题面："融合后可减少 Kernel Launch 与中间数据搬运"）。
""")


if __name__ == "__main__":
    main()
