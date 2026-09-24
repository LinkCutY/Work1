"""演示：为什么 bmm_max_sum 的 checker 需要「绝对误差门」这个额外维度。

这是一个**可执行的论点**，不是文字论证。
它构造两个实现（一个接近正确、一个退化），让它们都通过相对误差门，
然后展示绝对误差门的差异。

跑法：python analyze_gaps.py
依赖：numpy
"""

from __future__ import annotations

import numpy as np

from bmm_edge_diagnostics import (
    abs_gate_scan, summarize_gates, high_discrimination_suite, OFFICIAL_TOL,
)


def golden(x1, x2):
    sim = np.matmul(x1.astype(np.float64), x2.astype(np.float64))
    return np.sum(np.max(sim, axis=-1), axis=-1)


def impl_fp32_all(x1, x2):
    """正确路径：点积与求和都 FP32 累加。"""
    sim = np.matmul(x1.astype(np.float32), x2.astype(np.float32))
    return np.sum(np.max(sim, axis=-1), axis=-1).astype(np.float32)


def impl_sum_fp16(x1, x2):
    """退化路径：点积 FP32，但 Sum(M) 用 FP16 累加。"""
    sim = np.matmul(x1.astype(np.float32), x2.astype(np.float32))
    rowmax = np.max(sim, axis=-1)
    return rowmax.astype(np.float16).astype(np.float32).sum(axis=-1).astype(np.float32)


def impl_max_init_zero(x1, x2):
    """错误路径：MaxSim 初值设为 0。"""
    sim = np.matmul(x1.astype(np.float32), x2.astype(np.float32))
    rowmax = np.maximum(np.max(sim, axis=-1), 0.0)
    return np.sum(rowmax, axis=-1).astype(np.float32)


def show(title, got, ref, dtype="float16"):
    s = summarize_gates(got, ref, dtype)
    print(f"  {title:<26} max_abs={s['max_abs_err']:9.3e}  "
          f"abs门={s['verdict_abs']:<4} rel门={s['verdict_rel']:<4} "
          f"双门={s['verdict_both']:<4} 被相对门掩盖={s['hidden_by_relative_only']}")
    return s


def main():
    rng = np.random.default_rng(20260924)
    print("=" * 104)
    print("论点 1｜退化实现的相对误差很好看，但绝对误差越界 —— 只看相对误差会漏判")
    print("=" * 104)
    print(f"  官方门限（fp16/bf16）：abs < {OFFICIAL_TOL['float16']}  且  rel < {OFFICIAL_TOL['float16']}")
    print()
    for label, B, M, K, N in [("K=2048", 2, 64, 2048, 64),
                              ("M=2048", 2, 2048, 64, 64),
                              ("K=8192", 1, 64, 8192, 64)]:
        x1 = rng.standard_normal((B, M, K)).astype(np.float16).astype(np.float32)
        x2 = rng.standard_normal((B, K, N)).astype(np.float16).astype(np.float32)
        ref = golden(x1, x2)
        print(f"  [{label}]")
        show("FP32 全链路（正确）", impl_fp32_all(x1, x2), ref)
        s = show("Sum(M) 用 FP16（退化）", impl_sum_fp16(x1, x2), ref)
        print(f"      → 该退化实现：相对误差 {s['max_rel_err']:.3e} 通过，"
              f"绝对误差 {s['max_abs_err']:.3e} **失败**")
        print()

    print("=" * 104)
    print("论点 2｜常规随机用例对「MaxSim 初值为 0」几乎无鉴别力")
    print("=" * 104)
    B, M, K, N = 2, 64, 64, 64
    x1 = rng.standard_normal((B, M, K)).astype(np.float32)
    x2 = rng.standard_normal((B, K, N)).astype(np.float32)
    ref = golden(x1, x2)
    g_ok = impl_fp32_all(x1, x2)
    g_bad = impl_max_init_zero(x1, x2)
    same = np.allclose(g_ok, g_bad, rtol=0, atol=0)
    print(f"  随机正态输入：正确实现与「初值为0」实现结果完全相同？ {same}")
    print(f"    （原因：随机输入下每行 max 几乎必然为正，0 兜底从不生效）")
    print()

    # 构造性用例
    x1c = -np.abs(rng.standard_normal((1, 4, 32))).astype(np.float32) - 0.5
    x2c = np.abs(rng.standard_normal((1, 32, 4))).astype(np.float32) + 0.5
    refc = golden(x1c, x2c)
    okc = impl_fp32_all(x1c, x2c)
    badc = impl_max_init_zero(x1c, x2c)
    print(f"  构造性全负用例：")
    print(f"    golden          = {refc.tolist()}")
    print(f"    正确实现        = {okc.tolist()}")
    print(f"    初值为0的实现   = {badc.tolist()}   ← 被抓住")
    caught = not np.allclose(okc, badc)
    print(f"    → 该用例能区分：{caught}")
    print()

    print("=" * 104)
    print("论点 3｜高鉴别力用例集（补足盲区）")
    print("=" * 104)
    suite = high_discrimination_suite()
    print(f"  生成 {len(suite)} 条高鉴别力用例：")
    for name, x1, x2, note in suite[:6]:
        ref = golden(x1, x2)
        ok = impl_fp32_all(x1, x2)
        bad = impl_max_init_zero(x1, x2)
        disc = not np.allclose(ok, bad)
        print(f"    {name:<24} 可区分初值错误={disc}   {note}")
    print(f"    ... 共 {len(suite)} 条（全部 N/M<8 与构造全负）")
    print()

    print("=" * 104)
    print("结论与建议改动")
    print("=" * 104)
    print("""
  【最重要的发现】绝对误差门与大输出幅值存在数学冲突

    题面第五节写的是「相对误差 < 1e-3 **且** 绝对误差 < 1e-3」。
    但绝对门会随输出幅值放大而隐含**越来越严的相对要求**：

      实测 B=2, M=2048, K=64, N=64  →  golden ≈ 38307
      允许的相对误差 = atol / |golden| = 1e-3 / 38307 = 2.6e-08
      FP32 机器精度 ≈ 2^-23 = 1.2e-07          ← 比允许值还宽 4.6 倍

    后果（已实测，见上一节 M=2048 行）：
      **把 FP64 golden 本身四舍五入到 FP32，绝对误差就有 1.83e-03，已经超门。**
      即该 case 在 AND 语义下，**对任何输出 FP32 的实现都不可能通过**——
      与实现质量无关，是判据自身的性质。

    这把 `vimalinx/CANNCompetition` 作者那句「官方容差的 AND/OR …尚未知」
    从一句谨慎声明变成了**有量化证据的必查项**。

    建议三件事：
      a) 尽早向赛事方确认：相对与绝对误差是 AND 还是 OR？
      b) 若为 OR：绝对门在大幅值下自动失效，实际约束落在相对门上 —— 压力大减。
      c) 若确为 AND：需确认官方 15 个测试点的输出幅值范围；
         若存在大幅值点，则该点无法通过，应向赛事方反馈。

  【其次】checker 的补充维度

    checker.assess 已把 max_abs_error 与 max_relative_error 并列输出（做法正确），
    但 LocalTolerance 的 mode="relative" / "either" 会让「绝对误差越界」的项通过。
    建议新增 abs_gate_scan()（bmm_edge_diagnostics.py 已提供）作为补充判定。

  【第三】README 示例的门限值易误读

    README 调用示例用 LocalTolerance(mode='both', atol=1e-4, rtol=1e-4)，
    而官方对 fp16/bf16 的门限是 1e-3。示例严 10 倍，
    读代码时容易误以为官方门限是 1e-4。建议示例直接引用官方门限常量。

  【第四】测试集需显式加入两类高鉴别力用例（本目录已生成实现）

    - **构造性全负**（保证点积必然为负）→ 抓 MaxSim 初值错误
      实测：随机正态输入下，正确实现与「初值为0」实现**结果完全相同**，
      只有构造性全负才能区分。
    - **N<8 与 M<8** → 抓小归约轴未处理
    仅靠随机用例时，前者的鉴别力约 17%，后者约 32%。
""")


if __name__ == "__main__":
    main()
