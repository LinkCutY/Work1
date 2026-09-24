"""针对 bmm_max_sum 的边界诊断器 —— 补足 reference/checker 覆盖不到的失效模式。

为什么需要它（两个已被实测支持的缺口，详见同目录 analyze_gaps.py）：

  缺口 1｜「绝对误差门」与大输出幅值存在**数学冲突**（比原先的理解更严重）
    题面写的是「相对误差 < 1e-3 **且** 绝对误差 < 1e-3」。但当输出幅值较大时，
    1e-3 的绝对门会隐含一个**比 FP32 可表示精度更严**的相对要求：
      实测 B=2,M=2048,K=64,N=64 → golden ≈ 38307
      ⇒ 允许的相对误差 = 1e-3/38307 = 2.6e-08
      而 FP32 机器精度约 1.2e-07（2^-23）
    结果：**把 FP64 golden 本身四舍五入到 FP32，绝对误差就有 1.83e-03，已超门**。
    ⇒ 该 case 在「AND」语义下**对任何 FP32 输出实现都不可能通过**。
    推论：官方判据要么是 OR（相对/绝对取其一），要么其 15 个测试点的输出幅值都很小。
    **这是必须尽早向赛事方确认的问题**，不是实现层面的 bug。

  缺口 2｜多数常规用例对两类错误无鉴别力
    实测统计（71 条用例）：
      E2 未处理小归约轴（N 或 M < 8）  仅 21/65（32.3%）可区分
      E3 MaxSim 初值设为 0            仅 11/65（16.9%）可区分
    即：不专门构造「全负」与「小归约轴」用例，写错了也测不出来。
    补充实测：**随机正态输入下，正确实现与「初值为0」实现结果完全相同**
    —— 必须用构造性全负输入（保证点积必然为负）才能抓住。

本模块只做诊断，不替代 checker，也不声称与官方一致。
依赖：numpy（因此可在无 torch 环境运行）。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np

# 官方门限（题面第五节）。注意是**严格小于**。
OFFICIAL_TOL = {"float16": 1e-3, "bfloat16": 1e-3, "float32": 1e-4}


@dataclass
class AbsGateFinding:
    """绝对误差通过/失败与相对误差判定不一致的情形。"""
    batch: int
    abs_err: float
    rel_err: float | None      # None 表示 golden==0 且误差非零（+inf 的显式表示）
    golden: float
    got: float
    abs_gate_pass: bool
    rel_gate_pass: bool
    hidden_by_relative_only: bool   # True = 只看相对误差会误判为通过


def abs_gate_scan(got, golden, atol: float, rtol: float,
                  golden_is_zero_eps: float = 0.0):
    """逐 batch 同时判绝对与相对误差，并标出「被相对误差掩盖」的项。

    这是对 checker.assess 的**补充**，不是替代：
      checker 的 mode="relative" 或 mode="either" 会让这类项通过；
      本函数专门把它们标出来。
    """
    g = np.asarray(got, dtype=np.float64).reshape(-1)
    r = np.asarray(golden, dtype=np.float64).reshape(-1)
    if g.shape != r.shape:
        raise ValueError(f"shape mismatch: got {g.shape}, golden {r.shape}")

    findings: list[AbsGateFinding] = []
    for i, (a, b) in enumerate(zip(g, r)):
        e = abs(a - b)
        if b != 0:
            rel = e / abs(b)
        else:
            rel = 0.0 if e == 0 else None
        abs_ok = e < atol                       # 严格小于，与题面一致
        rel_ok = (rel is not None) and (rel < rtol)
        hidden = abs_ok is False and rel_ok is True
        findings.append(AbsGateFinding(i, e, rel, b, a, abs_ok, rel_ok, hidden))
    return findings


def summarize_gates(got, golden, dtype: str = "float16"):
    """一行式汇总：绝对门、相对门、双门，以及被掩盖项数量。"""
    tol = OFFICIAL_TOL[dtype]
    f = abs_gate_scan(got, golden, tol, tol)
    n = len(f)
    n_abs_fail = sum(1 for x in f if not x.abs_gate_pass)
    n_rel_fail = sum(1 for x in f if not x.rel_gate_pass)
    n_hidden = sum(1 for x in f if x.hidden_by_relative_only)
    max_abs = max((x.abs_err for x in f), default=0.0)
    rels = [x.rel_err for x in f if x.rel_err is not None]
    max_rel = max(rels) if rels else None
    return {
        "n": n,
        "tol": tol,
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "abs_gate_failures": n_abs_fail,
        "rel_gate_failures": n_rel_fail,
        "hidden_by_relative_only": n_hidden,
        "verdict_abs": "PASS" if n_abs_fail == 0 else "FAIL",
        "verdict_rel": "PASS" if n_rel_fail == 0 else "FAIL",
        "verdict_both": "PASS" if (n_abs_fail == 0 and n_rel_fail == 0) else "FAIL",
    }


# --------------------------------------------------------------------------
# 高鉴别力边界用例（补足常规随机用例覆盖不到的两类错误）
# --------------------------------------------------------------------------

def case_all_negative(B: int = 2, M: int = 16, K: int = 32, N: int = 16,
                      seed: int = 0):
    """全负相似度：专门抓「MaxSim 初值设为 0」。

    构造：让 x2 为非正，x1 为负，使点积大概率为负。
    期望：输出必须是负的最大值之和，绝不能出现 0。
    """
    rng = np.random.default_rng(seed)
    x1 = -np.abs(rng.standard_normal((B, M, K))).astype(np.float32)
    x2 = -np.abs(rng.standard_normal((B, K, N))).astype(np.float32)
    return x1, x2, "all_negative（期望全负，出现 0 即初值错误）"


def case_constructed_zero_max(K: int = 32, seed: int = 0):
    """构造「若 max 初值为 0 则必然输出 0」的强用例。

    x1 全负、x2 全正 → 点积全负 → 正确输出为负；错误实现输出 0。
    这是比随机全负更**确定**的鉴别器（随机全负可能偶然出现正值点积）。
    """
    rng = np.random.default_rng(seed)
    x1 = -np.abs(rng.standard_normal((1, 4, K))).astype(np.float32) - 0.5
    x2 = np.abs(rng.standard_normal((1, K, 4))).astype(np.float32) + 0.5
    return x1, x2, "构造性全负（点积必然全负；错实现必输出 0）"


def case_tiny_reduce_axis(axis: str, size: int, seed: int = 0):
    """小归约轴：专门抓「N<8 或 M<8 未特殊处理」。

    axis='N' → 沿 N 取 max 的轴很小；axis='M' → 沿 M 求和的轴很小。
    """
    rng = np.random.default_rng(seed)
    B, M, K, N = 2, 32, 32, 32
    if axis == "N":
        N = size
    else:
        M = size
    x1 = rng.standard_normal((B, M, K)).astype(np.float32)
    x2 = rng.standard_normal((B, K, N)).astype(np.float32)
    return x1, x2, f"小{axis}轴={size}（<8 时 Reduce 行为需特殊处理）"


def high_discrimination_suite():
    """返回一组「高鉴别力」用例，用于补足常规随机用例的盲区。"""
    suite = []
    for s in range(3):
        suite.append((f"all_negative_seed{s}", *case_all_negative(seed=s)))
    suite.append(("constructed_zero_max", *case_constructed_zero_max()))
    for n in (1, 2, 3, 4, 5, 6, 7):
        suite.append((f"tiny_N_{n}", *case_tiny_reduce_axis("N", n)))
    for m in (1, 2, 3, 4, 5, 6, 7):
        suite.append((f"tiny_M_{m}", *case_tiny_reduce_axis("M", m)))
    return suite


if __name__ == "__main__":
    print("本模块为库；诊断示例见同目录 analyze_gaps.py")
