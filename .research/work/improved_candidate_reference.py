"""candidate 算子的改进版 —— 保留原件，本文件为替代实现.

对应缺陷与改进（行号指 original-candidate/.../reference.py 原件）：

  [B1] L11-13 `_to_logical` 在 transpose 路径做 `.permute(0,2,1).contiguous()`
       → **实体化一份完整转置拷贝**。题面 3.5 明确 transpose 只声明 storage shape，
         "不表示算子需要额外执行 transpose 操作"。
       改进：改为视图（`.transpose(1,2)`），不拷贝；只有取 tile 时才按需读取。

  [B2] L33 `torch.bmm(x1, x2)` 实体化整个 [B,M,N] 相似度矩阵
       → 合法最大 (64,8192,8192) 需 64*8192*8192*4 = 16 GiB（fp32），
         题面 2^26 输入约束下这不可接受；而 M/N 维上的 max/sum 完全不需要它。
       改进：沿 M 分块，每块只算 [m_tile, N] 的相似度，算完立刻归约 N，再累加 M。
         峰值内存降到 m_tile*N。

  [B3] L20 `out_dtype=torch.float32` 参数
       → 允许改变输出 dtype，与题面 3.6「输出 y 的数据类型固定为 FLOAT32」冲突。
       改进：删除该参数。

  [B4] L64-68 `tolerance(dtype)` 以 **out_dtype** 选容差
       → 输出恒为 fp32，于是永远返回 (1e-4, 1e-4)；
         而题面第五节的容差是按**输入 dtype** 定的（fp16/bf16 → 1e-3）。
       改进：容差由输入 dtype 决定。

  [B5] L66-73 `check_close` 用 `atol + rtol*|golden|` 单一混合判据
       → 题面要求的是「相对误差 < t **且** 绝对误差 < t」双门。
         混合式在 golden 很大时会被 rtol 项主导，绝对误差越界也不报。
       改进：分别计算并同时判定，且返回诊断数值。

  [B6] 原件没有对超大 shape 的保护，`torch.bmm` 会直接 OOM 或长时间占用。
       改进：M 向分块 + 可选 work 上限。

依赖：torch（与原算子一致）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# --------------------------------------------------------------------------
# 容差：按**输入 dtype** 取，对齐题面第五节
# --------------------------------------------------------------------------
def tolerance_for_input(dtype: torch.dtype) -> tuple[float, float]:
    """返回 (rtol, atol)。题面：fp32 → 1e-4；fp16/bf16 → 1e-3。"""
    if dtype == torch.float32:
        return 1e-4, 1e-4
    return 1e-3, 1e-3


@dataclass(frozen=True)
class ErrorReport:
    max_abs: float
    max_rel: float | None
    n_abs_fail: int
    n_rel_fail: int
    n_hidden_by_relative_only: int

    @property
    def ok(self) -> bool:
        return self.n_abs_fail == 0 and self.n_rel_fail == 0


def error_report(got: torch.Tensor, golden: torch.Tensor,
                 rtol: float, atol: float) -> ErrorReport:
    """双门诊断：分别判相对与绝对，并标出「相对过、绝对挂」的项。

    [B5] 的改进：不使用 atol + rtol*|g| 混合式，避免 rtol 项掩盖绝对误差越界。
    """
    g = got.detach().to(torch.float64)
    r = golden.detach().to(torch.float64)
    if g.shape != r.shape:
        raise ValueError(f"shape mismatch: {tuple(g.shape)} vs {tuple(r.shape)}")
    e = (g - r).abs()
    nz = r != 0
    rel = torch.where(nz, e / r.abs(), torch.full_like(e, float("nan")))
    max_abs = float(e.max()) if e.numel() else 0.0
    max_rel = float(rel[~torch.isnan(rel)].max()) if bool((~torch.isnan(rel)).any()) else None
    abs_fail = e >= atol                      # 题面为严格 "<"，故 >= 即失败
    rel_fail = (~torch.isnan(rel)) & (rel >= rtol)
    hidden = abs_fail & (~rel_fail)
    return ErrorReport(max_abs, max_rel, int(abs_fail.sum()), int(rel_fail.sum()),
                       int(hidden.sum()))


# --------------------------------------------------------------------------
# 算子本体
# --------------------------------------------------------------------------
def batch_matmul_max_sum(
    x1_storage: torch.Tensor,
    x2_storage: torch.Tensor,
    transpose_x1: bool = False,
    transpose_x2: bool = False,
    *,
    m_tile: int = 128,
    k_tile: int = 512,
    work_limit_macs: int | None = None,
) -> torch.Tensor:
    """改进版算子：视图化输入 + M/K 双向分块 + K 完整累加后再沿 N 取 max + 沿 M 求和。

    与原件的行为差异（全部为修正，不是特性）：
      - 输出恒为 float32，无 out_dtype（[B3]）
      - transpose 走视图，不实体化转置拷贝（[B1]）
      - 不物化 [B,M,N]（[B2]）；也不一次物化 [K,N] 的 fp32 副本
      - K 完整累加后才进入 max，N 尾部 padding 不参与

    分块后峰值 ≈ m_tile*N*4 + k_tile*N*4 + m_tile*k_tile*4。
    """
    if not isinstance(x1_storage, torch.Tensor) or not isinstance(x2_storage, torch.Tensor):
        raise TypeError("inputs must be torch.Tensor")
    if x1_storage.ndim != 3 or x2_storage.ndim != 3:
        raise ValueError("inputs must be 3-D storage tensors")
    if x1_storage.dtype != x2_storage.dtype:
        raise ValueError("inputs must share dtype")
    if x1_storage.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("inputs must be float16/bfloat16/float32")
    if type(transpose_x1) is not bool or type(transpose_x2) is not bool:
        raise ValueError("transpose flags must be bool")
    if m_tile <= 0 or k_tile <= 0:
        raise ValueError("m_tile and k_tile must be positive")

    # [B1] 视图，不 contiguous()
    #   transpose_x1=true: storage (B,K,M) -> 逻辑 (B,M,K)
    #   transpose_x2=true: storage (B,N,K) -> 逻辑 (B,K,N)
    x1 = x1_storage.transpose(1, 2) if transpose_x1 else x1_storage
    x2 = x2_storage.transpose(1, 2) if transpose_x2 else x2_storage

    B, M, K = x1.shape
    B2, K2, N = x2.shape
    if B != B2 or K != K2:
        raise ValueError(f"Batch/K mismatch: x1={tuple(x1.shape)} x2={tuple(x2.shape)}; "
                         "broadcasting is forbidden")

    if work_limit_macs is not None and B * M * N * K > work_limit_macs:
        raise RuntimeError(
            f"work budget exceeded: {B*M*N*K:,} macs > {work_limit_macs:,}"
        )

    acc = torch.float32
    out = torch.empty((B,), dtype=torch.float32)
    step_m = min(m_tile, M)
    step_k = min(k_tile, K)

    for b in range(B):
        total = 0.0
        for m0 in range(0, M, step_m):
            mr = min(step_m, M - m0)
            # 跨 K 分块累加：dots 是 [mr, N] 的 fp32 累加器
            dots = torch.zeros((mr, N), dtype=acc)
            for k0 in range(0, K, step_k):
                kr = min(step_k, K - k0)
                a_blk = x1[b, m0:m0 + mr, k0:k0 + kr].to(acc)   # [mr, kr]
                c_blk = x2[b, k0:k0 + kr, :].to(acc)            # [kr, N]
                dots += a_blk @ c_blk
            # K 已完整累加，才进入沿 N 的 max（题面第 4 节：顺序不可交换）
            row_max = torch.amax(dots, dim=1)                   # [mr]
            total += float(row_max.sum(dtype=torch.float64))
        out[b] = total
    return out


def golden_fp64(
    x1_storage: torch.Tensor,
    x2_storage: torch.Tensor,
    transpose_x1: bool = False,
    transpose_x2: bool = False,
    *,
    m_tile: int = 128,
    k_tile: int = 512,
) -> torch.Tensor:
    """FP64 golden：同样视图化 + M/K 分块，不物化 [B,M,N] 或 [K,N] 的 fp64 副本。"""
    x1 = x1_storage.transpose(1, 2) if transpose_x1 else x1_storage
    x2 = x2_storage.transpose(1, 2) if transpose_x2 else x2_storage
    B, M, K = x1.shape
    N = x2.shape[2]
    out = torch.empty((B,), dtype=torch.float32)
    step_m = min(m_tile, M)
    step_k = min(k_tile, K)
    for b in range(B):
        acc_total = 0.0
        for m0 in range(0, M, step_m):
            mr = min(step_m, M - m0)
            dots = torch.zeros((mr, N), dtype=torch.float64)
            for k0 in range(0, K, step_k):
                kr = min(step_k, K - k0)
                a_blk = x1[b, m0:m0 + mr, k0:k0 + kr].to(torch.float64)
                c_blk = x2[b, k0:k0 + kr, :].to(torch.float64)
                dots += a_blk @ c_blk
            acc_total += float(torch.amax(dots, dim=1).sum())
        out[b] = acc_total
    return out


if __name__ == "__main__":
    # 最小自检：题面三个示例
    y1 = batch_matmul_max_sum(torch.tensor([[[1., 0.], [0., 1.]]]),
                              torch.tensor([[[1., 0., -1.], [0., 1., 0.]]]))
    assert torch.allclose(y1, torch.tensor([2.0])), y1

    x1l = torch.tensor([[[1., 0.], [0., 1.]]])
    x2l = torch.tensor([[[1., 0., -1.], [0., 1., 0.]]])
    y2 = batch_matmul_max_sum(x1l.transpose(1, 2).contiguous(),
                              x2l.transpose(1, 2).contiguous(), True, True)
    assert torch.allclose(y2, torch.tensor([2.0])), y2

    y3 = batch_matmul_max_sum(torch.tensor([[[1., 0.]]]),
                              torch.tensor([[[-1., -2.], [0., 0.]]]))
    assert torch.allclose(y3, torch.tensor([-1.0])), y3
    print("improved operator: 官方三示例 PASS")
