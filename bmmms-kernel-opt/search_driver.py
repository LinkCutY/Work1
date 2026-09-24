#!/usr/bin/env python3
"""search_driver.py — 按实测用时驱动"下一个该提交哪个变体"的搜索逻辑（可离线自测）

用户目标：针对比例/参数轴做"三分查找"，每次按实测用时收敛到最优解。
本文件把**决策逻辑**与**提交动作**分开：
  * decide(axis, results) 是纯函数：输入已累积的 results（submit_sweep 的 jsonl
    记录），输出下一批该提交的变体名。不碰网络、不碰凭据 → 可离线用合成数据验证。
  * --run 才会真的通过 submit_sweep 提交。

轴的定义（都必须是**有序**的，三分查找才成立；若实测曲线非单峰，脚本会退化为
全枚举——5 个候选点最多 5 次提交，代价可控）：
  ratio : AIV:AIC = 1:2 → 1:1（2 个取值，退化成一次比较）
  tile  : L0C 占用从小到大的 (baseM, baseN) 序列：(64,128) (96,192) (128,128)
          (160,192) (192,160) —— L0C 分别 32/72/64/120/120 KB（<128KB 合法）
  gran  : BMM_GROUPS_MULT = 1 → 2 → 4（任务粒度，削末波次空转）

目标函数：先比"通过数"（Pass 数），再比"总用时"（15 个 case 用时之和）。
理由：判分是每个 case 单独算分再取均值，任何 case 掉精度都会拖垮均分，因此
"全过 + 总时间最短"比"个别 case 更快"重要。

用法：
  python search_driver.py --selftest          # 用合成用时验证搜索逻辑（离线）
  python search_driver.py --axis tile --plan  # 只打印下一步该提交哪些
  python search_driver.py --axis tile --run   # 真的提交（需要有效 session）
"""
from __future__ import annotations

import argparse
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import submit_sweep as ss  # noqa: E402

# 轴 → 有序候选 [(变体名, overrides)]
AXES: dict[str, list[tuple[str, list[tuple[str, str]]]]] = {
    "ratio": [
        ("base", []),
        ("r1", [("BMM_TASK_RATIO", "1")]),
    ],
    "tile": [
        ("bm64_bn128", [("BASE_M", "64")]),               # BASE_N 保持 128
        ("bm96_bn192", [("BASE_M", "96"), ("BASE_N", "192")]),
        ("base", []),                                   # 128/128
        ("bm160", [("BASE_M", "160"), ("BASE_N", "192")]),
        ("bm192_bn160", [("BASE_M", "192"), ("BASE_N", "160")]),
    ],
    "gran": [
        ("base", []),                                   # GROUPS_MULT=1
        ("g2", [("BMM_GROUPS_MULT", "2")]),
        ("g4", [("BMM_GROUPS_MULT", "4")]),
    ],
}


def _score(rec: dict) -> tuple[int, float]:
    """越大越好：先通过数，再(负)总用时。"""
    total = rec.get("total_time")
    return (int(rec.get("pass_n") or 0), -float(total if total is not None else 1e18))


def decide(axis: str, results: list[dict]) -> list[tuple[str, list[tuple[str, str]]]]:
    """纯函数：给定已测结果，返回下一批要提交的 (变体名, overrides)。

    已测过的变体不会重复提交（避免浪费额度）；全部测完则返回空列表。
    """
    cands = AXES[axis]
    names = [n for n, _ in cands]
    by_name = {r["variant"]: r for r in results if r.get("variant") in names}
    measured = [n for n in names if n in by_name]

    # 未测：先补齐端点和中点（三分查找的第一次探针位置）
    untested = [n for n in names if n not in by_name]
    if len(measured) < 3 or names[0] not in by_name or names[-1] not in by_name:
        todo = []
        for n in (names[0], names[len(names) // 2], names[-1]):
            if n in untested and n not in [t for t, _ in todo]:
                todo.append((n, dict(cands)[n]))
        return todo or [(untested[0], dict(cands)[untested[0]])]

    # 三分：用已测端点/中点把区间缩小到更优的一半，再测其内侧两个三等分点
    lo, hi = 0, len(names) - 1
    while hi - lo >= 2:
        mid = (lo + hi) // 2
        s_lo, s_mid, s_hi = (by_name.get(names[lo]), by_name.get(names[mid]),
                             by_name.get(names[hi]))
        if s_lo is None or s_mid is None or s_hi is None:
            break
        # 比较 mid 与两端：若 mid 比两端都好 → 单峰在山顶侧，缩到 [lo,mid] / [mid,hi]
        if _score(s_mid) >= _score(s_lo) and _score(s_mid) >= _score(s_hi):
            break                        # 已定位到最优点
        if _score(s_lo) > _score(s_hi):
            hi = mid
        else:
            lo = mid
    # 在 [lo,hi] 内挑未测的：先三等分点
    for frac in (1 / 3, 2 / 3):
        idx = lo + int(round((hi - lo) * frac))
        if idx > lo and idx < hi and names[idx] not in by_name:
            return [(names[idx], dict(cands)[names[idx]])]
    # 区间已很小：把剩下的都测掉（最多 2 个）
    rest = [(n, dict(cands)[n]) for n in names[lo:hi + 1] if n not in by_name]
    return rest[:2]


def best_of(axis: str, results: list[dict]) -> str | None:
    cands = {n: o for n, o in AXES[axis]}
    hits = [r for r in results if r.get("variant") in cands and
            r.get("total_time") is not None]
    if not hits:
        return None
    return max(hits, key=_score)["variant"]


def selftest() -> int:
    """用合成用时验证：单峰曲线能被三分查找收敛到最优点。"""
    print("=== 合成单峰曲线（tile 轴，真峰在 bm160）===")
    fake_total = {"bm64_bn128": 120.0, "bm96_bn192": 95.0, "base": 88.0,
                  "bm160": 61.0, "bm192_bn160": 70.0}
    results: list[dict] = []
    for step in range(1, 6):
        nxt = decide("tile", results)
        if not nxt:
            break
        for name, _ov in nxt:
            print(f"  step{step} 提交 {name}（合成用时 {fake_total[name]}）")
            results.append({"variant": name, "pass_n": 15,
                            "total_time": fake_total[name]})
    got = best_of("tile", results)
    print(f"  收敛结果: {got}（期望 bm160）  {'OK' if got == 'bm160' else 'FAIL'}")

    print("\n=== 非单峰（谷底在 base）时的行为 ===")
    fake2 = {"bm64_bn128": 200.0, "bm96_bn192": 90.0, "base": 50.0,
             "bm160": 61.0, "bm192_bn160": 300.0}
    res2: list[dict] = []
    for step in range(1, 7):
        nxt = decide("tile", res2)
        if not nxt:
            break
        for name, _ov in nxt:
            res2.append({"variant": name, "pass_n": 15, "total_time": fake2[name]})
    got2 = best_of("tile", res2)
    print(f"  收敛结果: {got2}（期望 base）"
          f"  测了点: {sorted(r['variant'] for r in res2)}")

    print("\n=== 精度优先：某个变体少过 1 个 case ===")
    res3 = [{"variant": "base", "pass_n": 15, "total_time": 88.0},
            {"variant": "bm160", "pass_n": 14, "total_time": 40.0}]
    print(f"  best_of = {best_of('tile', res3)}（期望 base：少过一个 case 更贵）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--axis", choices=list(AXES), default="ratio")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--wait", type=int, default=1800)
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    results = ss.load_results()
    nxt = decide(a.axis, results)
    print(f"轴 {a.axis}：候选 {[n for n, _ in AXES[a.axis]]}")
    print(f"已测: {sorted({r['variant'] for r in results})}")
    if not nxt:
        print(f"该轴已收敛/测完；当前最优: {best_of(a.axis, results)}")
        return 0
    print(f"下一步提交: {[n for n, _ in nxt]}")
    if not a.run:
        print("（--plan 模式，未提交；加 --run 真提交）")
        return 0
    for name, ov in nxt:
        try:
            ss.submit_one(name, ov, a.wait)
        except Exception as e:
            print(f"[{name}] 失败: {type(e).__name__}: {e}")
    print(f"更新后最优: {best_of(a.axis, ss.load_results())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
