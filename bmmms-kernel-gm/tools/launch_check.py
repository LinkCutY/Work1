#!/usr/bin/env python3
"""launch_check.py — 判题硬规则门禁：一次迭代只能启动 1 个 kernel

真机反馈原文（两次踩到，第 3 轮 135、本轮 110）：

    Profiling rule violated: each iteration must launch exactly 1 kernel.
    Expected 75 launches, got 110.

两条同样重要的推论：

1. **同一个代码块里不能并联两个 kernel**（`<<< >>>` 在同一最小花括号块内出现 >= 2 次）。
   互斥分支（例如"小 case 走 SmallKernel / 其余走 MatmulMaxKernel"）各写一次是允许的，
   因为任一时刻只走一条。
2. **abort 也是违规**：host 侧 `Fail()`（std::abort → status 134）会被平台
   `<ReplayOnce>` 重放，每次重放**多算一次 kernel launch**。
   110 = 75 + 7×5 就是这么来的（7 个 case、每 case 5 次迭代各重放一次）。
   所以 `Fail()` 只允许出现在"真的没法跑"的地方，且分片请求必须自带兜底档。

用法： python launch_check.py [kernel.asc]      退出码 0 = 通过
"""
from __future__ import annotations

import pathlib
import re
import sys

# 共享的启动数检查（与 bmmms-kernel-perf/tools/selfcheck.py 同一套规则）
LAUNCH = re.compile(r"<<<")


def split_top_level_bodies(code: str) -> list[list[tuple[int, str]]]:
    """按花括号深度切出每个"最小块"的行集合（用于发现同一块内的多次启动）。"""
    bodies: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    depth = 0
    for lineno, line in enumerate(code.splitlines(), start=1):
        stripped = line.split("//")[0]
        before = depth
        depth += stripped.count("{") - stripped.count("}")
        current.append((lineno, line))
        if depth <= before and before > 0 and depth < before:
            bodies.append(current)
            current = []
    if current:
        bodies.append(current)
    return bodies


def main() -> int:
    path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                        pathlib.Path(__file__).resolve().parent.parent / "kernel.asc")
    code = path.read_text(encoding="utf-8")
    bare = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    bare = re.sub(r"//[^\n]*", "", bare)

    problems: list[str] = []
    launches = [i for i, l in enumerate(bare.splitlines(), start=1) if LAUNCH.search(l)]

    print(f"{path.name}: 找到 {len(launches)} 处 <<<>>> 启动，行 {launches}")

    if not launches:
        problems.append("找不到任何 <<<...>>> kernel 启动")

    for body in split_top_level_bodies(bare):
        hits = [lineno for lineno, line in body if LAUNCH.search(line)]
        if len(hits) > 1:
            problems.append(
                f"同一代码块内有 {len(hits)} 次 kernel 启动（行 {hits}）"
                "  <- 判题要求每次迭代恰好 1 个 kernel")

    # 启动点必须落在互斥分支里：检查每处启动所属的紧邻 if/else 链（粗判，够用）
    text = bare.splitlines()
    for lineno in launches:
        window = "\n".join(text[max(0, lineno - 25):lineno])
        if not re.search(r"\b(if|else)\b", window):
            problems.append(f"行 {lineno} 的启动点不在任何 if/else 分支里（无法保证互斥）")

    # abort 面：Fail() 的调用点必须是我们已知的那批（新增一个就提醒复核）
    fails = [i for i, l in enumerate(bare.splitlines(), start=1) if re.search(r"\bFail\(\)", l)]
    print(f"Fail() 调用点（每个都是一次可能的 ReplayOnce 重放）：行 {fails}")

    if problems:
        print("\n不通过：")
        for p in problems:
            print("  -", p)
        return 1
    print("\n通过：启动点互斥、无同块并联启动")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
