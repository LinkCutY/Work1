#!/usr/bin/env python3
"""api_delta.py — 新 kernel.asc 相对**真机 15/15 基线**多用了哪些 API

本机做不了真机编译，所以"有没有引入未验证的 API"只能这样回答：
把两份源码里出现的调用名 / 限定名集合做差，差集就是"基线没用过、因此没有真机
证据"的那部分。差集不是错误，但它必须逐条有出处、并在上板清单里被点名。

用法： python api_delta.py [baseline.asc] [candidate.asc]
退出码恒为 0（它只报告，不判定）。
"""
from __future__ import annotations

import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent

# 语言关键字 / 局部定义 / 标准库：不算"外部 API"
NOISE = {
    "if", "while", "for", "return", "sizeof", "static_cast", "reinterpret_cast",
    "const_cast", "alignof", "decltype", "true", "false", "switch", "catch",
    "std", "min", "max", "abort", "AscendC",
    # 本文件自定义
    "MinU32", "AbsF32", "Decode", "AddCompensated", "Fence", "Fail", "CheckAcl",
    "CheckTiling", "SingleTensor", "ParseShape", "LaunchMatmulPath", "Launch",
    "SmallKernel", "MatmulMaxKernel", "Shape", "run_kernel",
}

CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
QUAL = re.compile(r"\b([A-Za-z_]\w*(?:::[A-Za-z_]\w*)+)")


def tokens(path: pathlib.Path) -> set[str]:
    t = path.read_text(encoding="utf-8", errors="replace")
    t = re.sub(r"/\*.*?\*/", " ", t, flags=re.S)   # 块注释
    t = re.sub(r"//[^\n]*", " ", t)                # 行注释
    out = {m.group(1) for m in CALL.finditer(t)}
    out |= {m.group(1) for m in QUAL.finditer(t)}
    return {x for x in out if x.split("::")[-1] not in NOISE and x not in NOISE}


def main() -> int:
    base = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else HERE.parent / "teammate-latest.asc")
    cand = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else HERE.parent / "kernel.asc")
    b, c = tokens(base), tokens(cand)
    print(f"baseline  : {base.name}  ({len(b)} 个外部 API 名)")
    print(f"candidate : {cand.name}  ({len(c)} 个外部 API 名)")
    print("\n-- 候选**多出**的（基线无真机证据 → 上板必须点名）--")
    for x in sorted(c - b):
        print("  +", x)
    print("\n-- 候选**少用**的（基线有、本版没有）--")
    for x in sorted(b - c):
        print("  -", x)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
