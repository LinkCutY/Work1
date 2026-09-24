#!/usr/bin/env python3
"""sync_submit.py — 提交前把"真源" kernel.asc 同步到所有会被提交的副本，并核对哈希

为什么需要它：判题用的是某个**副本**（判题项目目录 / 本工具的 submit 目录），
不是你看的那个文件。曾经连续两轮报同一个错（Expected 75, got 110），
原因就是副本还是旧的（副本 BMM_SINGLE_LAUNCH=0，真源已经是 1）。

用法：
  python sync_submit.py            # 只核对，不一致就退出码 1
  python sync_submit.py --write    # 把真源覆盖到所有副本
"""
from __future__ import annotations

import hashlib
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "kernel.asc"
COPIES = [
    HERE / "submit/project/kernel.asc",
    HERE.parent / ".research/repos/CANNCompetition/projects/batchmatmul-maxsum/npu-v1/project/kernel.asc",
]


def digest(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:12]


def main() -> int:
    write = "--write" in sys.argv
    src = digest(SRC)
    print(f"真源 {SRC}  {src}")
    bad = 0
    for c in COPIES:
        if not c.exists():
            print(f"  缺失   {c}")
            bad += 1
            continue
        h = digest(c)
        same = h == src
        print(f"  {'一致' if same else '不一致'} {h}  {c}")
        if not same:
            bad += 1
            if write:
                shutil.copyfile(SRC, c)
                print(f"        -> 已覆盖为 {digest(c)}")
    if bad and not write:
        print("有副本与真源不一致：改完务必 `python sync_submit.py --write` 再提交。")
        return 1
    print("全部一致。" if not bad else "已全部覆盖为真源。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
