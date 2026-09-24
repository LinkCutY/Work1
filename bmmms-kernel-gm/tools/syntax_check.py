#!/usr/bin/env python3
"""syntax_check.py — 用本机 g++ 对 kernel.asc（或其变体）做语法/语义检查

为什么值得做：首次提交最贵的失败是"编译不过"。真机工具链本机没有，但 kernel.asc
的**自身语法**（漏分号、错字、实参个数/类型、重载挑选、结构体字段名）用任何 C++14
编译器都能查。stubs/ 里的头文件按 CANN 8.5 文档抄签名，因此"字段名写错/重载挑错"
会被抓出来；它**不能**证明真机可编译（NPU_ARCH、真实头文件、工具链特性都不覆盖）。

可被安全 import：`_check_one(label, text) -> (ok, messages)` 是纯函数。
脚本式用法：
  python syntax_check.py [kernel.asc]
  python syntax_check.py --variants
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent

# g++ 对 stub 头 / 框架 ABI 参数产生的噪声（非我方代码问题）
_NOISE = ("stubs", "In instantiation", "required from",
          "unused parameter 'systemWorkspace'", "__kfc_workspace__ GM_ADDR systemWorkspace,")


def _normalise(src: str) -> str:
    """Kernel<<<dim, l2, stream>>>(args) 是 AscendC 扩展语法，g++ 不认 → 去掉启动参数。"""
    return re.sub(r"<<<[^>]*>>>", "", src)


def _check_one(label: str, text: str) -> tuple[bool, list[str]]:
    out = pathlib.Path(tempfile.mkdtemp(prefix="kcheck_")) / "kernel_check.cpp"
    out.write_text(_normalise(text), encoding="utf-8")
    cmd = ["g++", "-std=c++14", "-x", "c++", "-fsyntax-only", "-Wall", "-Wextra",
           "-I", str(HERE / "stubs"), "-include", str(HERE / "stubs/contest_types.h"),
           str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    body = (r.stdout + r.stderr).replace(str(out), "kernel.asc")
    msgs = [l for l in body.splitlines()
            if l.strip() and not any(n in l for n in _NOISE) and "~~~~" not in l]
    return r.returncode == 0, msgs


def check_variants() -> int:
    sys.path.insert(0, str(HERE))
    import submit_sweep as ss
    bad = 0
    items = list(ss.VARIANTS.items())
    for name, ov in items:
        try:
            src, _ = ss.patch_kernel(ov) if ov else (
                (HERE / "kernel.asc").read_text(encoding="utf-8"), [])
        except SystemExit as e:
            print(f"  {name:12s} 生成失败: {e}")
            bad += 1
            continue
        ok, msg = _check_one(name, src)
        bad += 0 if ok else 1
        print(f"  {name:12s} {'OK' if ok else 'FAIL'}")
        for l in msg[:6]:
            print("      " + l)
    print(f"变体语法检查：{len(items) - bad}/{len(items)} 通过")
    return 0 if bad == 0 else 1


def main() -> int:
    if "--variants" in sys.argv:
        return check_variants()
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "kernel.asc"
    ok, msgs = _check_one(path.name, path.read_text(encoding="utf-8"))
    print(f"g++ 语法检查 {path.name}（stub 签名版）: {'通过' if ok else '有错误'}")
    for l in msgs[:80]:
        print("  " + l)
    if ok and not msgs:
        print("  无 warning/error")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
