#!/usr/bin/env python3
"""api_gate.py — 提交前的「API 风险分级」门禁

背景：本文件第一次提交被真机打回 5 条编译错误（`DataCopyPadExtParams` 的字段名
与 CANN 8.5 文档不一致；`!ASCEND_IS_AIV` 非法）。教训：**本地文档 ≠ 真机头文件**，
所以每次提交前要能一眼看出"这一版引入了哪些真机没验证过的 API"。

分级依据（全部来自真机证据，不是猜）：
  PROVEN       : 队友 15/15 那份文件里用到的 API —— 真机编译+运行都过了
  TYPECHECKED  : 在我那版被真机编译时**没有被报错**的新 API。证据：clang 默认错误上限 20，
                 那次只报 5 条（未截断），且报错行(451/452 → 648 → 686/687)把这些调用
                 夹在中间 —— 说明它们通过了类型检查。仍需运行期证明。
  UNKNOWN      : 其余新 API → 高风险，应单独提交、不要和其它改动混在一起

用法：
  python api_gate.py [kernel.asc]
  python api_gate.py --variants
"""
from __future__ import annotations

import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
PROVEN_FILE = (HERE.parent / ".research/repos/CANNCompetition/projects/"
               "batchmatmul-maxsum/npu-v1/project/kernel.asc")

# 真机编译未被报错的新 API（证据见模块 docstring）
TYPECHECKED = {
    "IterateAll", "WholeReduceMax", "ReduceOrder", "ORDER_ONLY_VALUE",
    "MTE2_S", "MTE3_V", "S_MTE2", "V_MTE2", "V_MTE3",
    "SetAtomicAdd", "SetAtomicNone",
}

# 同族已证、仅"用法形式"未证：只告警不拦
LOWRISK = {"ASCEND_IS_AIC", "ASCEND_IS_AIV"}

_SKIP = {
    "GM_ADDR", "TPipe", "TBuf", "LocalTensor", "GlobalTensor", "Matmul",
    "MatmulType", "CubeTiling", "Shape", "TensorInfo", "TensorGroupInfo",
    "DataCopyExtParams", "DataCopyPadExtParams", "TPosition", "CubeFormat",
    "HardEvent", "DataType", "AscendC", "MatmulApiTiling", "PlatformAscendC",
    "PlatformAscendCManager", "TCubeTiling", "Tensor", "Min", "Max", "M", "N",
    "K", "B", "A", "T", "V", "E", "P", "GM", "UB", "KB", "MB", "IL", "OK",
    "IF", "NOT", "AND", "OR", "ASCEND", "CANN", "BMM", "AIV", "AIC", "MIX",
    "CPU", "NPU", "DMA", "HBM", "TLB", "KFC", "S5", "IT", "IS", "TO", "IN",
    "ON", "AT", "OF", "AS", "AN", "BY", "SO", "UP", "NO", "ID", "M1", "M2",
}


def _self_defined(t: str) -> set[str]:
    """文件里**自己定义**的符号（函数/结构体/类型/宏/常量），不算"新引入的外部 API"。"""
    pats = [r"__global__\s+__aicore__\s+void\s+(\w+)",
            r"(?:inline|void|uint32_t|int32_t|bool|float|uint64_t)\s+(\w+)\s*\(",
            r"(?:struct|class|enum)\s+(\w+)",
            r"using\s+(\w+)\s*=",
            r"static\s+constexpr\s+\w+\s+(\w+)\s*=",
            r"#define\s+(\w+)"]
    out: set[str] = set()
    for pat in pats:
        out |= set(re.findall(pat, t))
    return out


def _prune_false_branches(t: str) -> str:
    """只处理本文件自己的 `#if BMM_TASK_RATIO == N`：按 #define 的值裁掉假分支，
    否则文本分析会把 r1 专属代码算进所有变体里。"""
    m = re.search(r"#define\s+BMM_TASK_RATIO\s+(\d+)", t)
    val = m.group(1) if m else None
    out, skip = [], 0
    for line in t.splitlines():
        if re.match(r"\s*#if\s+BMM_TASK_RATIO\s*==\s*(\d+)", line):
            want = re.match(r"\s*#if\s+BMM_TASK_RATIO\s*==\s*(\d+)", line).group(1)
            skip += 1 if (val is not None and want != val) else 0
            out.append("")
            continue
        if re.match(r"\s*#endif", line) and skip > 0:
            skip -= 1
            out.append("")
            continue
        out.append("" if skip else line)
    return "\n".join(out)


def used_apis(path: pathlib.Path) -> set[str]:
    t = path.read_text(encoding="utf-8", errors="replace")
    t = _prune_false_branches(t)
    t = re.sub(r"/\*.*?\*/", "", t, flags=re.S)
    t = re.sub(r"//[^\n]*", "", t)
    names = set(re.findall(r"\b([A-Z][A-Za-z0-9_]{2,})\b", t))
    noise = {"NULL", "UINT32_MAX", "FLT_MAX", "NAN", "INFINITY", "NULLPTR"}
    return {n for n in names
            if n not in _SKIP and n not in noise and n not in _self_defined(t)}


def classify(path: pathlib.Path) -> tuple[set, set, set, set]:
    proven = used_apis(PROVEN_FILE)
    mine = used_apis(path)
    delta = mine - proven
    tc = {d for d in delta if d in TYPECHECKED}
    low = {d for d in delta if d in LOWRISK}
    unknown = delta - tc - low
    return tc, low, unknown, proven


def report(label: str, path: pathlib.Path) -> int:
    tc, low, unknown, proven = classify(path)
    print(f"[{label}] 新增(非 15/15 已验证) API：")
    print(f"    TYPECHECKED(真机类型检查通过) : "
          f"{sorted(tc) if tc else '（无）'}")
    print(f"    LOWRISK(同族已证、形式未证)   : "
          f"{sorted(low) if low else '（无）'}")
    print(f"    UNKNOWN(未经验证，高风险)     : "
          f"{sorted(unknown) if unknown else '（无）'}")
    if unknown:
        print("    ⚠ 建议：把 UNKNOWN 单独提交一次，确认编译通过后再叠加其它改动")
    return 1 if unknown else 0


def main() -> int:
    if "--variants" in sys.argv:
        sys.path.insert(0, str(HERE))
        import submit_sweep as ss
        worst = 0
        for name, ov in ss.VARIANTS.items():
            src = (ss.patch_kernel(ov)[0] if ov
                   else (HERE / "kernel.asc").read_text(encoding="utf-8"))
            tmp = pathlib.Path(HERE / f".api_gate_{name}.asc")
            tmp.write_text(src, encoding="utf-8")
            worst |= report(name, tmp)
            tmp.unlink(missing_ok=True)
        return worst
    p = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "kernel.asc"
    return report(p.name, p)


if __name__ == "__main__":
    raise SystemExit(main())
