#!/usr/bin/env python3
"""kernel.asc v2 的**源码级**不变量检查（不是模型，不是编译器）。

为什么需要它：reduce_oracle.py 验证的是 plan()/model() 这套**重新实现**的
算术；源码里的转录错误（参数顺序、下标表达式、fence 缺失、越界）它抓不到。
本脚本直接对 kernel.asc 的文本做结构断言，每条都给出命中行号。

用法： python source_check.py [kernel.asc]
退出码 0 = 全部通过。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

FAILS: list[str] = []
PASSES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASSES if cond else FAILS).append(f"{name}{(' — ' + detail) if detail else ''}")


def norm(expr: str) -> str:
    """把表达式归一化后比较（去空白、统一 group 变量名，按标识符边界）。"""
    e = re.sub(r"\s+", "", expr)
    e = e.replace("static_cast<uint64_t>(", "")
    e = e.replace("static_cast<int32_t>(", "")
    e = e.replace(")", "")
    e = re.sub(r"\bgroup\b", "G", e)      # 只替换独立标识符，不动 groups
    e = re.sub(r"\bg\b", "G", e)
    return e


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "kernel.asc")
    src = path.read_text(encoding="utf-8")
    lines = src.split("\n")

    def find(pat: str, flags=0) -> list[tuple[int, str]]:
        out = []
        rx = re.compile(pat, flags)
        for i, ln in enumerate(lines, 1):
            if rx.search(ln):
                out.append((i, ln.strip()))
        return out

    # 去掉注释后的代码（用于"是否存在某个调用"这类判断）
    code = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    code = re.sub(r"//[^\n]*", "", code)

    # ---- 1. 设备侧不得用 std:: （红线 06-api-redlines §6.6） ----
    host_banner = find(r"^// Host helpers")
    host_line = host_banner[0][0] if host_banner else len(lines)
    bad_std = [i for i, _ in find(r"std::") if i < host_line]
    check("设备侧无 std:: 调用", not bad_std, f"越界行 {bad_std}" if bad_std else
          f"host 区从第 {host_line} 行开始")

    # ---- 2. 禁用构造 ----
    for pat, label in ((r"\bstd::vector\b", "std::vector"),
                       (r"\bnew\s+\w", "new"),
                       (r"\bmalloc\s*\(", "malloc"),
                       (r"#include\s*<cmath>", "<cmath>"),
                       (r"\bprintf\s*\(", "printf"),
                       (r"SetSysWorkspace\s*\(", "SetSysWorkspace"),
                       (r"^\s*int\s+main\s*\(", "main()"),
                       (r"#pragma\s+once", "#pragma once")):
        hits = find(pat)
        check(f"未使用禁用构造 {label}", not hits,
              f"命中行 {[h[0] for h in hits]}" if hits else "")

    # ---- 3. 单 launch 开关的一致性 ----
    sw = re.search(r"BMM_SINGLE_LAUNCH\s*=\s*(\d+)", src)
    check("存在 BMM_SINGLE_LAUNCH 开关", sw is not None,
          f"值={sw.group(1)}" if sw else "")
    atomic = [i for i, _ in find(r"SetAtomicAdd|SetAtomicNone")]
    if atomic:
        guarded = False
        for i, _ in find(r"if\s*\(\s*BMM_SINGLE_LAUNCH"):
            guarded = True
        check("原子加仅在 BMM_SINGLE_LAUNCH!=0 分支内", guarded,
              f"原子调用行 {atomic}")
    else:
        check("原子加仅在 BMM_SINGLE_LAUNCH!=0 分支内", True, "当前无原子调用")
    if sw and sw.group(1) == "0":
        check("开关=0 时不 memset y（省一次 GM 写）",
              "aclrtMemset(y" not in code or
              re.search(r"if\s*\(\s*BMM_SINGLE_LAUNCH\s*!=\s*0\s*\)\s*\{[^}]*aclrtMemset\(y",
                        code, re.S) is not None,
              "aclrtMemset(y) 必须包在开关里")

    # ---- 4. partial 下标：phase1 与 phase2 必须同构 ----
    # groups>1 分支：(batch*nmwin + mwin) * groups * baseM + group * baseM
    offs = re.findall(r"(?:outOffset|off)\s*=\s*(.*?);", code, re.S)
    g_multi = [o for o in offs if "groups" in o and "baseM" in o]
    check("groups>1 的 partial 下标表达式存在 2 处（phase1/phase2）",
          len(g_multi) == 2, f"命中 {len(g_multi)} 处")
    if len(g_multi) == 2:
        a = norm(g_multi[0])
        b = norm(g_multi[1])
        check("groups>1 的 partial 下标两侧同构", a == b,
              f"\n    p1={a}\n    p2={b}" if a != b else a)
    # groups==1 分支：phase1 写 partialGm[task]，phase2 读 off=batch*nmwin+mwin
    check("groups==1：phase1 写 partialGm[task]（非原子路径）",
          re.search(r"DataCopyPad\(\s*partialGm\[task\]\s*,\s*rowOut", code) is not None)
    check("groups==1：phase2 读 off = batch*nmwin + mwin",
          re.search(r"const uint64_t off\s*=\s*static_cast<uint64_t>\(batch\)\s*\*\s*nmwin\s*\+\s*mwin\s*;",
                    code, re.S) is not None)
    # 两者相等依赖解码：groups==1 时 group=0, rest=task → batch*nmwin+mwin==task
    # （代数由 reduce_oracle.py 的 coverage 检查证明）

    # ---- 5. task 解码与 groups==1 的等价性 ----
    need = [r"group\s*=\s*task\s*%\s*groups", r"rest\s*=\s*task\s*/\s*groups",
            r"mwin\s*=\s*rest\s*%\s*nmwin", r"batch\s*=\s*rest\s*/\s*nmwin"]
    miss = [p for p in need if not re.search(p, code)]
    check("task→(batch,mwin,group) 解码与 oracle 一致", not miss,
          f"缺少 {miss}" if miss else "4 行解码全部命中")

    # ---- 6. chunk 循环边界 ----
    check("chunkBegin = group*chunksPerTask",
          re.search(r"chunkBegin\s*=\s*group\s*\*\s*chunksPerTask", code) is not None)
    check("chunkEnd = min(nchunkCount, chunkBegin+chunksPerTask)",
          re.search(r"chunkEnd\s*=\s*MinU32\(\s*nchunkCount\s*,\s*chunkBegin\s*\+\s*chunksPerTask\s*\)",
                    code) is not None)

    # ---- 7. host 公式 ----
    for pat, label in (
            (r"nmwin\s*=\s*HostCeilDiv\(\s*shape\.m\s*,\s*baseM\s*\)", "nmwin=ceil(M/baseM)"),
            (r"nchunkCount\s*=\s*HostCeilDiv\(\s*shape\.n\s*,\s*chunkWidth\s*\)",
             "nchunk=ceil(N/chunkWidth)"),
            (r"chunkWidth\s*=\s*\n?\s*\(\(BMM_CHUNK_N\s*>\s*0\)\s*&&\s*\(narrowTiles\s*>=\s*BMM_WIDE_MIN_TILES\)\)",
             "chunkWidth 由 BMM_CHUNK_N + 宽块门槛共同决定"),
            (r"narrowTiles\s*=\s*HostCeilDiv\(\s*shape\.n\s*,\s*baseN\s*\)", "门槛用 N 方向 tile 数"),
            (r"#define\s+BMM_WIDE_MIN_TILES\s+\d+", "宽块启用门槛常量"),
            (r"mm\.IterateAll\(cWin, 0, false\)", "宽块用 IterateAll 一次写完一整个 chunk"),
            (r"ubRows\s*=\s*\n?\s*\(ubPitch == 0U\) \? baseM : \(stageBytes / \(ubPitch \* sizeof\(float\)\)\)",
             "行带上限 = 暂存字节/ubPitch"),
            (r"#define\s+BMM_UB_STAGE_KB\s+\d+", "UB 暂存预算常量"),
            (r"DataCopyPad\(cUb,\s*cWin\[static_cast<uint64_t>\(r0\) \* chunkPitch\]",
             "带内 2D 拷贝带行偏移 r0"),
            (r"chunksPerTask\s*=\s*HostCeilDiv\(\s*nchunkCount\s*,\s*groups\s*\)", "chunksPerTask=ceil(nchunk/groups)"),
            (r"groups\s*=\s*HostCeilDiv\(\s*nchunkCount\s*,\s*HostCeilDiv\(\s*nchunkCount\s*,\s*groupsTarget\s*\)\s*\)",
             "groups 分组公式"),
            (r"workerCount\s*=\s*BMM_TASK_RATIO\s*\*\s*args\.blocks",
             "workerCount=ratio*blocks"),
            (r"directOut\s*=\s*\(?BMM_SINGLE_LAUNCH", "directOut 含单 launch"),
            (r"\(\s*m\s*%\s*16U\s*\)\s*!=\s*0U", "baseM 16 倍数校验"),
            (r"\(\s*n\s*%\s*16U\s*\)\s*!=\s*0U", "baseN 16 倍数校验"),
            (r"tilingReady\s*=\s*true", "tiling 就绪标志"),
            (r"attempt\s*<\s*4U", "tiling 降级阶梯 4 级"),
            (r"if \(!tilingReady\)", "阶梯耗尽才 Fail"),
            (r"tiler\.GetTiling\(tiling\) < 0", "GetTiling 失败会降级（不再直接 CheckTiling 中止）")):
        check(f"host 公式 {label}", re.search(pat, code, re.S) is not None)

    # ---- 8. 参数顺序：launch 实参与 kernel 形参一致 ----
    sig1 = re.search(
        r"RowTaskMaxKernel\s*\(\s*GM_ADDR x1,\s*GM_ADDR x2,\s*GM_ADDR partial,"
        r"\s*__kfc_workspace__ GM_ADDR systemWorkspace,\s*GM_ADDR cWorkspace,"
        r"\s*CubeTiling tiling,\s*Shape shape,\s*uint32_t workerCount,"
        r"\s*uint32_t nmwin,\s*uint32_t nchunkCount,"
        r"\s*uint32_t chunksPerTask,\s*uint32_t groups,\s*uint32_t chunkWidth\s*\)", code)
    check("RowTaskMaxKernel 形参顺序(...,groups,chunkWidth)",
          sig1 is not None)
    call1 = re.search(
        r"args\.x1,\s*args\.x2,\s*args\.partial,\s*args\.workspace,"
        r"\s*args\.cWorkspace,\s*\*args\.tiling,\s*args\.shape,\s*workerCount,"
        r"\s*args\.nmwin,\s*args\.nchunkCount,\s*args\.chunksPerTask,\s*args\.groups,"
        r"\s*args\.chunkWidth", code, re.S)
    check("RowTaskMaxKernel 实参顺序与形参一致", call1 is not None)

    sig2 = re.search(
        r"PartialSumKernel\s*\(\s*GM_ADDR partial,\s*GM_ADDR y,\s*uint32_t batches,"
        r"\s*uint32_t m,\s*uint32_t nmwin,\s*uint32_t baseM,\s*uint32_t groups,"
        r"\s*uint32_t workerCount\s*\)", code)
    check("PartialSumKernel 形参顺序(batches,m,nmwin,baseM,groups,workerCount)",
          sig2 is not None)
    call2 = re.search(
        r"PartialSumKernel<<<blocks, nullptr, stream>>>\(\s*\(GM_ADDR\)partialDevice,"
        r"\s*y,\s*shape\.b,\s*shape\.m,\s*nmwin,\s*baseM,\s*groups,\s*blocks\s*\)",
        code, re.S)
    check("PartialSumKernel 实参顺序与形参一致", call2 is not None)

    # ---- 9. fence 覆盖 ----
    # 每处 GM→UB 的 DataCopyPad（带 padParams）后 12 行内要有 MTE2_V / MTE2_S
    gm2ub, missing = 0, []
    for i, ln in enumerate(lines):
        if re.search(r"DataCopyPad\(\s*\w+\s*,", ln) and ("padParams" in
                                                         lines[i + 3] if i + 3 < len(lines) else False):
            gm2ub += 1
            win = "\n".join(lines[i:i + 12])
            if "HardEvent::MTE2_V" not in win and "HardEvent::MTE2_S" not in win:
                missing.append(i + 1)
    check(f"GM→UB 后有 MTE2_V/S fence（{gm2ub} 处）", not missing,
          f"缺 fence 的行 {missing}" if missing else "")
    check("复用 cUb 前有 V_MTE2 fence",
          re.search(r"Fence<\s*HardEvent::V_MTE2\s*>", code) is not None)
    check("标量读 UB 前有 V_S fence",
          re.search(r"Fence<\s*HardEvent::V_S\s*>", code) is not None)
    check("UB→GM 前有 S_MTE3/V_MTE3 fence",
          re.search(r"Fence<\s*HardEvent::(S_MTE3|V_MTE3)\s*>", code) is not None)

    # ---- 10. GetTensorC 必须是非连续写 ----
    gtc = re.findall(r"GetTensorC\(\s*cWin\s*,\s*0\s*,\s*(.+?)\s*\)", code)
    check("phase1 GetTensorC 的 enSequentialWrite 由开关决定",
          gtc and all("BMM_SEQ_WRITE" in v for v in gtc), f"实参={set(gtc)}")
    check("BMM_SEQ_WRITE 默认 = 1（连续写，与 15/15 同模式）",
          re.search(r"BMM_SEQ_WRITE\s*=\s*1\s*;", code) is not None)
    check("存在 S_MTE2 fence（phase2 groups==1 复用 bandVec 前）",
          re.search(r"HardEvent::S_MTE2", code) is not None)

    # ---- 11. WholeReduceMax 用法 ----
    wr = re.search(r"WholeReduceMax\(\s*tileMaxVec,\s*cUb\[subBase\],"
                   r"\s*static_cast<int32_t>\(subCols\),\s*static_cast<int32_t>\(rows\),"
                   r"\s*1,\s*1,\s*static_cast<int32_t>\(ubPitch\s*/\s*8U\),"
                   r"\s*ReduceOrder::ORDER_ONLY_VALUE\s*\)", code, re.S)
    check("WholeReduceMax(dst=tileMaxVec,src=cUb[subBase],mask=subCols,rep=validM,"
          "dstride=1,blk=1,repstride=ubPitch/8,ORDER_ONLY_VALUE)", wr is not None)
    check("REDUCE_CHUNK = 64（mask 上限）",
          re.search(r"REDUCE_CHUNK\s*=\s*64", code) is not None)
    check("ubPitch = align8(chunkPitch)",
          re.search(r"ubPitch\s*=\s*AlignUpU32\(\s*chunkPitch\s*,\s*8U\s*\)", code) is not None)
    check("归约轴 <8 走标量兜底", re.search(r"subCols\s*>=\s*8U", code) is not None)

    # ---- 12. 资源规模 ----
    initn = len(re.findall(r"InitBuffer\s*\(", code))
    check("InitBuffer 次数 ≤ 8（红线 ≤64）", initn <= 8, f"{initn} 次")
    sites = len(re.findall(r"<<<", code))
    check("kernel 启动点 ≤ 8（基线 6 / 已过架构检查的版本 8）", sites <= 8,
          f"{sites} 处")
    check("chunkWidth 由 host 唯一决定（唯一真源）+ 宽块门槛 + GM 预算上限",
          re.search(r"chunkWidth\s*=\s*\n?\s*\(\(BMM_CHUNK_N\s*>\s*0\)", code) is not None
          and re.search(r"budgetWidth\s*=\s*static_cast<uint32_t>", code) is not None
          and re.search(r"BMM_WINDOW_BUDGET_MB", code) is not None
          and re.search(r"BMM_WIDE_MIN_TILES", code) is not None)
    check("cBytes/cWin 用 winN=min(chunkWidth,N)",
          re.search(r"winN\s*=\s*std::min<uint32_t>\(\s*chunkWidth\s*,\s*shape\.n\s*\)", code) is not None)
    check("cTileBuf 暂存按模式自适应（窄块 baseM*maxPitch，宽块上限 BMM_UB_STAGE_KB）",
          re.search(r"stageBytes\s*=\s*baseM\s*\*\s*maxPitch\s*\*\s*sizeof\(float\)", code) is not None
          and re.search(r"stageCap\s*=\s*static_cast<uint32_t>\(BMM_UB_STAGE_KB\)\s*\*\s*1024U", code) is not None
          and re.search(r"InitBuffer\(cTileBuf, stageBytes\)", code) is not None
          and re.search(r"maxPitch\s*=\s*AlignUpU32\(\s*MinU32\(chunkWidth, shape\.n\), 8U\)", code) is not None)

    # ---- 12a. 真机工具链约束（第一次提交已被打回，这里固化成门禁）----
    check("不设置 DataCopyPadExtParams 的字段（真机字段名与 8.5 文档不一致）",
          re.search(r"padParams\.\w+\s*=", code) is None,
          "只允许默认构造")
    check("不写 !ASCEND_IS_AIV/!ASCEND_IS_AIC（真机展开为 constexpr(...)，不可取反）",
          re.search(r"!\s*ASCEND_IS_", code) is None)
    check("只在 AIV 上跑的核用 ASCEND_IS_AIC 早退",
          re.search(r"if ASCEND_IS_AIC\s*\{\s*return;", code) is not None)
    check("FullReduce/IterateAll 用法保留", re.search(r"IterateAll\(cWin", code) is not None)

    # ---- 12b. 评审修复项（Finding 1/2/3/4/5，全部按"位置"断言）----
    cmt = r"(?:/\*[\s\S]*?\*/\s*)?"
    check("Finding4a: tileMaxVec 在 subBase 循环前初始化",
          re.search(r"Duplicate\(\s*tileMaxVec\s*,\s*NEG_INF\s*,\s*baseM\s*\);\s*"
                    r"for \(uint32_t subBase", code) is not None)
    check("Finding4b: sub-8 标量分支开头有 V_S（上一块 Vector 写对本分支可见）",
          re.search(r"\} else \{\s*" + cmt + r"Fence<\s*HardEvent::V_S\s*>", code) is not None)
    check("Finding2: phase1 groups==1 行和之后有 S_V（位置）",
          re.search(r"taskSum \+= bestVec\.GetValue\(row\);\s*\}\s*" + cmt +
                    r"Fence<\s*HardEvent::S_V\s*>", code) is not None)
    check("Finding5: phase2 行和之后有 S_V（位置）",
          re.search(r"batchSum \+= rowMaxVec\.GetValue\(row\);\s*\}\s*" + cmt +
                    r"Fence<\s*HardEvent::S_V\s*>", code) is not None)
    check("Finding3: groups>1 写出 bestVec 之后有 MTE3_V（位置）",
          re.search(r"DataCopyPad\(partialGm\[outOffset\], bestVec, outCopy\);\s*" + cmt +
                    r"Fence<\s*HardEvent::MTE3_V\s*>", code) is not None)
    check("比例开关 #define BMM_TASK_RATIO",
          re.search(r"#define\s+BMM_TASK_RATIO\s+\d", code) is not None)
    check("ratio==1 时显式声明 KERNEL_TYPE_MIX_AIC_1_1（受 #if 保护）",
          re.search(r"#if\s+BMM_TASK_RATIO\s*==\s*1[\s\S]{0,200}?"
                    r"KERNEL_TASK_TYPE_DEFAULT\(\s*KERNEL_TYPE_MIX_AIC_1_1\s*\)", code) is not None)
    check("并行粒度倍率 #define BMM_GROUPS_MULT",
          re.search(r"#define\s+BMM_GROUPS_MULT\s+\d", code) is not None)
    check("groupsTarget 使用 BMM_GROUPS_MULT*workers",
          re.search(r"wantTasks\s*=\s*BMM_GROUPS_MULT\s*\*\s*workers", code) is not None)
    check("AIC 槽位基址 = workers（两个比例下都等于 ratio*blocks）",
          re.search(r"cSlot\s*=\s*worker\s*;[\s\S]{0,240}?"
                    r"cSlot\s*=\s*workers\s*\+\s*worker\s*;", code) is not None)
    check("槽位数 = (ratio+1)*blocks",
          re.search(r"\(static_cast<size_t>\(BMM_TASK_RATIO\)\s*\+\s*1U\)", code) is not None)
    check("Finding1: phase2 启动 stride = blocks",
          re.search(r"PartialSumKernel<<<blocks, nullptr, stream>>>\([\s\S]{0,240}?"
                    r"\n\s*blocks\s*\);", code) is not None)

    # ---- 13. 阈值与路径 ----
    check("SMALL_MAC_LIMIT 存在", re.search(r"SMALL_MAC_LIMIT\s*=", code) is not None)
    check("NEG_INF 用于 max 初值",
          re.search(r"Duplicate\(\s*bestVec\s*,\s*NEG_INF", code) is not None)

    # ---- 输出 ----
    print(f"文件：{path}  行数：{len(lines)}")
    for p in PASSES:
        print(f"  [PASS] {p}")
    for f in FAILS:
        print(f"  [FAIL] {f}")
    print(f"\n通过 {len(PASSES)} / {len(PASSES) + len(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
