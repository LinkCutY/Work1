#!/usr/bin/env python3
"""submit_sweep.py — 变体生成 + 真提交 + 用时采集（AIV/AIC 比例搜索的执行器）

用法（都需要本机已有 ~/.cannjudge/session.json）：
  python submit_sweep.py --list                    # 列出变体与将要改的行
  python submit_sweep.py --dry-run                 # 只生成 payload 校验（不提交，无需鉴权）
  python submit_sweep.py --submit --variants base,r1
  python submit_sweep.py --submit --variants r2_bm160 --final best
  python submit_sweep.py --show                    # 打印已采集结果

设计要点
  * 变体 = 对 bmmms-kernel-opt/kernel.asc 顶部几个常量做**受校验的**文本替换
    （替换没命中就报错，绝不静默生成同一份代码）。
  * 每个变体复制一份官方 project 目录（只读文件保持原样），只替换 kernel.asc，
    然后用 tools/cann/cann.py 的 Judge 直接提交并轮询到结束。
  * 结果落 sweep_results/results.jsonl：每个变体一行，含每 case 的
    testcase_status / time / msg，便于"按实际用时找最优"。
  * **判题按最后一次提交计成绩**：全部跑完后默认再提交一次胜者（--final best），
    要关掉用 --no-final。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import shutil
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
WORKSPACE = HERE.parent
CLONE = WORKSPACE / ".research/repos/CANNCompetition"
CANN_TOOLS = CLONE / "tools/cann"
PROJECT_SRC = CLONE / "projects/batchmatmul-maxsum/npu-v1/project"
KERNEL = HERE / "kernel.asc"
RESULTS = HERE / "sweep_results"
PROBLEM_ID = "6a9aa054bf41025d6014f3ef"

sys.path.insert(0, str(CANN_TOOLS))
import cann  # noqa: E402  （复用已验证的 Judge / 提交流程）

# ---------------------------------------------------------------------------
# 变体定义：name -> [(常量名, 新值)]，只允许改 kernel.asc 顶部那几个开关
# ---------------------------------------------------------------------------
VARIANTS: dict[str, list[tuple[str, str]]] = {
    "base": [],                                   # 原样（ratio=2, 128/128, 连续写=真机已验证模式）
    "r1": [("BMM_TASK_RATIO", "1")],               # AIV:AIC = 1:1
    "r1_bm160": [("BMM_TASK_RATIO", "1"), ("BASE_M", "160"), ("BASE_N", "192")],
    "bm160": [("BASE_M", "160"), ("BASE_N", "192")],   # 只放大 tile（ratio=2）
    "bm64": [("BASE_M", "64"), ("BASE_N", "256")],
    "narrow": [("BMM_CHUNK_N", "0")],              # 回退：每 chunk 一个 baseN tile（已验证模式）
    "nonseq": [("BMM_SEQ_WRITE", "0")],            # 非连续写实验（需先确认 orgKc 语义）
    "g2": [("BMM_GROUPS_MULT", "2")],              # 任务粒度 x2（削末波次空转）
    "g4": [("BMM_GROUPS_MULT", "4")],
    "atomic": [("BMM_SINGLE_LAUNCH", "1")],        # 单 launch 兜底
}


def patch_kernel(overrides: list[tuple[str, str]]) -> tuple[str, list[str]]:
    src = KERNEL.read_text(encoding="utf-8")
    log = []
    for name, value in overrides:
        # 覆盖 static constexpr int/uint32_t 与 #define 两种形态
        pat = re.compile(rf"(#define\s+{name}\s+)(\S+)")
        m = pat.search(src)
        if not m:
            pat = re.compile(rf"(static constexpr\s+\w+\s+{name}\s*=\s*)(\d+)")
            m = pat.search(src)
        if not m:
            raise SystemExit(f"变体常量 {name} 在 kernel.asc 里找不到，拒绝生成")
        old = m.group(2)
        if old == value:
            raise SystemExit(f"变体常量 {name} 已是 {value}，替换无意义")
        src = src[:m.start(2)] + value + src[m.end(2):]
        log.append(f"{name}: {old} -> {value}")
    return src, log


def project_from_source(name: str, src: str, log: list[str]) -> pathlib.Path:
    d = pathlib.Path(tempfile.mkdtemp(prefix=f"sweep_{name}_"))
    for item in PROJECT_SRC.iterdir():
        if item.is_file():
            shutil.copy2(item, d / item.name)
        elif item.is_dir():
            shutil.copytree(item, d / item.name)
    (d / "kernel.asc").write_text(src, encoding="utf-8")
    print(f"[{name}] {', '.join(log)}  dir={d}")
    return d


def project_for(name: str, overrides: list[tuple[str, str]]) -> pathlib.Path:
    src, log = patch_kernel(overrides)
    d = pathlib.Path(tempfile.mkdtemp(prefix=f"sweep_{name}_"))
    for item in PROJECT_SRC.iterdir():
        if item.is_file():
            shutil.copy2(item, d / item.name)
        elif item.is_dir():
            shutil.copytree(item, d / item.name)
    (d / "kernel.asc").write_text(src, encoding="utf-8")
    # 变体自检：与基线内核必须不同（除 base）
    if overrides and src == KERNEL.read_text(encoding="utf-8"):
        raise SystemExit(f"变体 {name} 生成后与基线逐字节相同，拒绝")
    print(f"[{name}] {', '.join(log) if log else '基线原样'}  dir={d}")
    return d


def summarise(record: dict) -> dict:
    res = record.get("result") or []
    times = [t.get("time") for t in res]
    numeric = [float(t) for t in times if isinstance(t, (int, float))]
    return {
        "status": record.get("status"),
        "msg": record.get("msg"),
        "cases": [{"i": i + 1, "st": t.get("testcase_status"),
                   "time": t.get("time"), "msg": (t.get("msg") or "")[:200]}
                  for i, t in enumerate(res)],
        "pass_n": sum(t.get("testcase_status") in ("Pass", "Accepted") for t in res),
        "total_time": sum(numeric) if numeric else None,
        "max_time": max(numeric) if numeric else None,
    }


def save(name: str, sid, summary: dict, log: list[str]) -> None:
    RESULTS.mkdir(exist_ok=True)
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "variant": name,
           "submission_id": sid, "patches": log, **summary}
    with (RESULTS / "results.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[{name}] {rec['status']} pass={rec['pass_n']}/{len(rec['cases'])} "
          f"total={rec['total_time']} max={rec['max_time']} sid={sid}")
    for c in rec["cases"]:
        print(f"    case{c['i']:>2}: {c['st']:<10} time={c['time']}"
              + (f"  {c['msg']}" if c["msg"] else ""))


def load_results() -> list[dict]:
    f = RESULTS / "results.jsonl"
    if not f.exists():
        return []
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]


def submit_one(name: str, overrides: list[tuple[str, str]], wait_s: int) -> dict | None:
    # 提交前先做本机 g++ 语法检查：不通过就不浪费一次提交额度
    try:
        import syntax_check as sc
        src_txt, _ = patch_kernel(overrides)
        ok, msg = sc._check_one(name, src_txt)
        if not ok:
            print(f"[{name}] 语法检查未通过，跳过提交：")
            for l in msg[:8]:
                print("    " + l)
            return None
        print(f"[{name}] 语法检查 OK")

        # 第二道门禁：API 风险分级（真机未验证的调用要显式提示，必要时单独提交）
        import api_gate as ag
        tmp = pathlib.Path(HERE / f".api_gate_submit_{name}.asc")
        tmp.write_text(src_txt, encoding="utf-8")
        try:
            rc = ag.report(name, tmp)
            if rc != 0:
                print(f"[{name}] ⚠ 引入了真机未验证的 API —— 建议单独提交验证")
        finally:
            tmp.unlink(missing_ok=True)
    except Exception as e:
        print(f"[{name}] 语法检查无法执行（{type(e).__name__}: {e}），仍继续提交")

    d = project_for(name, overrides)
    j = cann.Judge()
    payload = j.prepare_submit(PROBLEM_ID, d)
    h = hashlib.sha256(payload["files"][0]["content"].encode()).hexdigest()[:16]
    print(f"[{name}] payload kernel.asc sha256={h} bytes="
          f"{len(payload['files'][0]['content'].encode())}")
    sid = j.submit_payload(payload)
    print(f"[{name}] submitted: {sid}")
    rec = j.wait(sid, timeout=wait_s)
    s = summarise(rec)
    save(name, sid, s, [f"{k}:{v}" for k, v in overrides])
    return {"name": name, "sid": sid, **s}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--variants", default="base,r1")
    ap.add_argument("--wait", type=int, default=1800, help="单次提交轮询上限(秒)")
    ap.add_argument("--final", default="best", choices=["best", "none"],
                    help="收尾是否再提交胜者（判题按最后一次提交计成绩）")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    if a.list:
        for n, ov in VARIANTS.items():
            print(f"  {n:10s} {ov if ov else '(基线原样)'}")
        return 0

    if a.show:
        for r in load_results():
            print(f"{r['ts']} {r['variant']:10s} {r['status']} "
                  f"pass={r['pass_n']} total={r['total_time']} max={r['max_time']} "
                  f"sid={r['submission_id']}")
        return 0

    names = [n.strip() for n in a.variants.split(",") if n.strip()]
    for n in names:
        if n not in VARIANTS:
            raise SystemExit(f"未知变体 {n}；用 --list 看可用项")

    if a.dry_run:
        j = cann.Judge()
        for n in names:
            d = project_for(n, VARIANTS[n])
            payload = j.prepare_submit(PROBLEM_ID, d)
            print(f"  dry-run OK: {n} files={[f['path'] for f in payload['files']]} "
                  f"userId={payload['userId'] or 'EMPTY'}")
        print("全部变体 payload 通过校验（未提交）")
        return 0

    if not a.submit:
        ap.print_help()
        return 1

    done = []
    for n in names:
        try:
            r = submit_one(n, VARIANTS[n], a.wait)
            if r:
                done.append(r)
        except SystemExit:
            raise
        except Exception as e:            # 单个变体失败不阻断其余实验
            print(f"[{n}] 提交/轮询失败: {type(e).__name__}: {e}")

    # 收尾：按"通过数优先、总用时次之"选胜者再提一次
    if a.final == "best" and done:
        ok = [r for r in done if r["total_time"] is not None]
        if ok:
            best = max(ok, key=lambda r: (r["pass_n"], -(r["total_time"] or 1e9)))
            print(f"\n=== 胜者: {best['name']} pass={best['pass_n']} "
                  f"total={best['total_time']}")
            if best["name"] != names[-1]:
                try:
                    submit_one(best["name"], VARIANTS[best["name"]], a.wait)
                except Exception as e:
                    print(f"收尾复提失败: {type(e).__name__}: {e}")
        else:
            print("\n没有拿到任何可比较的用时数据 —— 恢复提交队友的 15/15 版本")
            # 恢复源用不可变快照（官方项目路径可能已被换成新候选）
            good = WORKSPACE / "kernel-src/_pushed_123b2b8_15of15.asc"
            src = good.read_text(encoding="utf-8")
            d = project_from_source("restore_15of15", src, ["恢复 15/15 内核"])
            j = cann.Judge()
            payload = j.prepare_submit(PROBLEM_ID, d)
            sid = j.submit_payload(payload)
            print(f"已恢复提交 15/15: {sid}")
            save("restore_15of15", sid, summarise(j.wait(sid, timeout=a.wait)),
                 ["restore upstream 15/15"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
