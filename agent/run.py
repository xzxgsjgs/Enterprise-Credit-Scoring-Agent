"""agent.run — CLI 入口

用法：
    # 1) 一次性跑完整流程
    python -m agent.run --csv data.csv --target creditability

    # 2) 若 hitl 中断，人工决策后：
    python -m agent.run --resume <thread_id> --action approve --note "..."

    # 3) 查询状态：
    python -m agent.run --thread <thread_id> --show-state
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

from langgraph.types import Command

from agent.graph import compile_with_sqlite
from agent.state import AgentState

logger = logging.getLogger(__name__)


def _load_config_overrides(config_path: str | None) -> dict[str, Any]:
    """从 YAML 加载配置覆盖。"""
    if not config_path:
        return {}
    try:
        import yaml
    except ImportError:
        logger.warning("pyyaml 未安装，跳过配置文件")
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # 扁平化：把 yaml 里嵌套的 key 合并到一层
    overrides: dict[str, Any] = {}
    for section, vals in cfg.items():
        if isinstance(vals, dict):
            overrides.update(vals)
        else:
            overrides[section] = vals
    # 项目名/版本不参与覆盖
    overrides.pop("project", None)
    overrides.pop("name", None)
    overrides.pop("version", None)
    return overrides


def _print_chunk(chunk: dict[str, Any]) -> None:
    """格式化打印单个 stream chunk。"""
    for node, update in chunk.items():
        if node.startswith("__"):
            continue
        # 只打印 step_history 与 errors/warnings，避免打印巨大 DataFrame
        if isinstance(update, dict):
            hist = update.get("step_history", [])
            errs = update.get("errors", [])
            warns = update.get("warnings", [])
            for h in hist:
                print(f"  {h}")
            for e in errs:
                print(f"  ERROR: {e}")
            for w in warns:
                print(f"  WARN: {w}")
        else:
            print(f"  [{node}] updated")


def run_full(
    csv_path: str,
    target: str = "creditability",
    config_path: str | None = None,
    thread_id: str | None = None,
    db_path: str = "checkpoints.db",
) -> dict[str, Any]:
    """运行完整图，stream 打印每节点结果，若中断则停在 hitl。"""
    overrides = _load_config_overrides(config_path)
    initial: AgentState = {
        "csv_path": csv_path,
        "target": target,
        "config_overrides": overrides,
    }
    thread_id = thread_id or str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}

    app, conn = compile_with_sqlite(db_path)
    try:
        print(f"\n=== Credit Scorecard Agent Run ===")
        print(f"thread_id: {thread_id}")
        print(f"csv: {csv_path} | target: {target}")
        print(f"overrides: {overrides}\n")

        for chunk in app.stream(initial, config=cfg, stream_mode="updates"):
            _print_chunk(chunk)

        # 检查是否中断（state.next 非空表示停在某节点上）
        snap = app.get_state(cfg)
        if snap.next:
            print(f"\n[HITL REQUIRED] next={snap.next}")
            print(f"resume: python -m agent.run --resume {thread_id} --action approve --note '...'")
            return {"thread_id": thread_id, "interrupted": True, "next": list(snap.next)}

        print(f"\n=== Run Complete ===")
        final = snap.values
        print(f"metrics: {final.get('metrics', {})}")
        print(f"guardrail.passed: {final.get('guardrail').passed if final.get('guardrail') else 'N/A'}")
        print(f"hitl_decision: {final.get('hitl_decision', 'N/A')}")
        return {"thread_id": thread_id, "interrupted": False, "final": final}
    finally:
        conn.close()


def resume_hitl(
    thread_id: str,
    action: str = "approve",
    note: str = "",
    db_path: str = "checkpoints.db",
) -> dict[str, Any]:
    """HITL 恢复：用 Command(resume=...) 继续执行。"""
    app, conn = compile_with_sqlite(db_path)
    try:
        cfg = {"configurable": {"thread_id": thread_id}}
        snap = app.get_state(cfg)
        if not snap.next:
            print(f"[WARN] thread {thread_id} 不在 HITL 中断状态")
            return {"thread_id": thread_id, "resumed": False}

        print(f"[HITL RESUME] thread={thread_id} action={action} note={note}")
        resume_payload = {"action": action, "note": note}
        for chunk in app.stream(Command(resume=resume_payload), config=cfg, stream_mode="updates"):
            _print_chunk(chunk)

        final = app.get_state(cfg).values
        print(f"\nhitl_decision: {final.get('hitl_decision')}")
        print(f"hitl_note: {final.get('hitl_note')}")
        return {"thread_id": thread_id, "resumed": True, "final": final}
    finally:
        conn.close()


def show_state(thread_id: str, db_path: str = "checkpoints.db") -> None:
    """打印指定 thread 的当前状态。"""
    app, conn = compile_with_sqlite(db_path)
    try:
        cfg = {"configurable": {"thread_id": thread_id}}
        snap = app.get_state(cfg)
        print(f"=== State for thread {thread_id} ===")
        print(f"next: {list(snap.next) if snap.next else 'END (已完成)'}")
        # 序列化为可打印 dict（DataFrame 替换为摘要）
        printable: dict = {}
        for k, val in snap.values.items():
            if hasattr(val, "shape"):
                printable[k] = f"<DataFrame shape={val.shape}>"
            elif hasattr(val, "__dict__"):
                printable[k] = f"<{type(val).__name__}: {val.__dict__}>"
            elif isinstance(val, (list, dict, str, int, float, bool)) or val is None:
                printable[k] = val
            else:
                printable[k] = f"<{type(val).__name__}>"
        print(json.dumps(printable, indent=2, ensure_ascii=False, default=str))
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Credit Scorecard Agent CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # run
    p_run = sub.add_parser("start", help="启动一次完整运行")
    p_run.add_argument("--csv", required=True, help="数据文件路径")
    p_run.add_argument("--target", default="creditability", help="目标列名")
    p_run.add_argument("--config", default=None, help="YAML 配置文件路径")
    p_run.add_argument("--thread", default=None, help="自定义 thread_id")
    p_run.add_argument("--db", default="checkpoints.db", help="SQLite 文件路径")

    # resume
    p_res = sub.add_parser("resume", help="HITL 恢复")
    p_res.add_argument("--thread", required=True, help="thread_id")
    p_res.add_argument("--action", default="approve", choices=["approve", "reject"])
    p_res.add_argument("--note", default="", help="备注")
    p_res.add_argument("--db", default="checkpoints.db", help="SQLite 文件路径")

    # show
    p_show = sub.add_parser("show", help="查看指定 thread 的状态")
    p_show.add_argument("--thread", required=True, help="thread_id")
    p_show.add_argument("--db", default="checkpoints.db", help="SQLite 文件路径")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.cmd == "start":
        result = run_full(args.csv, args.target, args.config, args.thread, args.db)
        return 0 if not result.get("interrupted") else 1
    if args.cmd == "resume":
        result = resume_hitl(args.thread, args.action, args.note, args.db)
        return 0 if result.get("resumed") else 1
    if args.cmd == "show":
        show_state(args.thread, args.db)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())