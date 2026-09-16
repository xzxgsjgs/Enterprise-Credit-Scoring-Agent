"""app.core.agent_runner —— 在 Streamlit 中驱动 LangGraph Agent 闭环。

职责边界（与 `app/core/training.py` 的分工）：
- `training.py`：确定性 pipeline，跑一次出模型，**无守门/无自愈/无复核/无报告**
- `agent_runner.py`（本文件）：驱动 18 节点 LangGraph 图，把 V2 六个模块
  （数据守门 / 诊断重规划 / 分级自愈 / 自然语言配置 / cut-off 与报告 / Critic）
  串成闭环，并把每个节点的执行情况以事件流形式喂给 UI

设计要点：
1. **不在 UI 里做任何数值计算**：本文件只负责「跑图 + 转发事件 + 摘状态」，
   所有指标、护栏、cut-off、报告都由图内节点经 `tools/` 确定性算出
2. **可中断可恢复**：图在 `hitl_review` 处 interrupt，UI 收集人的决策后用
   `Command(resume=...)` 继续，thread_id 与 SqliteSaver 绑定（同一 thread 可续跑）
3. **对 Streamlit 友好**：`run_agent()` 接受 `on_event` 回调，UI 用占位符实时刷新；
   图与 sqlite 连接用 `get_agent()` 缓存，避免每次 rerun 重编译
"""
from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from agent.graph import compile_with_sqlite
from app.core.paths import CACHE_DIR

logger = logging.getLogger(__name__)

# UI 用的 checkpoint 库（与 CLI 的 checkpoints.db 分开，避免互相污染）
UI_CHECKPOINT_DB = CACHE_DIR / "ui_checkpoints.db"

# ============================================================================
# 节点元数据：UI 时间线展示用（顺序 = 图的实际拓扑顺序）
# ============================================================================
NODE_META: dict[str, tuple[str, str]] = {
    # node_name: (中文标签, 所属模块)
    "load_data": ("加载数据", "基础"),
    "explore_data": ("数据探索", "基础"),
    "data_gate": ("数据质量守门", "模块 6"),
    "split": ("训练/测试切分", "基础"),
    "preprocess": ("预处理（缺失+异常）", "基础"),
    "var_filter": ("变量筛选（IV）", "基础"),
    "woebin": ("WOE 分箱", "基础"),
    "woebin_apply": ("WOE 转换", "基础"),
    "model_train": ("逻辑回归训练", "基础"),
    "model_predict": ("模型预测", "基础"),
    "evaluate_performance": ("指标评估", "基础"),
    "build_scorecard": ("评分卡构建", "基础"),
    "critic": ("Critic 模型复核", "模块 3"),
    "scorecard_ply": ("评分卡打分", "基础"),
    "guardrail_check": ("合规护栏检查", "基础"),
    "diagnose": ("诊断-重规划决策", "模块 1/2"),
    "hitl_review": ("人工复核", "护栏"),
    "reporter": ("报告与 cut-off", "模块 5"),
}

# 时间线的主干顺序（diagnose 是回跳，单独标注，不占主干位置）
NODE_ORDER: list[str] = [
    "load_data", "explore_data", "data_gate", "split", "preprocess", "var_filter",
    "woebin", "woebin_apply", "model_train", "model_predict", "evaluate_performance",
    "build_scorecard", "critic", "scorecard_ply", "guardrail_check",
    "diagnose", "hitl_review", "reporter",
]

# 不可恢复节点（失败即短路 END），UI 用红色标注
CRITICAL_NODES = {"load_data", "split", "woebin", "model_train"}


def node_label(node: str) -> str:
    return NODE_META.get(node, (node, "其他"))[0]


def node_module(node: str) -> str:
    return NODE_META.get(node, (node, "其他"))[1]


# ============================================================================
# 配置组装（纯函数，便于单测）
# ============================================================================
AGENT_PARAM_KEYS = (
    "max_retry", "allow_llm_diagnosis", "allow_llm_critic", "allow_llm_report",
    "generate_report", "report_dir", "cutoff_method", "cutoff_target",
    # data_gate 阈值
    "min_rows", "min_default_rate", "max_default_rate", "min_years",
    "max_missing_ratio", "max_identical_ratio",
    # critic 阈值
    "vif_threshold", "concentration_threshold", "critic_top_features",
)


def build_agent_config(
    base: dict[str, Any],
    agent_opts: dict[str, Any] | None = None,
    nl_patch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把 UI 的基础配置、Agent 设置、自然语言解析结果合并成一张扁平 config_overrides。

    三层来源的信任级别不同，处理方式也不同：
    - `base`：UI 侧边栏的基础建模参数（test_size / max_bins / min_ks ...），全部保留
    - `agent_opts`：Agent 专属开关与阈值，**只接受 AGENT_PARAM_KEYS 白名单**，
      避免 UI 误传键污染 `agent/nodes.py::_validate_diagnosis` 的校验层
    - `nl_patch`：自然语言解析结果。它已经过 `agent/nl_config.validate_patch`
      的白名单与 `param_bounds` 夹取校验，因此**原样合并、不二次过滤**——
      否则像 `max_bins` / `iv_threshold` 这类「基础建模参数」会被误拦（实测踩过）
    """
    cfg = dict(base or {})
    for k, v in (agent_opts or {}).items():
        if k in AGENT_PARAM_KEYS:
            cfg[k] = v
    for k, v in (nl_patch or {}).items():
        cfg[k] = v
    return cfg


# ============================================================================
# 图与连接的持有
# ============================================================================
def get_agent(db_path: str | Path = UI_CHECKPOINT_DB) -> tuple[Any, sqlite3.Connection]:
    """编译图并返回 (app, conn)。调用方负责复用，进程退出时随进程释放。

    注意：不使用 st.cache_resource，便于在非 Streamlit 环境（测试）中直接调用；
    UI 层自己用 `@st.cache_resource` 包一层即可。
    """
    return compile_with_sqlite(db_path)


# ============================================================================
# 运行结果
# ============================================================================
@dataclass
class NodeEvent:
    """一次节点执行的 UI 事件。"""
    node: str
    label: str
    module: str
    status: str                     # ok | warn | fail | retry
    detail: str = ""
    step_history: list[str] = field(default_factory=list)


@dataclass
class AgentRunResult:
    thread_id: str
    status: str                     # completed | interrupted | failed
    state: dict[str, Any]
    node_log: list[NodeEvent]
    elapsed: float
    interrupt_payload: dict[str, Any] | None = None
    error: str = ""

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    @property
    def interrupted(self) -> bool:
        return self.status == "interrupted"


def _event_of(node: str, update: Any) -> NodeEvent:
    """把某个节点的 state 更新转成 UI 事件（含成功/告警/失败判定）。"""
    upd = update if isinstance(update, dict) else {}
    errs = upd.get("errors") or []
    warns = upd.get("warnings") or []
    hist = upd.get("step_history") or []
    critical = bool(upd.get("critical_error"))

    if errs:
        status = "fail"
        detail = errs[-1]
    elif warns:
        status = "warn"
        detail = warns[-1]
    else:
        status = "ok"
        detail = hist[-1] if hist else ""

    # diagnose 是回跳决策，单独标色（UI 上表现为回路）
    if node == "diagnose":
        status = "retry"
        diag = upd.get("diagnosis") or {}
        if diag:
            detail = (
                f"{diag.get('action', '?')} → {diag.get('goto', '?')}"
                f"（{diag.get('source', 'rule')}）"
            )
    if critical:
        status = "fail"

    return NodeEvent(
        node=node, label=node_label(node), module=node_module(node),
        status=status, detail=str(detail), step_history=list(hist),
    )


def iter_agent(
    csv_path: str,
    target: str,
    config: dict[str, Any],
    thread_id: str | None = None,
    db_path: str | Path = UI_CHECKPOINT_DB,
    on_event: Callable[[NodeEvent], None] | None = None,
    resume_payload: dict[str, Any] | None = None,
) -> Iterator[NodeEvent]:
    """流式跑图，逐个 yield 节点事件；结束后把结果写入 result 变量。

    用法（UI）：
        result = None
        for ev in stream_agent(..., on_event=cb):
            ...
        # 或直接调 run_agent() 拿汇总结果
    """
    app, conn = get_agent(db_path)
    tid = thread_id or str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": tid}}
    try:
        if resume_payload is None:
            initial: dict[str, Any] = {
                "csv_path": csv_path,
                "target": target,
                "config_overrides": dict(config or {}),
            }
            stream_input: Any = initial
        else:
            from langgraph.types import Command

            stream_input = Command(resume=resume_payload)

        for chunk in app.stream(stream_input, config=cfg, stream_mode="updates"):
            for node, update in (chunk or {}).items():
                if str(node).startswith("__"):
                    continue
                ev = _event_of(str(node), update)
                if on_event is not None:
                    on_event(ev)
                yield ev
    finally:
        conn.close()


def run_agent(
    csv_path: str,
    target: str,
    config: dict[str, Any],
    thread_id: str | None = None,
    db_path: str | Path = UI_CHECKPOINT_DB,
    on_event: Callable[[NodeEvent], None] | None = None,
    resume_payload: dict[str, Any] | None = None,
) -> AgentRunResult:
    """跑完（或跑到中断）整张图，返回可渲染的汇总结果。

    Returns:
        AgentRunResult.status ∈ {"completed", "interrupted", "failed"}
        - interrupted：图停在 `hitl_review`，需带 `resume_payload` 再调一次
        - failed：不可恢复节点短路，`error` 里有原因
    """
    started = time.time()
    tid = thread_id or str(uuid.uuid4())
    node_log: list[NodeEvent] = []

    def _collect(ev: NodeEvent) -> None:
        node_log.append(ev)
        if on_event is not None:
            on_event(ev)

    error = ""
    try:
        for _ in iter_agent(
            csv_path, target, config, tid, db_path,
            on_event=_collect, resume_payload=resume_payload,
        ):
            pass
    except Exception as e:  # noqa: BLE001 - 图内异常向 UI 暴露，不静默
        logger.exception("agent 运行异常")
        error = f"{type(e).__name__}: {e}"

    # 取最终 state（无论完成、中断还是失败都要拿，UI 才有东西展示）
    state: dict[str, Any] = {}
    snap_next: list[str] = []
    app, conn = get_agent(db_path)
    try:
        snap = app.get_state({"configurable": {"thread_id": tid}})
        state = dict(snap.values or {})
        snap_next = [str(x) for x in (snap.next or [])]
    except Exception as e:  # noqa: BLE001
        logger.warning("读取 state 失败: %s", e)
    finally:
        conn.close()

    if error:
        status = "failed"
    elif snap_next:
        status = "interrupted"
    elif state.get("critical_error"):
        status = "failed"
        errs = state.get("errors") or []
        error = errs[-1] if errs else "不可恢复节点失败，流程已短路终止"
    else:
        status = "completed"

    return AgentRunResult(
        thread_id=tid,
        status=status,
        state=state,
        node_log=node_log,
        elapsed=time.time() - started,
        interrupt_payload=_interrupt_payload(state),
        error=error,
    )


def _interrupt_payload(state: dict[str, Any]) -> dict[str, Any] | None:
    """从 state 里摘出 HITL 需要展示的信息（图中断时用）。"""
    gr = state.get("guardrail")
    if gr is None and not state.get("critical_error"):
        return None
    return {
        "guardrail": gr.to_dict() if hasattr(gr, "to_dict") else {},
        "metrics": state.get("metrics") or {},
        "warnings": (state.get("warnings") or [])[-8:],
        "retry_history": state.get("retry_history") or [],
    }


# ============================================================================
# 状态摘要（顶部卡片 / 概览区用）
# ============================================================================
def summarize_state(state: dict[str, Any]) -> dict[str, Any]:
    """把 state 提炼成 UI 概要，避免 UI 里散落一堆 .get()。"""
    metrics = state.get("metrics") or {}
    test_m = metrics.get("test") or {}
    guardrail = state.get("guardrail")
    gate = state.get("data_gate") or {}
    critic = state.get("critic_report") or {}
    cutoff = state.get("cutoff_info") or {}
    hist = state.get("retry_history") or []

    return {
        "auc": test_m.get("auc"),
        "ks": test_m.get("ks"),
        "gini": test_m.get("gini"),
        "psi": test_m.get("psi"),
        "guardrail_passed": getattr(guardrail, "passed", None),
        "needs_human_review": getattr(guardrail, "requires_human_review", None),
        "gate_passed": gate.get("passed"),
        "gate_suggestion": gate.get("suggestion", ""),
        "n_retry": len(hist),
        "retry_history": hist,
        "critic": critic,
        "cutoff": cutoff,
        "critic_has_high": (critic.get("n_high") or 0) > 0,
        "report_path": state.get("report_path"),
        "report_md": state.get("report_md") or "",
        "hitl_decision": state.get("hitl_decision"),
        "critical_error": bool(state.get("critical_error")),
        "errors": state.get("errors") or [],
        "warnings": state.get("warnings") or [],
    }


# ============================================================================
# 产物落盘：让「实时评分」页能直接用上 Agent 产出的评分卡
# ============================================================================
def save_scorecard(state: dict[str, Any], dest: Path) -> bool:
    """把 state 里的评分卡 pickle 到 dest（供实时评分页加载）。"""
    import pickle

    card = state.get("scorecard")
    if not card:
        return False
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(pickle.dumps(card))
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("评分卡落盘失败: %s", e)
        return False


# ============================================================================
# 自然语言配置（模块 4 的 UI 侧入口）
# ============================================================================
def parse_requirement(text: str, use_llm: bool = True) -> tuple[dict[str, Any], list[str]]:
    """自然语言需求 → 参数 patch。解析与校验都在 agent/nl_config.py 内完成。"""
    from agent.nl_config import parse_nl_config

    if not text or not text.strip():
        return {}, []
    return parse_nl_config(text, use_llm=use_llm)
