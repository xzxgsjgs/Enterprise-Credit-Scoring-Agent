"""agent.routing — 所有 LangGraph 路由函数集中管理。

集中原因：路由逻辑散落在 nodes.py 会让条件边与节点实现耦合，
且图改造（加 diagnose 回跳 / 加短路）时需要同时改两处，容易漏。

所有函数签名统一为 (state: AgentState) -> str，返回条件边的目标 key。
"""
from __future__ import annotations

from typing import Any, Literal

logger = __import__("logging").getLogger(__name__)

# 不可恢复节点：失败后无需再跑后续 13 个节点，直接短路到 END
CRITICAL_NODES = ("load_data", "split", "woebin", "model_train")


def has_critical_error(state: dict[str, Any]) -> bool:
    """是否存在不可恢复错误。

    判定：state["critical_error"] 为真，或 step_history 中出现 [ERR] <CRITICAL_NODES>。
    后者是兜底——万一某节点没显式置位，也能从历史里推断。
    """
    if state.get("critical_error"):
        return True
    history = state.get("step_history") or []
    for item in history:
        for node in CRITICAL_NODES:
            if item.startswith(f"[ERR] {node}"):
                return True
    return False


def route_after_critical(state: dict[str, Any]) -> Literal["end", "continue"]:
    """挂在不可恢复节点之后的条件边：有致命错误直接 END。"""
    return "end" if has_critical_error(state) else "continue"


def route_after_data_gate(state: dict[str, Any]) -> Literal["end", "continue"]:
    """data_gate 之后：未通过数据质量守门 → 直接 END（附原因）。"""
    gate = state.get("data_gate") or {}
    if not gate:
        # 节点未产出（异常场景），放行以免阻塞
        return "continue"
    return "continue" if gate.get("passed") else "end"


def _has_actionable_plan(state: dict[str, Any], cfg: dict[str, Any]) -> bool:
    """是否至少存在一条「尚未尝试过」的确定性处置方案。

    用于避免在明显无从下手时还白跑一次 diagnose（每次回跳都要重跑整条链路，代价高）。
    仅当 LLM 诊断关闭时才做这个预判——LLM 开启时无法低成本预知 LLM 是否会给方案，
    一律放行。任何异常都按 True 处理（宁可多跑一次，也不能误判成「无解」直接转人工）。
    """
    try:
        from agent.nodes import _plan_signature, _rule_action_plan

        plan = _rule_action_plan(state, cfg)
        tried = {
            _plan_signature(h.get("goto", ""), h.get("param_patch") or {})
            for h in (state.get("retry_history") or [])
        }
        return any(
            goto != "__non_actionable__" and _plan_signature(goto, patch) not in tried
            for goto, patch, _reason in plan
        )
    except Exception:  # noqa: BLE001 - 路由不可抛异常
        logger.warning("可行动作预检失败，按有可行动作处理")
        return True


def route_after_guardrail(
    state: dict[str, Any],
) -> Literal["end", "diagnose", "hitl"]:
    """guardrail 之后的三岔路由（V2 新增 diagnose 分支）。

    优先级：
      1. 无护栏结果                       → hitl（异常兜底）
      2. 通过且无需人审                   → end
      3. 未通过但重试未超限且有可行动作   → diagnose（自动重规划）
      4. 其他                             → hitl
    """
    gr = state.get("guardrail")
    if gr is None:
        return "hitl"

    if getattr(gr, "passed", False) and not getattr(gr, "requires_human_review", True):
        return "end"

    # 读取 agent 配置（config_overrides 是扁平化后的一层 dict）
    cfg = state.get("config_overrides") or {}
    max_retry = int(cfg.get("max_retry", 3))
    retry_count = int(state.get("retry_count") or 0)

    if retry_count >= max_retry:
        return "hitl"

    # allow_llm_diagnosis=True 时不预判（LLM 可能给出规则库没有的方案）
    if not bool(cfg.get("allow_llm_diagnosis", True)) and not _has_actionable_plan(state, cfg):
        logger.info("无可用自动处置方案，直接转人工")
        return "hitl"

    return "diagnose"


def route_after_diagnose(
    state: dict[str, Any],
) -> Literal["preprocess", "var_filter", "woebin", "model_train",
            "build_scorecard", "hitl"]:
    """diagnose 之后按 diagnosis["goto"] 回跳；缺失/非法 → hitl。"""
    diagnosis = state.get("diagnosis") or {}
    goto = diagnosis.get("goto")
    allowed = {"preprocess", "var_filter", "woebin", "model_train", "build_scorecard"}
    if goto in allowed:
        return goto  # type: ignore[return-value]
    return "hitl"


def route_after_hitl(state: dict[str, Any]) -> Literal["end"]:
    """HITL 之后直接结束。"""
    return "end"