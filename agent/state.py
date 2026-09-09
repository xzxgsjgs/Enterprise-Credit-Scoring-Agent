"""agent.state — LangGraph 状态 Schema

设计原则（来自 Kalvium Labs 生产经验）：
- Accumulator 字段：用 Annotated[list, operator.add] 累积（errors/warnings/step_history）
- Overwrite 字段：用 plain TypedDict，每次节点写入覆盖（model/df_clean/bins/scorecard）
- HITL 字段：hitl_decision/hitl_note 由 hitl_review_node 写入
"""
from __future__ import annotations

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    # ====== 输入（一次写入） ======
    csv_path: str
    target: str
    config_overrides: dict[str, Any]

    # ====== 数据流（overwrite） ======
    raw_df: Any              # explore_data 前
    eda_report: dict[str, Any]
    train_df: Any
    test_df: Any
    df_clean: Any            # missing + outlier 后
    bins: dict[str, Any]     # WOE 分箱
    train_woe: Any
    test_woe: Any
    filtered_vars: list[str]
    model: Any
    train_proba: Any
    test_proba: Any
    metrics: dict[str, float]
    scorecard: dict[str, Any]
    scored_test: Any
    guardrail: Any           # GuardrailResult

    # ====== 累积字段（reducer） ======
    errors: Annotated[list[str], operator.add]
    warnings: Annotated[list[str], operator.add]
    step_history: Annotated[list[str], operator.add]

    # ====== HITL ======
    hitl_decision: str       # "approve" | "reject"
    hitl_note: str


# 输入/输出过滤（控制 graph 对外暴露的字段，参考 LangGraph 低阶概念）
class InputState(TypedDict):
    csv_path: str
    target: str
    config_overrides: dict[str, Any]


class OutputState(TypedDict):
    metrics: dict[str, float]
    scorecard: dict[str, Any]
    guardrail: Any
    errors: list[str]
    warnings: list[str]
    step_history: list[str]
    hitl_decision: str
    hitl_note: str