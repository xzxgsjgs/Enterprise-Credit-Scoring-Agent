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
    lgbm_model: Any          # 【V2】LGBM 对照模型（仅对照，不进评分卡）
    lgbm_metrics: dict[str, float]
    # 【V2 模块 3】LR+WOE 的「简化 SHAP」：DataFrame[feature, coef, abs_coef, rank]
    feature_importance: Any
    scorecard: dict[str, Any]
    scored_test: Any
    # 【V2 模块 5】逐变量分值明细（scorecard_ply(only_total_score=False) 的输出）
    scored_detail: Any
    guardrail: Any           # GuardrailResult

    # ====== 【V2 模块 5】cut-off 择优与分档 ======
    cutoff_info: dict[str, Any]      # optimize_cutoff 结果
    score_grades: Any                # DataFrame[score, grade]

    # ====== 【V2 模块 3】Reporter ======
    report_md: str
    report_path: str

    # ====== 累积字段（reducer） ======
    errors: Annotated[list[str], operator.add]
    warnings: Annotated[list[str], operator.add]
    step_history: Annotated[list[str], operator.add]

    # ====== HITL ======
    hitl_decision: str       # "approve" | "reject"
    hitl_note: str

    # ====== 【V2】诊断-重规划回路 ======
    # 最近一次诊断结果：{action, goto, param_patch, reason, source}
    diagnosis: dict[str, Any]
    # 已重试次数（普通字段，覆盖写；绝不能用 add reducer，否则回跳会重复累加）
    retry_count: int
    # 每轮诊断记录，用于报告与人工复核「Agent 试过什么」
    retry_history: Annotated[list[dict], operator.add]

    # ====== 【V2】Critic 多智能体 ======
    critic_findings: Annotated[list[dict], operator.add]
    critic_report: dict[str, Any]

    # ====== 【V2】数据质量前置守门 ======
    # {"passed": bool, "blocks": [...], "warnings": [...], "suggestion": str}
    data_gate: dict[str, Any]

    # ====== 【V2】错误短路 ======
    # 不可恢复节点（load_data/split/woebin/model_train）失败时置 True，直接 END
    critical_error: bool


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
    # 【V2】新增输出
    diagnosis: dict[str, Any]
    retry_count: int
    retry_history: list[dict]
    critic_report: dict[str, Any]
    data_gate: dict[str, Any]
    critical_error: bool
    # 【V2 模块 3/5】输出
    report_md: str
    report_path: str
    feature_importance: Any
    scored_detail: Any
    lgbm_metrics: dict[str, float]
    cutoff_info: dict[str, Any]
    score_grades: Any