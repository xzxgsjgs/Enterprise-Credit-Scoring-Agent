"""agent.nodes — 14 个 LangGraph 节点 + 1 个 HITL 节点

设计原则：
- 1 节点 = 1 个语义步骤（可能调多个工具，如 preprocess_node 调 missing+outlier）
- 每个节点捕获异常 → 写入 errors/step_history（reducer 累积），不让图崩溃
- guardrail_node 评估护栏 → hitl_review_node 用 LangGraph interrupt() 等待人工
- 节点函数签名为 (state: AgentState) -> dict，返回只含变更字段
"""
from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd

from agent.state import AgentState
from tools import (
    build_scorecard,
    compute_psi,
    evaluate_performance,
    explore_data,
    guardrail_check,
    handle_missing,
    handle_outliers,
    load_data,
    model_predict,
    model_train,
    scorecard_ply,
    split_dataset,
    var_filter,
    woebin,
    woebin_ply,
)

logger = logging.getLogger(__name__)


# ============================================================================
# 通用工具
# ============================================================================
def _ok(name: str, **fields: Any) -> dict[str, Any]:
    """构造成功节点的返回 dict。"""
    return {**fields, "step_history": [f"[OK] {name}"]}


def _err(name: str, exc: BaseException) -> dict[str, Any]:
    """构造异常节点的返回 dict（不抛异常，由路由函数决策）。"""
    msg = f"[{name}] {type(exc).__name__}: {exc}"
    logger.exception(msg)
    return {
        "errors": [msg],
        "step_history": [f"[ERR] {name}"],
    }


def safe(name: str):
    """装饰器：捕获节点函数异常，转为 errors/step_history 增量。
    节点函数返回 dict（不含 step_history），装饰器负责追加 step_history。
    """
    def deco(fn):
        def wrapped(state: AgentState) -> dict[str, Any]:
            try:
                result = fn(state) or {}
                return {**result, "step_history": [f"[OK] {name}"]}
            except Exception as e:
                return _err(name, e)
        return wrapped
    return deco


# ============================================================================
# 节点 1: load_data
# ============================================================================
@safe("load_data")
def load_data_node(state: AgentState) -> dict[str, Any]:
    cfg = state.get("config_overrides", {})
    df = load_data(
        state["csv_path"],
        target=state.get("target"),
        encoding=cfg.get("encoding", "utf-8"),
        sep=cfg.get("sep"),
    )
    return {"raw_df": df}


# ============================================================================
# 节点 2: explore_data
# ============================================================================
@safe("explore_data")
def explore_data_node(state: AgentState) -> dict[str, Any]:
    report = explore_data(state["raw_df"], state["target"])
    return {"eda_report": report}


# ============================================================================
# 节点 3: split
# ============================================================================
@safe("split")
def split_node(state: AgentState) -> dict[str, Any]:
    cfg = state.get("config_overrides", {})
    train, test = split_dataset(
        state["raw_df"],
        target=state["target"],
        test_size=cfg.get("test_size", 0.3),
        random_state=cfg.get("random_state", 42),
    )
    target = state["target"]
    # 若 target 是字符串（scorecardpy 的 'good'/'bad'），转 0/1
    if train[target].dtype == object or str(train[target].dtype) == "str":
        unique_vals = train[target].dropna().unique().tolist()
        if set(map(str, unique_vals)) == {"good", "bad"}:
            train[target] = (train[target] == "good").astype(int)
            test[target] = (test[target] == "good").astype(int)
    return {"train_df": train, "test_df": test}


# ============================================================================
# 节点 4: preprocess（missing + outlier，train/test 各自处理）
# ============================================================================
@safe("preprocess")
def preprocess_node(state: AgentState) -> dict[str, Any]:
    cfg = state.get("config_overrides", {})
    strategy = cfg.get("missing_strategy", "median")
    method = cfg.get("outlier_method", "cap")
    sigma = cfg.get("outlier_sigma", 5.0)

    def clean(df: pd.DataFrame) -> pd.DataFrame:
        df = handle_missing(df, strategy=strategy)
        df = handle_outliers(df, method=method, n_sigma=sigma)
        return df

    train = clean(state["train_df"])
    test = clean(state["test_df"])
    return {"train_df": train, "test_df": test}


# ============================================================================
# 节点 5: var_filter
# ============================================================================
@safe("var_filter")
def var_filter_node(state: AgentState) -> dict[str, Any]:
    cfg = state.get("config_overrides", {})
    df, iv_info = var_filter(
        state["train_df"],
        target=state["target"],
        iv_threshold=cfg.get("iv_threshold", 0.02),
        missing_threshold=cfg.get("missing_threshold", 0.5),
        identical_threshold=cfg.get("identical_threshold", 0.95),
    )
    # 保留筛选后变量在 train/test 上
    kept = [c for c in df.columns if c != state["target"]]
    train_kept = df.copy()
    test_kept = state["test_df"][[state["target"]] + kept].copy()
    return {
        "train_df": train_kept,
        "test_df": test_kept,
        "filtered_vars": kept,
    }


# ============================================================================
# 节点 6: woebin
# ============================================================================
@safe("woebin")
def woebin_node(state: AgentState) -> dict[str, Any]:
    cfg = state.get("config_overrides", {})
    bins = woebin(
        state["train_df"],
        target=state["target"],
        max_bins=cfg.get("max_bins", 8),
        min_bin_size=cfg.get("min_bin_size", 0.05),
        method=cfg.get("binning_method", "tree"),
        parallel=False,
    )
    return {"bins": bins}


# ============================================================================
# 节点 7: woebin_apply（train + test 一起 WOE 化）
# ============================================================================
@safe("woebin_apply")
def woebin_apply_node(state: AgentState) -> dict[str, Any]:
    train_woe_raw = woebin_ply(state["train_df"], state["bins"])
    test_woe_raw = woebin_ply(state["test_df"], state["bins"])
    target = state["target"]
    # 只保留数值列（去除 target 与未成功 WOE 化的 categorical 字符串列）
    def numeric_only(df):
        cols = [c for c in df.columns if c != target and df[c].dtype.kind in "iuf"]
        return df[cols].astype(float)
    return {
        "train_woe": numeric_only(train_woe_raw),
        "test_woe": numeric_only(test_woe_raw),
    }


# ============================================================================
# 节点 8: model_train
# ============================================================================
@safe("model_train")
def train_node(state: AgentState) -> dict[str, Any]:
    cfg = state.get("config_overrides", {})
    X = state["train_woe"]
    y = state["train_df"][state["target"]]
    model = model_train(
        X,
        y,
        C=cfg.get("regularization_C", 1.0 / cfg.get("regularization", 0.01)),
        max_iter=cfg.get("max_iter", 1000),
    )
    return {"model": model}


# ============================================================================
# 节点 9: model_predict
# ============================================================================
@safe("model_predict")
def predict_node(state: AgentState) -> dict[str, Any]:
    train_proba = model_predict(state["model"], state["train_woe"], as_prob=True)
    test_proba = model_predict(state["model"], state["test_woe"], as_prob=True)
    return {"train_proba": train_proba, "test_proba": test_proba}


# ============================================================================
# 节点 10: evaluate_performance
# ============================================================================
@safe("evaluate_performance")
def evaluate_node(state: AgentState) -> dict[str, Any]:
    y_train = state["train_df"][state["target"]].values
    y_test = state["test_df"][state["target"]].values
    train_metrics = evaluate_performance(y_train, state["train_proba"])
    test_metrics = evaluate_performance(y_test, state["test_proba"])
    # 用测试集指标作为最终护栏依据
    metrics = {"train": train_metrics, "test": test_metrics, **test_metrics}
    return {"metrics": metrics}


# ============================================================================
# 节点 11: build_scorecard
# ============================================================================
@safe("build_scorecard")
def scorecard_node(state: AgentState) -> dict[str, Any]:
    cfg = state.get("config_overrides", {})
    # xcolumns 为 WOE 化后的列名（不含 target）
    xcolumns = [c for c in state["train_woe"].columns]
    card = build_scorecard(
        bins=state["bins"],
        model=state["model"],
        xcolumns=xcolumns,
        base_score=cfg.get("base_score", 600),
        pdo=cfg.get("pdo", 20),
        base_odds=cfg.get("base_odds", 50),
        double_odds=cfg.get("double_odds", 1),
    )
    return {"scorecard": card}


# ============================================================================
# 节点 12: scorecard_ply（用原始 train/test 而非 WOE 化版本）
# ============================================================================
@safe("scorecard_ply")
def scorecard_apply_node(state: AgentState) -> dict[str, Any]:
    scored = scorecard_ply(state["test_df"], state["scorecard"], only_total_score=True)
    return {"scored_test": scored}


# ============================================================================
# 节点 13: guardrail_check（含 PSI 计算）
# ============================================================================
@safe("guardrail_check")
def guardrail_node(state: AgentState) -> dict[str, Any]:
    cfg = state.get("config_overrides", {})
    metrics = state["metrics"]
    # 计算 PSI：训练期分数分布 vs 测试期分数分布
    psi_val: float | None = None
    if "train" in metrics and "test" in metrics:
        try:
            psi_val = compute_psi(
                expected=np.asarray(state["train_proba"]),
                actual=np.asarray(state["test_proba"]),
            )
        except Exception as e:
            logger.warning("PSI 计算失败: %s", e)

    guardrail = guardrail_check(
        metrics=metrics,
        reference_dist=np.asarray(state["train_proba"]) if psi_val is not None else None,
        current_dist=np.asarray(state["test_proba"]) if psi_val is not None else None,
        min_ks=cfg.get("min_ks", 0.3),
        min_auc=cfg.get("min_auc", 0.7),
        max_psi=cfg.get("max_psi", 0.1),
        require_human_review=cfg.get("require_human_review", True),
    )
    # warnings 同步到 state.warnings（reducer 累积）
    return {"guardrail": guardrail, "warnings": guardrail.warnings}


# ============================================================================
# 节点 14: hitl_review（LangGraph interrupt）
# ============================================================================
def hitl_review_node(state: AgentState) -> dict[str, Any]:
    """HITL 节点：用 interrupt() 暂停等待人工决策。

    决策 payload 格式:
        {"action": "approve" | "reject", "note": "..."}

    resume 后：
        - approve: hitl_decision="approve", hitl_note="..."
        - reject: hitl_decision="reject", hitl_note="..."
    """
    from langgraph.types import interrupt

    gr = state.get("guardrail")
    payload = {
        "question": "护栏未通过，请人工复核模型质量",
        "guardrail": gr.to_dict() if gr and hasattr(gr, "to_dict") else {},
        "recent_warnings": (state.get("warnings") or [])[-5:],
        "recent_history": (state.get("step_history") or [])[-10:],
        "metrics": state.get("metrics", {}),
    }
    decision = interrupt(payload)
    action = decision.get("action", "reject") if isinstance(decision, dict) else "reject"
    note = decision.get("note", "") if isinstance(decision, dict) else str(decision)
    return {
        "hitl_decision": action,
        "hitl_note": note,
        "step_history": [f"[HITL] {action}: {note}"],
    }


# ============================================================================
# 路由函数
# ============================================================================
def route_after_guardrail(state: AgentState) -> Literal["end", "hitl"]:
    """guardrail_node 后路由：
    - 通过 (passed=True) 且 无需人审 → end
    - 需人审 (requires_human_review=True) → hitl
    - 严重违规 (critical) → hitl
    """
    gr = state.get("guardrail")
    if gr is None:
        # 节点失败场景（state 没有 guardrail 字段），强制 HITL
        return "hitl"
    if gr.passed and not gr.requires_human_review:
        return "end"
    # 任何警告/关键违规/需人审 → 触发 HITL
    return "hitl"


def route_after_hitl(state: AgentState) -> Literal["end"]:
    """hitl 之后直接结束。"""
    return "end"