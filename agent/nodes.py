"""agent.nodes — 14 个 LangGraph 节点 + 1 个 HITL 节点

设计原则：
- 1 节点 = 1 个语义步骤（可能调多个工具，如 preprocess_node 调 missing+outlier）
- 每个节点捕获异常 → 写入 errors/step_history（reducer 累积），不让图崩溃
- guardrail_node 评估护栏 → hitl_review_node 用 LangGraph interrupt() 等待人工
- 节点函数签名为 (state: AgentState) -> dict，返回只含变更字段
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from agent.state import AgentState
from tools import (
    build_scorecard,
    check_coef_consistency,
    check_data_quality,
    check_sample_concentration,
    coefficient_importance,
    compute_iv,
    compute_psi,
    compute_vif,
    diagnose_psi_sources,
    evaluate_performance,
    explore_data,
    guardrail_check,
    handle_missing,
    handle_outliers,
    load_data,
    model_predict,
    model_train,
    optimize_cutoff,
    grade_scores,
    render_model_report,
    scorecard_ply,
    split_dataset,
    var_filter,
    woebin,
    woebin_ply,
)

logger = logging.getLogger(__name__)

# 不参与数据质量校验的纯标识列
ID_LIKE_COLS = ("Symbol", "\ufeffSymbol", "ShortName", "EndDate")


# ============================================================================
# 【V2】LLM 调用封装（必须可降级）
# ============================================================================
def _llm_text(prompt: str, default: str = "", temperature: float | None = None) -> str:
    """调 LLM 生成自然语言文本；任何异常（无 key / 超时 / 解析失败）都降级返回 default。

    硬约束 4：LLM 必须可降级，绝不允许因 LLM 失败导致整图失败。
    本函数只用于「自然语言解释与建议」，不参与任何数值计算。
    """
    try:
        from app.llm.client import get_model

        model = get_model(temperature=temperature)
        resp = model.invoke(prompt)
        content = getattr(resp, "content", None)
        return str(content) if content else default
    except Exception as e:  # noqa: BLE001 - 故意宽捕获，LLM 失败必须静默降级
        logger.warning("LLM 调用失败，降级到模板文本: %s", type(e).__name__)
        return default


def _llm_json(prompt: str, temperature: float | None = None) -> dict | None:
    """调 LLM 并解析 JSON；失败返回 None（调用方需自行降级）。"""
    import json

    try:
        from app.llm.client import get_model

        model = get_model(temperature=temperature)
        resp = model.invoke(prompt)
        content = getattr(resp, "content", "") or ""
        # 容忍被 ```json ... ``` 包裹
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        return json.loads(cleaned)
    except Exception as e:  # noqa: BLE001
        logger.warning("LLM JSON 解析失败: %s", type(e).__name__)
        return None


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


def safe(name: str, critical: bool = False):
    """装饰器：捕获节点函数异常，转为 errors/step_history 增量。

    节点函数返回 dict（不含 step_history），装饰器负责追加 step_history。

    Args:
        name: 节点名（写进 step_history）
        critical: True 表示这是不可恢复节点（load_data/split/woebin/model_train）。
            失败时额外置 critical_error=True，供 route_after_critical 短路到 END，
            避免后续节点连环报错。
    """
    def deco(fn):
        def wrapped(state: AgentState) -> dict[str, Any]:
            try:
                result = fn(state) or {}
                return {**result, "step_history": [f"[OK] {name}"]}
            except Exception as e:
                out = _err(name, e)
                if critical:
                    out["critical_error"] = True
                return out
        return wrapped
    return deco


# ============================================================================
# 节点 1: load_data
# ============================================================================
@safe("load_data", critical=True)
def load_data_node(state: AgentState) -> dict[str, Any]:
    cfg = state.get("config_overrides", {})
    df = load_data(
        state["csv_path"],
        target=state.get("target"),
        encoding=cfg.get("encoding", "utf-8-sig"),
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
# 节点 2.5: data_gate（【V2】数据质量前置守门 + 数据过滤）
# 位置：explore_data 之后、split 之前
# ============================================================================
@safe("data_gate")
def data_gate_node(state: AgentState) -> dict[str, Any]:
    """数据质量前置守门：先按 config 过滤数据，再做硬性质量校验。

    过滤（按 config_overrides）：
      - min_year            : 只保留 year_col >= min_year 的行
      - exclude_industries  : 剔除指定行业代码（列 IndustryCode / IndustrySector）
      - exclude_vars        : 剔除指定变量列

    校验（确定性，见 tools.check_data_quality）：
      样本量 / 目标列 / 违约率 / 违约样本数 / 年份覆盖 / 单列缺失率 / 单一值占比

    未通过 → 由 route_after_data_gate 直接 END，不再硬跑后续节点。
    """
    cfg = state.get("config_overrides") or {}
    df = state["raw_df"].copy()
    target = state["target"]

    dropped: list[str] = []

    # ---- 过滤 1: min_year ----
    min_year = cfg.get("min_year")
    year_col = cfg.get("year_col", "year")
    if min_year is not None and year_col in df.columns:
        before = len(df)
        df = df[pd.to_numeric(df[year_col], errors="coerce") >= float(min_year)]
        dropped.append(f"min_year>={min_year}: {before} → {len(df)} 行")

    # ---- 过滤 2: exclude_industries ----
    exclude_ind = cfg.get("exclude_industries") or []
    if exclude_ind:
        for col in ("IndustryCode", "IndustrySector"):
            if col in df.columns:
                before = len(df)
                df = df[~df[col].astype(str).isin([str(x) for x in exclude_ind])]
                if len(df) != before:
                    dropped.append(f"exclude_industries({col}): {before} → {len(df)} 行")

    # ---- 过滤 3: exclude_vars ----
    exclude_vars = cfg.get("exclude_vars") or []
    if exclude_vars:
        real_drop = [c for c in exclude_vars if c in df.columns and c != target]
        if real_drop:
            df = df.drop(columns=real_drop)
            dropped.append(f"exclude_vars: 剔除 {real_drop}")

    # ---- 质量校验（纯确定性）----
    gate_cfg = {k: v for k, v in cfg.items() if k in {
        "min_rows", "min_default_rate", "max_default_rate", "min_years",
        "max_missing_ratio", "max_identical_ratio",
    }}
    result = check_data_quality(
        df,
        target=target,
        year_col=year_col,
        time_split=bool(cfg.get("time_split", False)),
        exclude_cols=list(ID_LIKE_COLS),
        **gate_cfg,
    )

    # ---- suggestion：LLM 撰写，失败降级到模板拼接 ----
    if result["passed"]:
        default_suggestion = "数据质量校验通过。" + (
            f"已执行过滤: {'; '.join(dropped)}。" if dropped else ""
        )
    else:
        default_suggestion = "数据未通过质量守门，阻断原因：" + "；".join(result["blocks"])
        if result["suggest_drop"]:
            default_suggestion += f"。建议剔除列: {result['suggest_drop']}"

    suggestion = _llm_text(
        "你是信用风险建模专家。以下是数据质量校验结果，请用一句话中文给出处理建议（不要输出数字以外的计算，只给建议）：\n"
        f"阻断项: {result['blocks']}\n警告项: {result['warnings']}\n统计: {result['stats']}\n已执行过滤: {dropped}",
        default=default_suggestion,
    )

    gate = {
        "passed": result["passed"],
        "blocks": result["blocks"],
        "warnings": result["warnings"],
        "stats": result["stats"],
        "suggest_drop": result["suggest_drop"],
        "applied_filters": dropped,
        "suggestion": suggestion,
    }

    warns = [f"[data_gate] {w}" for w in result["warnings"]]
    if not result["passed"]:
        warns += [f"[data_gate][BLOCK] {b}" for b in result["blocks"]]

    return {"raw_df": df, "data_gate": gate, "warnings": warns}


# ============================================================================
# 节点 3: split
# ============================================================================
@safe("split", critical=True)
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
    target = state["target"]

    # 【V2】自愈剔除：诊断回路追加的 exclude_vars 在这里落地
    exclude_vars = [c for c in (cfg.get("exclude_vars") or []) if c != target]
    dropped = [c for c in exclude_vars if c in state["train_df"].columns or c in state["test_df"].columns]

    def clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=[c for c in exclude_vars if c in df.columns])
        # 关键：target 与纯标识列绝不能参与缺失值/异常值处理。
        # 反例：违约率 3% 时 mu+5σ ≈ 0.88 < 1，cap 会把 1 压成 0.88，
        # 目标列出现 3 个取值 → scorecardpy 的 var_filter/woebin 直接报
        # "the length of unique values in y != 2"，后续节点全线崩溃。
        feature_cols = [
            c for c in df.columns
            if c != target and c not in ID_LIKE_COLS and pd.api.types.is_numeric_dtype(df[c])
        ]
        df = handle_missing(df, strategy=strategy, columns=feature_cols)
        df = handle_outliers(df, method=method, n_sigma=sigma, columns=feature_cols)
        return df

    train = clean(state["train_df"])
    test = clean(state["test_df"])

    out: dict[str, Any] = {"train_df": train, "test_df": test}
    if dropped:
        out["warnings"] = [f"[preprocess] 已剔除漂移/禁用变量: {dropped}"]
    return out


def _drop_excluded(df: pd.DataFrame, exclude_vars: list[str], target: str) -> pd.DataFrame:
    cols = [c for c in exclude_vars if c in df.columns and c != target]
    return df.drop(columns=cols) if cols else df


# ============================================================================
# 节点 5: var_filter
# ============================================================================
# var_filter 保留变量数下限：低于此值说明过滤条件过严（多由自愈调参引起），需自动放宽
MIN_KEPT_VARS = 5
DEFAULT_IV_THRESHOLD = 0.02


@safe("var_filter")
def var_filter_node(state: AgentState) -> dict[str, Any]:
    cfg = state.get("config_overrides", {})
    target = state["target"]
    # 【V2】诊断回路回跳到本节点时，exclude_vars 必须在这一步生效
    train_src = _drop_excluded(state["train_df"], list(cfg.get("exclude_vars") or []), target)
    test_src = _drop_excluded(state["test_df"], list(cfg.get("exclude_vars") or []), target)

    iv_t = float(cfg.get("iv_threshold", DEFAULT_IV_THRESHOLD))
    missing_t = float(cfg.get("missing_threshold", 0.5))
    identical_t = float(cfg.get("identical_threshold", 0.95))

    df, iv_info = _var_filter_with_relax(train_src, target, iv_t, missing_t, identical_t)

    # 保留筛选后变量在 train/test 上
    kept = [c for c in df.columns if c != target]
    train_kept = df.copy()
    test_kept = test_src[[target] + kept].copy()
    return {
        "train_df": train_kept,
        "test_df": test_kept,
        "filtered_vars": kept,
    }


def _var_filter_with_relax(
    train_src: pd.DataFrame,
    target: str,
    iv_t: float,
    missing_t: float,
    identical_t: float,
) -> tuple[pd.DataFrame, Any]:
    """带自动放宽的 var_filter：先按请求阈值筛，变量过少则逐级放宽。

    必要性：诊断回路可能把 iv_threshold 调得很高（例如 LLM 建议 0.15），
    导致变量全被剔除——scorecardpy 0.1.9.7 在 pandas 3.0 下遇到空结果会抛
    `reset_index() got an unexpected keyword argument 'name'`，整条链路崩掉。
    这里做确定性兜底：0.15 → 0.02(fallback) → 自研 IV 排序取 top30。
    """
    try:
        df, iv_info = var_filter(
            train_src, target=target,
            iv_threshold=iv_t, missing_threshold=missing_t, identical_threshold=identical_t,
        )
    except Exception as e:  # noqa: BLE001 - scorecardpy 对空结果抛 pandas 3.0 兼容异常
        logger.warning("var_filter(%.3f) 失败(%s)，回退到默认阈值 %.3f", iv_t, e, DEFAULT_IV_THRESHOLD)
        df, iv_info = var_filter(
            train_src, target=target,
            iv_threshold=DEFAULT_IV_THRESHOLD,
            missing_threshold=missing_t, identical_threshold=identical_t,
        )

    if len([c for c in df.columns if c != target]) >= MIN_KEPT_VARS:
        return df, iv_info

    # 变量过少 → 自研 IV 排序取 top30（确定性，不依赖 scorecardpy 的空结果路径）
    logger.warning("保留变量 %d 个 < %d，改用自研 IV 排序兜底", df.shape[1] - 1, MIN_KEPT_VARS)
    scored: list[tuple[str, float]] = []
    for c in train_src.columns:
        if c == target or c in ID_LIKE_COLS:
            continue
        try:
            iv = compute_iv(train_src, c, target)
        except Exception:  # noqa: BLE001 - 单变量算失败就跳过
            continue
        if iv == iv:  # 非 NaN
            scored.append((c, float(iv)))
    scored.sort(key=lambda x: x[1], reverse=True)
    keep = [c for c, _ in scored[:30]] or [c for c, _ in scored]
    if keep:
        return train_src[[target] + keep], None
    return df, iv_info


# ============================================================================
# 节点 6: woebin
# ============================================================================
@safe("woebin", critical=True)
def woebin_node(state: AgentState) -> dict[str, Any]:
    cfg = state.get("config_overrides", {})
    target = state["target"]
    bins = woebin(
        state["train_df"],
        target=target,
        max_bins=cfg.get("max_bins", 8),
        min_bin_size=cfg.get("min_bin_size", 0.05),
        method=cfg.get("binning_method", "tree"),
        parallel=False,
    )

    # 【V2】自愈：换分箱算法（尤其是 chimerge 遇到类别型变量）可能导致全部变量失败，
    # 进而 woebin_apply → model_train 连环崩溃。这里确定性回退到 tree 再试一次。
    if not bins:
        logger.warning("分箱结果为空（method=%s），自动回退 tree 重试", cfg.get("binning_method", "tree"))
        bins = woebin(
            state["train_df"],
            target=target,
            max_bins=cfg.get("max_bins", 8),
            min_bin_size=cfg.get("min_bin_size", 0.05),
            method="tree",
            parallel=False,
        )

    if not bins:
        raise RuntimeError(
            "woebin 对所有变量均失败（含 tree 回退）；请检查是否存在大量类别型变量或常量列"
        )

    out: dict[str, Any] = {"bins": bins}
    if cfg.get("binning_method") not in (None, "tree") and not bins:
        out["warnings"] = ["[woebin] 分箱算法不适用于当前数据，已回退 tree"]
    return out


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
@safe("model_train", critical=True)
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

    out: dict[str, Any] = {"model": model}

    # 【V2 模块 3】特征重要性：LR+WOE 的「简化 SHAP」，用 |coef| 排序（确定性）
    try:
        out["feature_importance"] = coefficient_importance(model, list(X.columns))
    except Exception as e:  # noqa: BLE001 - 重要性是增值信息，失败不得中断主流程
        logger.warning("特征重要性计算失败（报告章节将为空）: %s", e)

    # 【V2】LGBM 对照模型：仅作能力对照与诊断依据，不替换评分卡主模型
    # （评分卡刻度依赖 LR 的 WOE 系数，替换为树模型会导致 build_scorecard 失效）
    if cfg.get("use_lgbm"):
        try:
            from lightgbm import LGBMClassifier

            lgbm = LGBMClassifier(
                n_estimators=int(cfg.get("lgbm_n_estimators", 200)),
                learning_rate=float(cfg.get("lgbm_lr", 0.05)),
                num_leaves=int(cfg.get("lgbm_num_leaves", 15)),
                random_state=int(cfg.get("random_state", 42)),
                verbose=-1,
            )
            lgbm.fit(X, y)
            lgbm_proba = lgbm.predict_proba(state["test_woe"])[:, 1]
            lgbm_metrics = evaluate_performance(state["test_df"][state["target"]].values, lgbm_proba)
            out["lgbm_model"] = lgbm
            out["lgbm_metrics"] = lgbm_metrics
            out["warnings"] = [
                f"[model_train] LGBM 对照: AUC={lgbm_metrics.get('auc'):.3f} "
                f"KS={lgbm_metrics.get('ks'):.3f}（逻辑回归见 metrics）"
            ]
        except Exception as e:  # noqa: BLE001 - 未装 lightgbm 必须优雅降级
            out["warnings"] = [f"[model_train] LGBM 对照训练失败({type(e).__name__})，已忽略，仍用逻辑回归"]

    return out


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
# 【V2 模块 3】Critic：规则审计 + LLM 可降级复核
# ============================================================================
# 说明：Critic **不阻塞主流程**——任何一项检查失败只写 warning，
# 发现的 issues 也不参与 guardrail 路由，仅供 reporter 汇总与人工参考。

_SEVERITY_ORDER = {"high": 3, "mid": 2, "low": 1}


def _severity_rank(sev: str) -> int:
    return _SEVERITY_ORDER.get(str(sev).lower(), 0)


def _match_known_item(raw: Any, known_items: set[str]) -> str | None:
    """把 LLM 回传的 item 文本对齐回规则清单，命中则返回清单原文。

    LLM（尤其小模型）常常把整行复制回来，例如 prompt 里写的是
    `- [8] item=变量 X 系数符号... | 规则初判=low`，它返回的 item 就是这一整行。
    严格相等匹配会把它判成「清单外」→ 整轮 LLM 复核作废、source 永远退化成 rule。
    因此按三级降级匹配：精确 → 剥离 `[n] item=` 前缀 / 引号 → 子串包含。
    """
    text = str(raw).strip()
    if not text:
        return None
    if text in known_items:
        return text

    import re as _re

    # 1) 剥离 "[8] item=" / "item:" / "【8】" 等编号与键名前缀、收尾引号
    cand = text
    cand = _re.sub(r"^[\[［【]?\s*\d+\s*[\]】］]?\s*", "", cand)
    cand = _re.sub(r"^item\s*[:=＝]\s*", "", cand, flags=_re.IGNORECASE)
    cand = cand.strip().strip('"\'`').strip()
    cand = _re.split(r"\s*\|\s*", cand)[0].strip()      # 去掉尾部的 "| 规则初判=low"
    if cand in known_items:
        return cand

    # 2) 子串包含：只要能唯一命中清单中的一项，就认它
    hits = [k for k in known_items if k and (k in text or text in k)]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:                                    # 有歧义时取最长匹配
        return max(hits, key=len)
    return None


def _run_rule_audits(state: AgentState, cfg: dict[str, Any]) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    """跑三项确定性审计，返回 (findings, tables)。单项失败不影响其余项。"""
    findings: list[dict[str, Any]] = []
    tables: dict[str, pd.DataFrame] = {}

    top_n = int(cfg.get("critic_top_features", 10))
    vif_fail = float(cfg.get("vif_threshold", 10.0))
    conc_threshold = float(cfg.get("concentration_threshold", 0.5))
    conc_cols = cfg.get("concentration_cols") or None

    # ---- 1) VIF 多重共线性 ----
    try:
        X = state.get("train_woe")
        if X is not None and hasattr(X, "columns"):
            X_num = X.drop(columns=[state["target"]], errors="ignore") if state.get("target") in getattr(X, "columns", []) else X
            vif = compute_vif(X_num, fail=vif_fail)
            tables["vif"] = vif
            for _, r in vif.head(top_n).iterrows():
                if r["flag"] in ("fail", "warn"):
                    findings.append({
                        "check": "vif",
                        "item": f"变量 {r['variable']} 存在多重共线性（VIF={r['vif']:.2f}）",
                        "severity": "high" if r["flag"] == "fail" else "mid",
                        "suggestion": "剔除该变量或改用 stepwise / 主成分降维后重训",
                        "detail": float(r["vif"]) if np.isfinite(r["vif"]) else None,
                    })
    except Exception as e:  # noqa: BLE001
        logger.warning("[critic] VIF 审计失败（已跳过）: %s", e)

    # ---- 2) 系数符号 vs 业务方向 ----
    try:
        model = state.get("model")
        bins = state.get("bins")
        xcols = list(state["train_woe"].columns) if state.get("train_woe") is not None else None
        if xcols and state.get("target") in xcols:
            xcols.remove(state["target"])
        if model is not None and bins:
            coef_df = check_coef_consistency(bins, model, xcols)
            tables["coef"] = coef_df
            importance = state.get("feature_importance")
            strong = set()
            if importance is not None and "feature" in getattr(importance, "columns", []):
                strong = set(importance.head(top_n)["feature"].astype(str).str.replace(r"_woe$", "", regex=True))
            for _, r in coef_df.iterrows():
                if not r["ok"]:
                    sev = "high" if r["variable"] in strong else "low"
                    findings.append({
                        "check": "coef_sign",
                        "item": f"变量 {r['variable']} 系数符号（{int(r['actual_sign']):+d}）"
                                f"与 WOE 风险方向（{int(r['expected_sign']):+d}）相反",
                        "severity": sev,
                        "suggestion": "多为共线性导致的符号翻转；若该变量是核心业务变量需人工复核方向",
                        "detail": float(r["coef"]),
                    })
    except Exception as e:  # noqa: BLE001
        logger.warning("[critic] 系数符号审计失败（已跳过）: %s", e)

    # ---- 3) 样本集中度 ----
    try:
        df = state.get("raw_df") if state.get("raw_df") is not None else state.get("train_df")
        if df is not None:
            # 标签列天然极度不平衡（违约率 ~3%），算集中度只会产出「target 取值 0 占 97%」
            # 这类无意义的结论；ID / 年份同理，一律排除。
            skip_cols = [c for c in (state.get("target"), cfg.get("id_col"), cfg.get("year_col")) if c]
            conc = check_sample_concentration(
                df, cols=conc_cols, threshold=conc_threshold, exclude=skip_cols,
            )
            tables["concentration"] = conc
            for _, r in conc.head(top_n).iterrows():
                if r["flag"] == "fail":
                    findings.append({
                        "check": "concentration",
                        "item": f"列 {r['column']} 的取值 {r['top_value']} 占比 {r['share']:.1%}",
                        "severity": "mid",
                        "suggestion": "样本结构偏斜，模型外推到该维度其他取值时可靠性下降",
                        "detail": float(r["share"]),
                    })
    except Exception as e:  # noqa: BLE001
        logger.warning("[critic] 样本集中度审计失败（已跳过）: %s", e)

    return findings, tables


def _llm_review_findings(
    findings: list[dict[str, Any]],
    state: AgentState,
    cfg: dict[str, Any],
) -> dict[str, Any] | None:
    """把规则发现交给 LLM 做定性复核；失败返回 None（调用方降级到纯规则结论）。

    硬约束：LLM 只能重排/改写 issues 的文字，**不得生成任何数值**；
    返回 JSON 里若出现新的 item（不在规则清单里）会被丢弃。
    """
    if not findings:
        return {"passed": True, "issues": [], "source": "rule"}
    if not bool(cfg.get("allow_llm_critic", True)):
        return None

    known_items = {f["item"] for f in findings}
    metrics = state.get("metrics") or {}
    test_m = metrics.get("test", {}) if isinstance(metrics, dict) else {}
    importance = state.get("feature_importance")
    top_features: list[str] = []
    if importance is not None and "feature" in getattr(importance, "columns", []):
        top_features = list(importance.head(10)["feature"].astype(str))

    prompt = (
        "你是信用风险模型评审（Critic）。下面是一组由确定性审计规则发现的问题，"
        "请你判断它们对模型可用性的实际影响，并给出处置建议。\n\n"
        "严格要求：\n"
        "1. 只输出 JSON：{\"issues\": [{\"item\": \"<必须是下面清单里的原文>\", "
        "\"severity\": \"high|mid|low\", \"suggestion\": \"<中文，不要出现任何数字>\"}]}\n"
        "2. item 必须原样取自下面的清单，不得改写或新增；你可以少列，但不要多列。\n"
        "3. severity 只能取 high/mid/low；suggestion 里严禁出现数字、百分比、指标值。\n\n"
        "【发现的规则清单】\n"
        + "\n".join(f"- [{i}] item={f['item']} | 规则初判={f['severity']}" for i, f in enumerate(findings, 1))
        + "\n\n【背景（仅供理解，禁止复述其中的数值）】\n"
        "- 测试集 KS/AUC 见报告性能章节\n"
        f"- 入模变量（前若干）：{', '.join(top_features[:10]) or '见报告'}\n"
    )

    raw = _llm_json(prompt, temperature=0)
    if not isinstance(raw, dict):
        return None

    issues_raw = raw.get("issues")
    if not isinstance(issues_raw, list):
        return None

    cleaned: list[dict[str, Any]] = []
    for it in issues_raw:
        if not isinstance(it, dict):
            continue
        item = _match_known_item(it.get("item", ""), known_items)
        if item is None:            # 拦截 LLM 自己编的条目
            logger.warning("[critic] LLM 返回了清单外的 issue，已丢弃: %s", str(it.get('item'))[:60])
            continue
        sev = str(it.get("severity", "low")).lower()
        if sev not in _SEVERITY_ORDER:
            sev = next((f["severity"] for f in findings if f["item"] == item), "low")
        suggestion = str(it.get("suggestion", "")).strip()
        import re as _re
        if _re.search(r"\d", suggestion):   # 建议文案里出现数字 → 丢弃这段建议，保留规则原文
            logger.warning("[critic] LLM suggestion 含数字，已回退规则建议")
            suggestion = next((f["suggestion"] for f in findings if f["item"] == item), "")
        cleaned.append({"item": item, "severity": sev, "suggestion": suggestion})

    if not cleaned:
        return None
    return {"issues": cleaned, "source": "llm"}


@safe("critic")
def critic_node(state: AgentState) -> dict[str, Any]:
    """模型/数据侧的确定性审计 + LLM 定性复核（不阻塞主流程）。

    三项检查（全部由 tools/audit_tools.py 计算）：
        1. `compute_vif`              —— 多重共线性
        2. `check_coef_consistency`   —— 系数符号 vs WOE 风险方向
        3. `check_sample_concentration` —— 类别取值集中度

    Returns:
        {"critic_findings", "critic_report", "warnings", "step_history"}
    """
    cfg = dict(state.get("config_overrides") or {})
    out: dict[str, Any] = {"step_history": ["[OK] critic"]}

    findings, tables = _run_rule_audits(state, cfg)

    review = _llm_review_findings(findings, state, cfg)
    if review is not None and review.get("issues"):
        # LLM 复核成功：保留它给出的 severity/suggestion，item 仍锚定规则原文
        by_item = {f["item"]: f for f in findings}
        merged: list[dict[str, Any]] = []
        for it in review["issues"]:
            base = by_item.get(it["item"], {})
            merged.append({
                "check": base.get("check", "unknown"),
                "item": it["item"],
                "severity": it["severity"],
                "suggestion": it["suggestion"] or base.get("suggestion", ""),
                "detail": base.get("detail"),
            })
        issues = merged
        source = "rule+llm"
    else:
        issues = [
            {"check": f["check"], "item": f["item"], "severity": f["severity"],
             "suggestion": f["suggestion"], "detail": f.get("detail")}
            for f in findings
        ]
        source = "rule"

    issues.sort(key=lambda x: -_severity_rank(x["severity"]))
    has_high = any(_severity_rank(i["severity"]) >= 3 for i in issues)
    has_mid = any(_severity_rank(i["severity"]) == 2 for i in issues)

    report = {
        "passed": not (has_high or has_mid),
        "issues": issues,
        "source": source,
        "n_issues": len(issues),
        "n_high": sum(1 for i in issues if i["severity"] == "high"),
        "n_mid": sum(1 for i in issues if i["severity"] == "mid"),
        "n_low": sum(1 for i in issues if i["severity"] == "low"),
        "vif_fail": int((tables["vif"]["flag"] == "fail").sum()) if "vif" in tables else 0,
        "vif_warn": int((tables["vif"]["flag"] == "warn").sum()) if "vif" in tables else 0,
        "coef_flip": int((~tables["coef"]["ok"]).sum()) if "coef" in tables else 0,
        "concentration_fail": int((tables["concentration"]["flag"] == "fail").sum()) if "concentration" in tables else 0,
    }
    out["critic_report"] = report
    if issues:
        out["critic_findings"] = issues

    if has_high:
        out["warnings"] = [
            f"[critic] 发现 {report['n_high']} 项高危问题，需在报告中人工确认"
        ]
    logger.info(
        "[critic] 审计完成: %d 项（high=%d mid=%d low=%d，来源=%s）",
        len(issues), report["n_high"], report["n_mid"], report["n_low"], source,
    )
    return out


# ============================================================================
# 节点 12: scorecard_ply（用原始 train/test 而非 WOE 化版本）
# ============================================================================
@safe("scorecard_ply")
def scorecard_apply_node(state: AgentState) -> dict[str, Any]:
    scored = scorecard_ply(state["test_df"], state["scorecard"], only_total_score=True)
    out: dict[str, Any] = {"scored_test": scored}
    # 【V2 模块 5】逐变量分值明细，用于 cut-off 解释与《拒贷理由书》
    # 失败不影响主流程（某些类别变量缺失时 scorecardpy 可能报错）
    try:
        detail = scorecard_ply(state["test_df"], state["scorecard"], only_total_score=False)
        out["scored_detail"] = detail
    except Exception as e:  # noqa: BLE001
        logger.warning("逐变量分值明细生成失败（不影响主流程）: %s", e)
    return out


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
# 【V2 模块 3 + 5】reporter_node（报告生成 + cut-off 择优 + 分档）
# ============================================================================
# 同样不加 @safe：需要读 config 里的 thread_id（第二参数），且报告失败绝不能中断主流程。
def reporter_node(
    state: AgentState,
    config: Optional[RunnableConfig] = None,   # noqa: UP045 - LangGraph 需识别此注解才会注入 config
) -> dict[str, Any]:
    """汇总全流程结果 → cut-off 择优 → 渲染《模型开发报告》→ 落盘 Markdown。

    数值全部来自 tools/ 的确定性函数；LLM 只写 overview/performance/risk 三段
    「分析」文字，失败则保留占位符（见 tools.report_tools.DEFAULT_PLACEHOLDER）。

    Args:
        state: AgentState
        config: LangGraph RunnableConfig，用于取 thread_id

    Returns:
        {"report_md", "report_path", "cutoff_info", "score_grades", "step_history", ...}
    """
    cfg = dict(state.get("config_overrides") or {})
    thread_id = "unknown"
    try:
        thread_id = ((config or {}).get("configurable") or {}).get("thread_id", "unknown")
    except Exception:  # noqa: BLE001
        pass

    out: dict[str, Any] = {"step_history": ["[OK] reporter"]}

    # ---- 1) cut-off 择优 + 分档（确定性）----
    cutoff_info: dict[str, Any] | None = None
    grades = None
    scored = state.get("scored_test")
    try:
        if scored is not None and "score" in getattr(scored, "columns", []) \
                and state.get("test_df") is not None:
            y_test = state["test_df"][state["target"]]
            cutoff_info = optimize_cutoff(
                y_test,
                scored["score"],
                method=cfg.get("cutoff_method", "ks"),
                target=cfg.get("cutoff_target"),
            )
            grades = grade_scores(scored["score"])
    except Exception as e:  # noqa: BLE001
        logger.warning("cut-off 择优失败（报告将缺失该章节）: %s", e)

    if cutoff_info:
        out["cutoff_info"] = cutoff_info
    if grades is not None:
        out["score_grades"] = grades

    # ---- 2) 组装上下文 ----
    # PSI 是在 guardrail 里算的，回填进 test metrics 让性能表不至于显示 "-"
    metrics_ctx = state.get("metrics") or {}
    if isinstance(metrics_ctx, dict) and isinstance(metrics_ctx.get("test"), dict):
        gr_metrics = getattr(state.get("guardrail"), "metrics", {}) or {}
        test_copy = dict(metrics_ctx["test"])
        test_copy["psi"] = test_copy.get("psi", gr_metrics.get("psi"))
        metrics_ctx = {**metrics_ctx, "test": test_copy}

    ctx: dict[str, Any] = {
        "thread_id": thread_id,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "csv_path": state.get("csv_path"),
        "target": state.get("target"),
        "config_overrides": cfg,
        "eda": state.get("eda_report"),
        "data_gate": state.get("data_gate"),
        "metrics": metrics_ctx,
        "scorecard": state.get("scorecard"),
        "feature_importance": state.get("feature_importance"),
        "guardrail": state.get("guardrail"),
        "critic_report": state.get("critic_report"),
        "retry_history": state.get("retry_history"),
        "diagnosis": state.get("diagnosis"),
        "cv_per_fold": state.get("cv_per_fold"),
        "cv_mean": state.get("cv_mean"),
        "bins": state.get("bins"),
        "scored_test": scored,
        "cutoff": cutoff_info,
        "grades": grades,
    }

    # ---- 3) LLM 撰写分析段落（可降级）----
    ctx["analysis"] = _llm_analysis_sections(state, ctx)

    # ---- 4) 渲染 & 落盘 ----
    try:
        md = render_model_report(ctx)
        out["report_md"] = md
    except Exception as e:  # noqa: BLE001
        logger.exception("报告渲染失败: %s", e)
        out["errors"] = [f"[reporter] 报告渲染失败: {type(e).__name__}: {e}"]
        return out

    if not bool(cfg.get("generate_report", True)):
        logger.info("generate_report=False，跳过报告落盘")
        return out

    try:
        report_dir = Path(cfg.get("report_dir", "reports"))
        if not report_dir.is_absolute():
            # 相对目录统一挂到项目根（credit_agent/）下，避免受工作目录影响
            report_dir = Path(__file__).resolve().parents[1] / report_dir
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / f"model_report_{thread_id}.md"
        path.write_text(md, encoding="utf-8")
        out["report_path"] = str(path)
        logger.info("模型报告已生成: %s", path)
    except Exception as e:  # noqa: BLE001
        logger.warning("报告落盘失败: %s", e)
        out["warnings"] = [f"[reporter] 报告落盘失败: {e}"]

    return out


_PROMPT_TEMPLATE = """你是信用风险建模负责人，为一份企业信用评分卡模型报告撰写三段「分析」文字。

硬性要求：
1. 输出且仅输出三段，每段第一行必须是下面这行标记之一：
[OVERVIEW]
[PERFORMANCE]
[RISK]
2. 每段 2-4 句中文，纯定性描述，不要写 markdown 标题或列表符号。
3. **严禁出现任何数字、百分比、金额或指标数值**——报告的数值已由程序生成，
   你写的数字会被自动检测并丢弃，导致整段作废。

参考结构（照抄结构，不要照抄内容）：
[OVERVIEW]
面向企业未来一年违约风险排序，目标是提供可解释的信用评估依据。通过多维度财务数据整合，
提升对不同信用等级主体的区分能力。
[PERFORMANCE]
模型区分能力与稳定性处于合理水平，在不同年份样本上表现相对一致。整体表现符合建模预期。
[RISK]
主要局限来自标签口径与数据时效性，外部经济环境变化也会带来影响。需持续监控数据质量与人群迁移。

【背景事实，仅供理解，禁止在正文中复述其中的任何数值】
- 目标列：{target}
- 测试集 KS / AUC / Gini：见报告第六章表格，正文不要复述
- 护栏结论：{guardrail_text}
- cut-off 择优结果：见第九章
- Agent 自动重规划轮数：{n_retry}
- 建模变量示例：{vars_sample}
"""

# 允许 LLM 少量排版噪声：`**[OVERVIEW]**` / `## OVERVIEW：` 等都能识别
_TAG_RE = r"\[?\s*{tag}\s*\]?\s*[:：]?"
_DIGIT_RE = re.compile(r"\d")


def _parse_tags(raw: str) -> dict[str, str]:
    """从 LLM 原文里按 OVERVIEW/PERFORMANCE/RISK 三个标记切段。

    用「定位标记再切片」而非贪婪正则，容忍标记周围有 `**`、`:`、`## ` 等噪声，
    也容忍三段顺序错乱。
    """
    hits: list[tuple[str, int, int]] = []
    for key, tag in (("overview", "OVERVIEW"), ("performance", "PERFORMANCE"), ("risk", "RISK")):
        m = re.search(_TAG_RE.format(tag=tag), raw, re.IGNORECASE)
        if m:
            hits.append((key, m.start(), m.end()))
    if not hits:
        return {}
    hits.sort(key=lambda x: x[1])
    out: dict[str, str] = {}
    for i, (key, _start, end_head) in enumerate(hits):
        stop = hits[i + 1][1] if i + 1 < len(hits) else len(raw)
        # 先去掉 markdown 噪声字符（`**` / `##` / 冒号），再统一清空白，顺序不能反
        body = raw[end_head:stop].strip("*#>-—:： \t").strip()
        # 【硬约束 1】LLM 段落里一旦出现数字（多为幻觉指标）→ 整段作废
        if _DIGIT_RE.search(body):
            logger.warning("[reporter] LLM 段落 %s 检出数字，已丢弃（禁止 LLM 生成数值）", key)
            continue
        if body:
            out[key] = body
    return out


def _llm_analysis_sections(state: AgentState, ctx: dict[str, Any]) -> dict[str, str]:
    """让 LLM 撰写三段「分析与结论」；失败返回占位符 dict。

    三重保障（对应硬约束 1/3/4）：
    1. prompt 强制 `[OVERVIEW]/[PERFORMANCE]/[RISK]` 标记 + 示例 + 明令禁止写数字；
    2. `_parse_tags` 容忍排版噪声；任一检出数字 → 该段作废旧控风险；
    3. 全部失败（无 key / 超时 / 无标记） → 三段都用占位符，报告其余章节仍完整。
    """
    from tools.report_tools import DEFAULT_PLACEHOLDER

    placeholder = {
        "overview": DEFAULT_PLACEHOLDER,
        "performance": DEFAULT_PLACEHOLDER,
        "risk": DEFAULT_PLACEHOLDER,
    }
    cfg = ctx.get("config_overrides") or {}
    if not bool(cfg.get("allow_llm_report", True)):
        return placeholder

    metrics = ctx.get("metrics") or {}
    test_m = metrics.get("test", {}) if isinstance(metrics, dict) else {}
    gr = ctx.get("guardrail")
    cutoff = ctx.get("cutoff") or {}
    history = ctx.get("retry_history") or []

    prompt = _PROMPT_TEMPLATE.format(
        target=ctx.get("target"),
        guardrail_text=(
            f"通过={getattr(gr, 'passed', None)}，告警={len(getattr(gr, 'warnings', []) or [])} 条"
        ),
        n_retry=len(history),
        vars_sample="、".join(list((ctx.get("scorecard") or {}).keys())[:8]) or "见第七章",
    )
    # 上下文里的原始数值只写进 logger，绝不进 prompt（避免诱导 LLM 复述数字）
    logger.info(
        "reporter LLM 上下文: ks=%s auc=%s cutoff=%s", test_m.get("ks"), test_m.get("auc"),
        cutoff.get("cutoff"),
    )

    best: dict[str, str] = {}
    for attempt in range(2):
        raw = _llm_text(prompt, default="", temperature=0)
        parsed = _parse_tags(raw) if raw else {}
        if len(parsed) > len(best):
            best = parsed
        if len(best) == 3:
            break
        if attempt == 0:
            logger.info("[reporter] LLM 段落不完整(%d/3)，重试一次", len(best))

    out = dict(placeholder)
    out.update(best)
    return out


# ============================================================================
# 【V2 模块 1+2】诊断-重规划回路（diagnose_node）
# ============================================================================
# 说明：diagnose_node **不能**用 @safe 装饰——@safe 会把返回值 dict 再包一层
# step_history，导致 Command 被吞掉，图无法回跳。因此内部自建 try/except 兜底。

# ---- 默认白名单（config/agent.* 未加载时的兜底，与 default_config.yaml 一致）----
DEFAULT_ALLOWED_GOTO: tuple[str, ...] = (
    "preprocess", "var_filter", "woebin", "model_train", "build_scorecard",
)
DEFAULT_ALLOWED_PARAMS: tuple[str, ...] = (
    "iv_threshold", "missing_threshold", "identical_threshold", "max_bins",
    "min_bin_size", "binning_method", "missing_strategy", "outlier_method",
    "outlier_sigma", "regularization", "test_size", "use_lgbm",
    "exclude_vars", "min_ks", "min_auc", "max_psi",
)
DEFAULT_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "iv_threshold": (0.0, 0.2),
    "missing_threshold": (0.1, 0.9),
    "identical_threshold": (0.5, 1.0),
    "max_bins": (3, 12),
    "min_bin_size": (0.01, 0.2),
    "outlier_sigma": (2.0, 8.0),
    "regularization": (0.001, 1.0),
    "test_size": (0.1, 0.5),
    "min_ks": (0.05, 0.6),
    "min_auc": (0.5, 0.95),
    "max_psi": (0.05, 0.5),
}
DEFAULT_BINNING_CHOICES: tuple[str, ...] = ("tree", "chimerge", "quantile", "equal")
DEFAULT_MISSING_CHOICES: tuple[str, ...] = ("mean", "median", "mode", "constant", "drop")
BOOL_PARAMS = ("use_lgbm",)
INT_PARAMS = ("max_bins",)
LIST_PARAMS = ("exclude_vars",)

# CSI 集中度判定：最高 CSI >= 该阈值 且 >= 中位数的 CSI_FACTOR 倍 → 漂移集中在少数变量
CSI_CONCENTRATION_MIN = 0.1
CSI_FACTOR = 2.0

# 规则急救阶梯（自上而下匹配，第一次命中即返回；已试过的组合自动跳过）
def _rule_action_plan(state: AgentState, cfg: dict[str, Any]) -> list[tuple[str, dict, str]]:
    """按当前指标生成候选处置方案列表（确定性），调用方按顺序挑选未试过的。

    每个元素 = (goto, param_patch, reason)
    """
    metrics = state.get("metrics") or {}
    gr = state.get("guardrail")
    gr_metrics = getattr(gr, "metrics", None) or {}
    guard_warnings = list(getattr(gr, "warnings", None) or [])

    ks = _to_float(metrics.get("ks"))
    auc = _to_float(metrics.get("auc"))
    psi = _to_float(gr_metrics.get("psi"))
    min_ks = _to_float(cfg.get("min_ks"), 0.3)
    min_auc = _to_float(cfg.get("min_auc"), 0.7)

    plan: list[tuple[str, dict, str]] = []

    # ---- R1/R2: PSI 漂移（先定位 CSI 来源）----
    if psi > 0.25:
        csi_top = _csi_top_variables(state)
        if csi_top:
            top_var, top_csi, second_csi, median_csi = csi_top
            concentrated = (
                top_csi >= CSI_CONCENTRATION_MIN
                and top_csi >= CSI_FACTOR * max(median_csi, 1e-6)
            )
            if concentrated:
                already = list(cfg.get("exclude_vars") or [])
                new_exclude = already + [v for v in [top_var] if v not in already]
                plan.append((
                    "var_filter",
                    {"exclude_vars": new_exclude},
                    f"PSI={psi:.3f} 超限，CSI 集中在 {top_var}(CSI={top_csi:.3f}，次高 {second_csi:.3f})，剔除该漂移变量重跑",
                ))
            else:
                plan.append((
                    "woebin",
                    {"max_bins": 6},
                    f"PSI={psi:.3f} 超限但 CSI 分散（最高 {top_csi:.3f} < 集中度阈值），粗化分箱到 6 箱降低箱间波动",
                ))
        else:
            plan.append((
                "woebin",
                {"max_bins": 6},
                f"PSI={psi:.3f} 超限且无法定位 CSI 来源，粗化分箱到 6 箱",
            ))

    # ---- R3: KS 严重不足 ----
    # chimerge 不支持类别型变量，存在 object/string 列时必须跳过该方案
    if 0 < ks < 0.2 and not _has_categorical_features(state):
        plan.append((
            "woebin",
            {"binning_method": "chimerge"},
            f"KS={ks:.3f} < 0.2 区分度严重不足，换卡方分箱(chimerge)提升单调性",
        ))
    elif 0 < ks < 0.2:
        plan.append((
            "woebin",
            {"max_bins": 10, "min_bin_size": 0.05},
            f"KS={ks:.3f} < 0.2 区分度严重不足；数据含类别型变量不能换 chimerge，改为加细箱数",
        ))

    # ---- R4: KS 略低但 AUC 达标 → 合并小箱 ----
    if ks < min_ks and auc >= min_auc:
        plan.append((
            "woebin",
            {"min_bin_size": 0.08},
            f"KS={ks:.3f} < min_ks={min_ks} 但 AUC={auc:.3f} 达标，合并小箱(min_bin_size=0.08)提升稳定性",
        ))

    # ---- R5: AUC 不足 → 开 LGBM 对照 ----
    if auc < min_auc and not bool(cfg.get("use_lgbm", False)):
        plan.append((
            "model_train",
            {"use_lgbm": True},
            f"AUC={auc:.3f} < min_auc={min_auc}，训练 LGBM 对照模型验证是否为线性模型表达能力不足",
        ))

    # ---- R6: CV fold 单类别（不可自愈，直接记录并转人工）----
    if _has_single_class_fold(guard_warnings, state):
        plan.append((
            "__non_actionable__",
            {},
            "存在单类别 CV fold（某年正负样本不全），属数据问题，自愈无意义，转人工处理",
        ))

    return plan


def _to_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def _csi_top_variables(state: AgentState) -> tuple[str, float, float, float] | None:
    """调 diagnose_psi_sources 拿到 (top_var, top_csi, second_csi, median_csi)；失败返回 None。"""
    bins = state.get("bins")
    train_df = state.get("train_df")
    test_df = state.get("test_df")
    target = state.get("target")
    if bins is None or train_df is None or test_df is None:
        return None
    try:
        csi = diagnose_psi_sources(bins, train_df, test_df, target=target, top_n=5)
    except Exception as e:  # noqa: BLE001 - 诊断失败不影响主流程
        logger.warning("CSI 定位失败: %s", type(e).__name__)
        return None
    if csi is None or len(csi) == 0:
        return None
    vals = csi["csi"].astype(float).tolist()
    top_var = str(csi["variable"].iloc[0])
    top_csi = float(vals[0])
    second_csi = float(vals[1]) if len(vals) > 1 else 0.0
    median_csi = float(np.median(vals)) if vals else 0.0
    return top_var, top_csi, second_csi, median_csi


_CV_SINGLE_CLASS_PAT = (
    "single class", "单类别", "one class", "This solver needs samples of at least 2 classes",
)


def _has_categorical_features(state: AgentState, sample: int = 200) -> bool:
    """训练特征里是否含非数值列（object / string / category）。

    用途：chimerge 分箱不支持类别型变量，在全类别变量数据上会导致
    woebin 全失败 → woebin_apply 崩 → 后续 5 个节点连环报错。
    """
    df = state.get("train_df")
    target = state.get("target")
    if df is None:
        return True  # 未知情况按最保守处理
    for c in df.columns:
        if c == target or c in ID_LIKE_COLS:
            continue
        kind = getattr(df[c].dtype, "kind", "")
        if kind in {"O", "U", "b"} or str(df[c].dtype) in {"category", "string", "str"}:
            return True
    return False


def _has_single_class_fold(guard_warnings: list[str], state: AgentState) -> bool:
    haystack = " ".join(guard_warnings) + " " + " ".join((state.get("warnings") or []))
    low = haystack.lower()
    return any(p.lower() in low for p in _CV_SINGLE_CLASS_PAT)


def _plan_signature(goto: str, patch: dict) -> str:
    """方案指纹，用于「已试过就不再重试」，避免死循环重复同一动作。"""
    items = sorted((k, str(v)) for k, v in patch.items())
    return f"{goto}|{items}"


def _rule_based_diagnosis(state: AgentState, cfg: dict[str, Any]) -> dict[str, Any]:
    """确定性规则诊断（LLM 不可用时的兜底，也是可审核的基准策略）。

    返回 schema 同 LLM：{"action","goto","param_patch","reason","source"}
    """
    plan = _rule_action_plan(state, cfg)
    tried = {
        _plan_signature(h.get("goto", ""), h.get("param_patch", {}) or {})
        for h in (state.get("retry_history") or [])
    }

    for goto, patch, reason in plan:
        if goto == "__non_actionable__":
            return {
                "action": "escalate",
                "goto": "hitl_review",
                "param_patch": {},
                "reason": reason,
                "source": "rule",
                "non_actionable": True,
            }
        if _plan_signature(goto, patch) in tried:
            continue
        return {
            "action": "retune",
            "goto": goto,
            "param_patch": dict(patch),
            "reason": reason,
            "source": "rule",
        }

    return {
        "action": "escalate",
        "goto": "hitl_review",
        "param_patch": {},
        "reason": "规则库内已无可行的自动处置方案（或均已尝试过），转人工复核",
        "source": "rule",
    }


def _build_diagnose_prompt(
    state: AgentState,
    cfg: dict[str, Any],
    max_retry: int,
    retry_count: int,
) -> str:
    metrics = state.get("metrics") or {}
    gr = state.get("guardrail")
    gr_metrics = getattr(gr, "metrics", None) or {}
    allowed_goto = list(cfg.get("allowed_goto") or DEFAULT_ALLOWED_GOTO)
    allowed_params = list(cfg.get("allowed_params") or DEFAULT_ALLOWED_PARAMS)
    bounds = cfg.get("param_bounds") or DEFAULT_PARAM_BOUNDS

    lines = [
        "你是信用风险评分卡建模 Agent 的决策模块。模型未通过上线护栏，请给出唯一的一条处置决策。",
        "",
        "【当前指标】",
        f"  ks={metrics.get('ks')}, auc={metrics.get('auc')}, gini={metrics.get('gini')}, psi={gr_metrics.get('psi')}",
        f"【护栏阈值】min_ks={cfg.get('min_ks', 0.3)}, min_auc={cfg.get('min_auc', 0.7)}, max_psi={cfg.get('max_psi', 0.1)}",
        f"【护栏警告】{(getattr(gr, 'warnings', None) or [])}",
        f"【已重试】{retry_count}/{max_retry} 次",
        f"【历史已试方案】{[{'goto': h.get('goto'), 'patch': h.get('param_patch')} for h in (state.get('retry_history') or [])]}",
        "",
        "【可选回跳节点 goto】只允许从下面选一个，不得自创：",
        f"  {allowed_goto}",
        "  - preprocess    : 重做缺失值/异常值处理",
        "  - var_filter    : 重新筛选变量（如剔除漂移变量）",
        "  - woebin        : 重新分箱（箱数/箱占比/分箱算法）",
        "  - model_train   : 重新训练（如开启 LGBM 对照）",
        "  - build_scorecard: 重算评分卡刻度",
        "",
        "【可修改参数 param_patch】只允许下面这些键：",
        f"  {allowed_params}",
        f"【参数边界】{ {k: list(v) for k, v in bounds.items()} }",
        f"  binning_method 取值: {list(cfg.get('binning_method_choices') or DEFAULT_BINNING_CHOICES)}",
        f"  missing_strategy 取值: {list(cfg.get('missing_strategy_choices') or DEFAULT_MISSING_CHOICES)}",
        "",
        "【输出要求】严格输出一个 JSON 对象，不要 markdown 代码块，不要解释文字：",
        '  {"action": "retune|escalate|abort", "goto": "var_filter", "param_patch": {"max_bins": 6}, "reason": "中文一句话说明判断依据"}',
        "  - retune  : 有可行联动参数，重跑一次（必须给出 goto 与 param_patch）",
        "  - escalate: 转人工复核（无可行自动方案，或已达重试上限）",
        "  - abort   : 判定本数据无法建模，终止",
        "",
        "你现在输出 JSON：",
    ]
    return "\n".join(lines)


def _validate_diagnosis(
    raw: Any,
    cfg: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """把 LLM 原始输出净化成可信 diagnosis。

    规则：
      - 非 dict / action 非法          → 整条丢弃，降级 escalate
      - goto 不在 allowed_goto         → retune 降级为 escalate（不允许跳到未知节点）
      - param_patch key 不在白名单     → 丢弃该键并记 warning
      - value 越界                     → 夹到 param_bounds 边界并记 warning
      - binning_method/missing_strategy 不在 choices → 丢弃该键并记 warning
    """
    cfg = cfg or {}
    allowed_goto = set(cfg.get("allowed_goto") or DEFAULT_ALLOWED_GOTO)
    allowed_params = set(cfg.get("allowed_params") or DEFAULT_ALLOWED_PARAMS)
    bounds = cfg.get("param_bounds") or DEFAULT_PARAM_BOUNDS
    bin_choices = set(cfg.get("binning_method_choices") or DEFAULT_BINNING_CHOICES)
    miss_choices = set(cfg.get("missing_strategy_choices") or DEFAULT_MISSING_CHOICES)

    warns: list[str] = []

    if not isinstance(raw, dict):
        return {
            "action": "escalate", "goto": "hitl_review", "param_patch": {},
            "reason": f"诊断输出非法（{type(raw).__name__}），已转人工", "source": "llm(invalid)",
        }, [f"[诊断] 输出非 dict: {type(raw).__name__}"]

    action = str(raw.get("action", "")).strip().lower()
    if action not in {"retune", "escalate", "abort"}:
        warns.append(f"[诊断] 非法 action={raw.get('action')!r}，降级为 escalate")
        action = "escalate"

    reason = str(raw.get("reason", "") or "").strip() or "未提供理由"
    goto = raw.get("goto")
    patch_raw = raw.get("param_patch") or {}
    if not isinstance(patch_raw, dict):
        warns.append(f"[诊断] param_patch 非 dict: {type(patch_raw).__name__}，已忽略")
        patch_raw = {}

    # ---- 净化 param_patch ----
    clean_patch: dict[str, Any] = {}
    for k, v in patch_raw.items():
        if k not in allowed_params:
            warns.append(f"[诊断][丢弃] 参数 {k} 不在白名单")
            continue
        if k in DEFAULT_PARAM_BOUNDS and k in bounds:
            lo, hi = float(bounds[k][0]), float(bounds[k][1])
            try:
                num = int(v) if k in INT_PARAMS else float(v)
            except (TypeError, ValueError):
                warns.append(f"[诊断][丢弃] {k}: 值 {v!r} 无法转数值")
                continue
            if num < lo or num > hi:
                clamped = int(max(lo, min(hi, num))) if k in INT_PARAMS else float(max(lo, min(hi, num)))
                warns.append(f"[诊断][夹取] {k}: {num} 超出 [{lo}, {hi}] → {clamped}")
                num = clamped
            clean_patch[k] = num
            continue
        if k == "binning_method":
            if str(v) in bin_choices:
                clean_patch[k] = str(v)
            else:
                warns.append(f"[诊断][丢弃] binning_method={v!r} 不在 {sorted(bin_choices)}")
            continue
        if k == "missing_strategy":
            if str(v) in miss_choices:
                clean_patch[k] = str(v)
            else:
                warns.append(f"[诊断][丢弃] missing_strategy={v!r} 不在 {sorted(miss_choices)}")
            continue
        if k in LIST_PARAMS:
            if isinstance(v, list):
                clean_patch[k] = v
            elif isinstance(v, str):
                clean_patch[k] = [x.strip() for x in v.split(",") if x.strip()]
            else:
                warns.append(f"[诊断][丢弃] {k}: 需要列表，得到 {type(v).__name__}")
            continue
        if k in BOOL_PARAMS:
            if isinstance(v, bool):
                clean_patch[k] = v
            elif isinstance(v, str) and v.strip().lower() in {"true", "1", "yes"}:
                clean_patch[k] = True
            elif isinstance(v, str) and v.strip().lower() in {"false", "0", "no"}:
                clean_patch[k] = False
            else:
                warns.append(f"[诊断][丢弃] {k}: 无法识别布尔值 {v!r}")
            continue
        # outlier_method 等无 bounds 的字符串键：原样接受
        clean_patch[k] = v

    # ---- 处置 action / goto ----
    if action == "retune":
        if goto not in allowed_goto:
            warns.append(f"[诊断] goto={goto!r} 不在允许列表，降级为 escalate")
            action, goto = "escalate", "hitl_review"
        elif not clean_patch:
            warns.append("[诊断] retune 但无有效 param_patch，降级为 escalate")
            action, goto = "escalate", "hitl_review"
    else:
        goto = "hitl_review"
    # escalte/abort/降级：一律清空 patch，避免残留「看似要改但没执行」的参数
    if action != "retune":
        clean_patch = {}

    return {
        "action": action,
        "goto": goto,
        "param_patch": clean_patch,
        "reason": reason,
        "source": "llm",
    }, warns


def _merge_config(cfg: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """把 param_patch 合并进 config_overrides（exclude_vars 走集合并集，避免覆盖）。"""
    merged = dict(cfg)
    for k, v in patch.items():
        if k in LIST_PARAMS:
            old = list(merged.get(k) or [])
            for item in v:
                if item not in old:
                    old.append(item)
            merged[k] = old
        else:
            merged[k] = v
    return merged


def diagnose_node(state: AgentState) -> Command:
    """【V2 模块 1】诊断-重规划：读护栏结果 + 指标 + CSI → 决策回跳或转人工。

    不加 @safe（@safe 会把 Command 包成 dict，导致回跳失效），
    因此内部自建 try/except，任何异常都必须能退到 rule-based 兜底。

    Returns:
        Command(goto=<节点名>, update={diagnosis, retry_count, retry_history,
                                      config_overrides, warnings, step_history})
    """
    from langgraph.types import Command as LangGraphCommand  # noqa: F401

    cfg = dict(state.get("config_overrides") or {})
    max_retry = int(cfg.get("max_retry", 3))
    retry_count = int(state.get("retry_count") or 0)
    allow_llm = bool(cfg.get("allow_llm_diagnosis", True))

    # ---- 停止条件 1：已达最大重试次数 ----
    if retry_count >= max_retry:
        diag = {
            "action": "escalate", "goto": "hitl_review", "param_patch": {},
            "reason": f"已自动重规划 {retry_count} 次达到上限 {max_retry}，转人工复核",
            "source": "guard",
        }
        return _make_diagnose_command(state, diag, [], cfg)

    # ---- 1) LLM 决策（可失败）----
    diag: dict[str, Any] | None = None
    warns: list[str] = []
    if allow_llm:
        try:
            raw = _llm_json(
                _build_diagnose_prompt(state, cfg, max_retry, retry_count),
                temperature=0,
            )
            if raw is not None:
                diag, warns = _validate_diagnosis(raw, cfg)
            else:
                warns.append("[诊断] LLM 不可用或返回非法 JSON，已回退规则诊断")
        except Exception as e:  # noqa: BLE001 - 双保险：_llm_json 已吞异常，这里防 _validate 出错
            logger.warning("LLM 诊断失败: %s", type(e).__name__)
            warns.append(f"[诊断] LLM 诊断异常({type(e).__name__})，已回退规则诊断")
            diag = None
    else:
        warns.append("[诊断] allow_llm_diagnosis=False，走确定性规则诊断")

    # ---- 2) 规则兜底 ----
    if diag is None or diag.get("action") not in {"retune", "escalate", "abort"}:
        diag = _rule_based_diagnosis(state, cfg)

    # ---- 3) 防原地打转：LLM 给出的方案若已在历史里 → 退回规则阶梯挑新方案 ----
    if diag.get("action") == "retune":
        sig = _plan_signature(diag["goto"], diag.get("param_patch") or {})
        tried = {
            _plan_signature(h.get("goto", ""), h.get("param_patch") or {})
            for h in (state.get("retry_history") or [])
        }
        if sig in tried:
            warns = list(warns) + [
                f"[诊断] LLM 重复已试过的方案 {sig}，改由规则阶梯挑选新方案"
            ]
            diag = _rule_based_diagnosis(state, cfg)

    return _make_diagnose_command(state, diag, warns, cfg)


def _make_diagnose_command(
    state: AgentState,
    diag: dict[str, Any],
    warns: list[str],
    cfg: dict[str, Any],
) -> Command:
    """统一构造 diagnose 的 Command 返回（含计数、历史、config 合并）。"""
    action = diag.get("action", "escalate")
    retune = action == "retune"

    new_cfg = cfg
    if retune:
        new_cfg = _merge_config(cfg, dict(diag.get("param_patch") or {}))

    retry_count = int(state.get("retry_count") or 0)
    next_count = retry_count + 1 if retune else retry_count

    record = {
        "round": next_count,
        "action": action,
        "goto": diag.get("goto"),
        "param_patch": dict(diag.get("param_patch") or {}),
        "reason": diag.get("reason", ""),
        "source": diag.get("source", "rule"),
        "ks": state.get("metrics", {}).get("ks"),
        "auc": state.get("metrics", {}).get("auc"),
        "psi": (getattr(state.get("guardrail"), "metrics", None) or {}).get("psi"),
    }

    target = diag.get("goto") if retune else "hitl_review"
    return Command(
        goto=target,
        update={
            "diagnosis": diag,
            "retry_count": next_count,
            "retry_history": [record],
            "config_overrides": new_cfg,
            "warnings": warns,
            "step_history": [
                f"[DIAG#{next_count}] {action} → {target}"
                + (f" patch={diag.get('param_patch')}" if retune else "")
            ],
        },
    )


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
# 【已迁移】路由函数统一在 agent/routing.py（避免实现与图装配两处漂移）
# ============================================================================
from agent.routing import route_after_guardrail, route_after_hitl  # noqa: E402,F401