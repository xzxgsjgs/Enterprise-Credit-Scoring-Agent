"""模型验证与合规护栏工具集。

包含:
    - compute_psi: 群体稳定性指标 (Population Stability Index)
    - compute_csi: 特征稳定性指标 (Characteristic Stability Index)
    - guardrail_check: 统一 KS/AUC/PSI 校验
    - check_data_quality: 【V2】数据质量前置守门（确定性，无 LLM）
    - diagnose_psi_sources: 【V2】逐变量 CSI，定位分布漂移来源
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_psi(
    expected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
    bins: int = 10,
    method: str = "quantile",
) -> float:
    """计算 PSI。

    Args:
        expected: 训练/基准集分布 (e.g. model.score)
        actual: 验证/测试集分布
        bins: 分箱数
        method: 'quantile' 沿用 expected 分箱切分, 'equal' 等距

    Returns:
        PSI 浮点值
    """
    expected_arr = np.asarray(expected)
    actual_arr = np.asarray(actual)

    if method == "quantile":
        cut_points = np.quantile(expected_arr, np.linspace(0, 1, bins + 1))
    else:
        cut_points = np.linspace(expected_arr.min(), expected_arr.max(), bins + 1)
    cut_points = np.unique(cut_points)

    # 使用 pd.Series 包装以确保 value_counts 可用
    exp_series = pd.Series(expected_arr)
    act_series = pd.Series(actual_arr)
    expected_bins = pd.cut(exp_series, cut_points, include_lowest=True)
    actual_bins = pd.cut(act_series, cut_points, include_lowest=True)

    e_dist = expected_bins.value_counts(normalize=True).sort_index()
    a_dist = actual_bins.value_counts(normalize=True).sort_index()

    df = pd.DataFrame({"e": e_dist, "a": a_dist}).fillna(0)
    df["e"] = df["e"].replace(0, 0.0001)
    df["a"] = df["a"].replace(0, 0.0001)
    psi = float(((df["a"] - df["e"]) * np.log(df["a"] / df["e"])).sum())
    logger.info("PSI = %.4f (bins=%d, method=%s)", psi, bins, method)
    return psi


def compute_csi(
    expected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
    bins: int = 10,
) -> float:
    """CSI = 单变量的 PSI，特征级别稳定性。"""
    return compute_psi(expected, actual, bins=bins)


@dataclass
class GuardrailResult:
    """护栏校验结果结构化输出，可被 LangGraph 节点路由消费。"""
    passed: bool
    metrics: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    requires_human_review: bool = False
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "metrics": self.metrics,
            "warnings": self.warnings,
            "requires_human_review": self.requires_human_review,
            "rationale": self.rationale,
        }


def guardrail_check(
    metrics: dict[str, float],
    reference_dist: np.ndarray | pd.Series | None = None,
    current_dist: np.ndarray | pd.Series | None = None,
    min_ks: float = 0.3,
    min_auc: float = 0.7,
    max_psi: float = 0.1,
    require_human_review: bool = True,
) -> GuardrailResult:
    """综合校验 KS、AUC、PSI。

    当任一项不达标:
        - 严重违规(KS<0.2 或 PSI>0.25): 拒绝通过
        - 一般违规(KS<0.3 或 PSI>0.1): 警告 + 触发 HITL

    Args:
        metrics: {'ks', 'auc', 'gini', ...}
        reference_dist: 训练期分数/特征分布
        current_dist: 当前期分数/特征分布
        min_ks / min_auc / max_psi: 阈值
        require_human_review: 是否触发人工复核

    Returns:
        GuardrailResult 实例
    """
    warnings: list[str] = []
    requires_human = False
    passed = True

    ks = metrics.get("ks", 1.0)
    auc = metrics.get("auc", 1.0)

    if ks < 0.2:
        passed = False
        warnings.append(f"[CRITICAL] KS={ks:.3f} < 0.2, 模型区分度严重不足")
    elif ks < min_ks:
        requires_human = True
        warnings.append(f"[WARN] KS={ks:.3f} < {min_ks}, 建议复核")

    if auc < min_auc:
        requires_human = True
        warnings.append(f"[WARN] AUC={auc:.3f} < {min_auc}, 模型辨别力较弱")

    psi_val: float | None = None
    if reference_dist is not None and current_dist is not None:
        psi_val = compute_psi(reference_dist, current_dist)
        metrics = dict(metrics)
        metrics["psi"] = psi_val
        if psi_val > 0.25:
            passed = False
            warnings.append(f"[CRITICAL] PSI={psi_val:.3f} > 0.25, 分布严重漂移")
        elif psi_val > max_psi:
            requires_human = True
            warnings.append(f"[WARN] PSI={psi_val:.3f} > {max_psi}, 分布漂移触发复核")

    if requires_human and not require_human_review:
        requires_human = False

    rationale = "模型校验通过" if passed and not requires_human else (
        "模型存在风险，建议人工复核" if passed else "模型未通过护栏，禁止上线"
    )

    result = GuardrailResult(
        passed=passed,
        metrics=metrics,
        warnings=warnings,
        requires_human_review=requires_human,
        rationale=rationale,
    )
    logger.info("护栏检查: %s | 警告=%d | 人审=%s", result.rationale, len(warnings), requires_human)
    return result


# ============================================================================
# 【V2】数据质量前置守门（模块 6）
# ============================================================================
def check_data_quality(
    df: pd.DataFrame,
    target: str,
    min_rows: int = 1000,
    min_default_rate: float = 0.005,
    max_default_rate: float = 0.5,
    min_years: int = 3,
    max_missing_ratio: float = 0.6,
    max_identical_ratio: float = 0.98,
    year_col: str = "year",
    time_split: bool = False,
    exclude_cols: list[str] | None = None,
) -> dict[str, Any]:
    """训练前的数据质量硬校验，全部为确定性规则（不含任何 LLM 判断）。

    校验项（任一 blocks 非空即 passed=False）：
      1. 行数 < min_rows                          → block
      2. 目标列缺失                                → block
      3. 违约率 < min_default_rate 或 > max_default_rate → block
      4. time_split=True 时年份数 < min_years      → block
      5. 单列缺失率 > max_missing_ratio            → warning（建议剔除）
      6. 单列单一值占比 > max_identical_ratio      → warning（建议剔除）
      7. 违约样本数 < 10                           → block（逻辑回归无法学）

    Args:
        df: 待校验数据
        target: 目标列名
        min_rows: 最小行数
        min_default_rate / max_default_rate: 违约率合法区间
        min_years: time_split 模式下最少年份数
        max_missing_ratio: 单列缺失率上限
        max_identical_ratio: 单列单一值占比上限
        year_col: 年份列名
        time_split: 是否启用时序切分
        exclude_cols: 不参与质量校验的列（如 Symbol / ShortName 等 ID 列）

    Returns:
        dict: {"passed": bool, "blocks": list[str], "warnings": list[str],
               "stats": dict, "suggest_drop": list[str]}
        - suggest_drop: 建议剔除的列名（缺失率/单一值超限）
    """
    blocks: list[str] = []
    warns: list[str] = []
    suggest_drop: list[str] = []

    n_rows = int(len(df))
    stats: dict[str, Any] = {"n_rows": n_rows, "n_cols": int(df.shape[1])}

    # ---- 1. 行数 ----
    if n_rows < min_rows:
        blocks.append(f"样本量不足: {n_rows} 行 < 要求 {min_rows} 行")

    # ---- 2. 目标列 ----
    if target not in df.columns:
        blocks.append(f"目标列缺失: '{target}' 不在数据列中")
        return {
            "passed": False,
            "blocks": blocks,
            "warnings": warns,
            "stats": stats,
            "suggest_drop": suggest_drop,
        }

    # ---- 3. 违约率与违约样本数 ----
    y = df[target]
    # 兼容 'good'/'bad' 字符串标签
    if y.dtype == object or str(y.dtype) == "str":
        y_num = (y.astype(str).str.lower() == "bad").astype(int)
    else:
        y_num = pd.to_numeric(y, errors="coerce")

    n_pos = int(pd.Series(y_num).fillna(0).sum())
    default_rate = float(n_pos / n_rows) if n_rows else 0.0
    stats["n_default"] = n_pos
    stats["default_rate"] = default_rate

    if default_rate < min_default_rate:
        blocks.append(
            f"违约率过低: {default_rate:.4%} ({n_pos}/{n_rows}) < 下限 {min_default_rate:.4%}，正样本不足无法建模"
        )
    elif default_rate > max_default_rate:
        blocks.append(
            f"违约率过高: {default_rate:.4%} ({n_pos}/{n_rows}) > 上限 {max_default_rate:.4%}，标签定义可能有问题"
        )
    if 0 < n_pos < 10:
        blocks.append(f"违约样本过少: 仅 {n_pos} 个，逻辑回归无法收敛（建议 ≥ 10）")

    # ---- 4. 年份覆盖（时序模式）----
    if time_split:
        if year_col not in df.columns:
            blocks.append(f"时序模式需要年份列 '{year_col}'，但数据中不存在")
        else:
            n_years = int(pd.Series(df[year_col]).dropna().nunique())
            stats["n_years"] = n_years
            if n_years < min_years:
                blocks.append(
                    f"年份覆盖不足: {n_years} 年 < 要求 {min_years} 年（Expanding Window CV 需要更多年份）"
                )

    # ---- 5/6. 逐列质量（跳过 ID 列与目标列）----
    skip = set(exclude_cols or []) | {target}
    for c in df.columns:
        if c in skip:
            continue
        col = df[c]
        # 缺失率
        miss_ratio = float(col.isna().mean()) if n_rows else 0.0
        if miss_ratio > max_missing_ratio:
            warns.append(f"列 '{c}' 缺失率 {miss_ratio:.2%} > {max_missing_ratio:.0%}，建议剔除")
            suggest_drop.append(c)
        # 单一值占比
        if n_rows and col.notna().any():
            top_share = float(col.value_counts(normalize=True, dropna=True).iloc[0])
            if top_share > max_identical_ratio:
                warns.append(
                    f"列 '{c}' 单一取值占比 {top_share:.2%} > {max_identical_ratio:.0%}，几乎无区分度，建议剔除"
                )
                if c not in suggest_drop:
                    suggest_drop.append(c)

    passed = len(blocks) == 0
    logger.info(
        "数据质量守门: passed=%s | blocks=%d | warnings=%d | rows=%d | default_rate=%.4f",
        passed, len(blocks), len(warns), n_rows, default_rate,
    )
    return {
        "passed": passed,
        "blocks": blocks,
        "warnings": warns,
        "stats": stats,
        "suggest_drop": suggest_drop,
    }


# ============================================================================
# 【V2】PSI 漂移来源定位（模块 2）
# ============================================================================
def diagnose_psi_sources(
    bins: dict[str, pd.DataFrame],
    expected_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    target: str,
    top_n: int = 5,
) -> pd.DataFrame:
    """逐变量计算 CSI（特征级 PSI），定位分布漂移来源。

    对变量 j：按 bins[j] 的箱边界分别统计 expected / actual 的箱占比，
        CSI_j = Σ_i (a_i − e_i) · ln(a_i / e_i)
    空箱用 1e-4 平滑，避免除零与 ln(0)。

    Args:
        bins: woebin() 返回的分箱字典 {var: DataFrame[variable, bin, ...]}
        expected_df: 训练集原始值（未 WOE）
        actual_df: 测试集原始值（未 WOE）
        target: 目标列名（不参与计算）
        top_n: 返回 CSI 最高的前 N 个变量

    Returns:
        DataFrame 列: variable / csi / top_moved_bin / expected_pct / actual_pct
        按 csi 降序；无结果时返回空 DataFrame（列名一致）
    """
    empty = pd.DataFrame(
        columns=["variable", "csi", "top_moved_bin", "expected_pct", "actual_pct"]
    )
    if not bins:
        return empty

    rows: list[dict[str, Any]] = []
    for var, bin_df in bins.items():
        if var == target or var not in expected_df.columns or var not in actual_df.columns:
            continue
        # 取出该变量的箱边界字符串，如 '[-inf,0.5)' / '[0.5,1.0)'
        if "bin" not in bin_df.columns:
            continue
        edges = bin_df["bin"].astype(str).tolist()
        if not edges:
            continue

        exp_col = expected_df[var]
        act_col = actual_df[var]

        # 数值变量：按 bin 边界切分；类别变量：直接按取值占比
        try:
            exp_numeric = pd.api.types.is_numeric_dtype(exp_col)
            act_numeric = pd.api.types.is_numeric_dtype(act_col)
            if exp_numeric and act_numeric:
                cuts = _parse_bin_edges(edges)
                if cuts is None:
                    continue
                exp_bins = pd.cut(exp_col, cuts, include_lowest=True)
                act_bins = pd.cut(act_col, cuts, include_lowest=True)
                exp_dist = exp_bins.value_counts(normalize=True).sort_index()
                act_dist = act_bins.value_counts(normalize=True).sort_index()
            else:
                exp_dist = (
                    exp_col.astype(str).value_counts(normalize=True)
                )
                act_dist = (
                    act_col.astype(str).value_counts(normalize=True)
                )
                exp_dist, act_dist = exp_dist.align(act_dist, fill_value=0.0)
        except Exception as e:  # pragma: no cover - 防御性
            logger.warning("变量 %s CSI 计算失败: %s", var, e)
            continue

        combined = pd.DataFrame({"e": exp_dist, "a": act_dist}).fillna(0.0)
        # 平滑：避免 ln(0) 与除零
        combined["e"] = combined["e"].replace(0, 1e-4)
        combined["a"] = combined["a"].replace(0, 1e-4)
        csi = float(((combined["a"] - combined["e"]) * np.log(combined["a"] / combined["e"])).sum())

        # 找贡献最大的箱
        contrib = (combined["a"] - combined["e"]) * np.log(combined["a"] / combined["e"])
        top_idx = contrib.idxmax() if len(contrib) else None

        rows.append({
            "variable": var,
            "csi": csi,
            "top_moved_bin": str(top_idx),
            "expected_pct": float(combined.loc[top_idx, "e"]) if top_idx is not None else 0.0,
            "actual_pct": float(combined.loc[top_idx, "a"]) if top_idx is not None else 0.0,
        })

    if not rows:
        return empty

    out = pd.DataFrame(rows).sort_values("csi", ascending=False).reset_index(drop=True)
    return out.head(top_n)


def _parse_bin_edges(edges: list[str]) -> list[float] | None:
    """把 woebin 的 bin 字符串列表解析为 pd.cut 可用的切点数组。

    例：['[-inf,0.5)', '[0.5,1.0)', '[1.0,inf)']
        → [-inf, 0.5, 1.0, inf]
    """
    pts: list[float] = []
    for e in edges:
        s = e.strip()
        if not (s.startswith("[") or s.startswith("(")):
            return None
        inner = s[1:-1] if s[-1] in "])" else s[1:]
        parts = inner.split(",")
        if len(parts) != 2:
            return None
        try:
            lo = float(parts[0])
            hi = float(parts[1])
        except ValueError:
            return None
        if not pts:
            pts.append(lo)
        pts.append(hi)
    if len(pts) < 2:
        return None
    # 保证单调
    pts = sorted(set(pts))
    return pts