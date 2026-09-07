"""模型验证与合规护栏工具集。

包含:
    - compute_psi: 群体稳定性指标 (Population Stability Index)
    - compute_csi: 特征稳定性指标 (Characteristic Stability Index)
    - guardrail_check: 统一 KS/AUC/PSI 校验
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