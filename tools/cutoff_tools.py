"""cut-off 择优与分数分档工具集（模块 5）。

约定（与评分卡一致）：
    **分数越高 = 违约风险越低**。因此 cut-off t 的语义是「**拒** score < t，通过 score ≥ t」。

包含：
    - optimize_cutoff: 网格搜索最优 cut-off（最大化 KS / 命中目标通过率 / 命中目标坏率）
    - grade_scores: 按分位或固定分界给分数分档（A/B/C/D ...）

全部为确定性计算，不依赖 LLM。
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 网格默认取到 1%~99% 分位，避免被极端单点分拉着走
_GRID_Q = np.linspace(0.01, 0.99, 99)


def _as_arrays(y_true, score) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(pd.Series(y_true).astype(float).to_numpy(), dtype=float)
    s = np.asarray(pd.Series(score).astype(float).to_numpy(), dtype=float)
    mask = ~(np.isnan(y) | np.isnan(s))
    return y[mask], s[mask]


def _empty_result(reason: str) -> dict[str, Any]:
    table = pd.DataFrame(
        columns=[
            "cutoff", "n_approved", "approve_rate", "bad_rate_after",
            "capture_rate", "fpr", "ks", "lift",
        ]
    )
    return {
        "method": None,
        "cutoff": None,
        "approve_rate": None,
        "bad_rate_after": None,
        "ks_at_cutoff": None,
        "capture_rate": None,
        "lift": None,
        "table": table,
        "warnings": [reason],
    }


def optimize_cutoff(
    y_true,
    score,
    method: Literal["ks", "approve_rate", "bad_rate"] = "ks",
    target: float | None = None,
    n_grid: int | None = None,
) -> dict[str, Any]:
    """在分数区间上网格搜索最优 cut-off。

    语义：拒绝 `score < cutoff`，通过 `score >= cutoff`（分数越高风险越低）。

    Args:
        y_true: 真实标签，1 = 违约/坏客户
        score: 模型分数（评分卡总分或违约概率的反向刻度，越高越安全）
        method:
            - "ks"           : 最大化 KS = max_t (捕获率 − 误伤率)
            - "approve_rate" : 命中目标通过率 target（如 0.7 表示通过 70%）
            - "bad_rate"     : 命中目标通过后坏率 target（如 0.02）
        target: method 为后两者时必填
        n_grid: 网格点数；None 时用 1%~99% 分位取 99 点

    Returns:
        dict:
            cutoff          : 选定的分数临界值
            approve_rate    : 通过率
            bad_rate_after  : 通过人群中的实际坏率
            ks_at_cutoff    : 该切点下的 KS
            capture_rate    : 违约捕获率（被拒样本覆盖了多少真实坏客户）
            lift            : 提升度 = 整体坏率 / 通过后坏率（>1 表示筛选有效）
            table           : 全网格明细 DataFrame
            warnings        : 边界/降级提示
    """
    if method in {"approve_rate", "bad_rate"} and target is None:
        raise ValueError(f"method={method} 必须提供 target")

    y, s = _as_arrays(y_true, score)
    if len(y) == 0:
        return _empty_result("样本为空，无法择优 cut-off")
    n_bad = float(y.sum())
    n_good = float((y == 0).sum())
    base_rate = float(y.mean())

    if n_bad == 0 or n_good == 0:
        return _empty_result("标签只有单一类别，无法计算 KS/捕获率")

    # ---- 构造候选切点 ----
    unique_vals = np.unique(s)
    if n_grid is not None:
        qs = np.linspace(0.0, 1.0, int(n_grid) + 2)[1:-1]
        cand = np.quantile(s, qs)
    else:
        cand = np.quantile(s, _GRID_Q)
    cand = np.unique(np.concatenate([cand, [unique_vals.min() - 1e-9, unique_vals.max() + 1e-9]]))
    # 限制在合理范围内：全部通过（cutoff = min-ε）与全部拒绝（cutoff = max+ε）保留，
    # 但择优时不选这两个退化端点。
    interior_mask = (cand > unique_vals.min()) & (cand < unique_vals.max())
    pick_pool = cand[interior_mask] if interior_mask.any() else cand

    rows: list[dict[str, float]] = []
    for t in pick_pool:
        reject = s < t
        approve = ~reject
        n_app = int(approve.sum())
        if n_app == 0:
            continue
        bad_app = float(y[approve].sum())
        tpr = float((y[reject] == 1).sum() / n_bad)      # 违约捕获率
        fpr = float((y[reject] == 0).sum() / n_good)     # 误伤率（好客户被拒）
        ks = tpr - fpr
        rows.append({
            "cutoff": float(t),
            "n_approved": n_app,
            "approve_rate": float(n_app / len(y)),
            "bad_rate_after": float(bad_app / n_app) if n_app else 0.0,
            "capture_rate": float(tpr),
            "fpr": float(fpr),
            "ks": float(ks),
            "lift": float(base_rate / (bad_app / n_app)) if bad_app > 0 else float("inf"),
        })

    if not rows:
        return _empty_result("网格候选点全部退化（可能分数取值过少）")

    table = pd.DataFrame(rows)
    warnings: list[str] = []

    # ---- 按 method 择优 ----
    if method == "ks":
        best_idx = int(table["ks"].idxmax())
    elif method == "approve_rate":
        best_idx = int((table["approve_rate"] - float(target)).abs().idxmin())
        warnings.append(
            f"目标通过率 {float(target):.2%}，实际最接近 {table.loc[best_idx, 'approve_rate']:.2%}"
        )
    else:  # bad_rate
        best_idx = int((table["bad_rate_after"] - float(target)).abs().idxmin())
        warnings.append(
            f"目标通过后坏率 {float(target):.2%}，实际最接近 {table.loc[best_idx, 'bad_rate_after']:.2%}"
        )

    best = table.loc[best_idx]
    result = {
        "method": method,
        "cutoff": float(best["cutoff"]),
        "approve_rate": float(best["approve_rate"]),
        "bad_rate_after": float(best["bad_rate_after"]),
        "ks_at_cutoff": float(best["ks"]),
        "capture_rate": float(best["capture_rate"]),
        "lift": float(best["lift"]),
        "table": table.reset_index(drop=True),
        "warnings": warnings,
        "base_bad_rate": base_rate,
    }
    logger.info(
        "cut-off 择优: method=%s cutoff=%.2f 通过率=%.2f%% 通过后坏率=%.2f%% 捕获率=%.2f%% lift=%.2f",
        method, result["cutoff"], result["approve_rate"] * 100,
        result["bad_rate_after"] * 100, result["capture_rate"] * 100, result["lift"],
    )
    return result


def grade_scores(
    score,
    cutoffs: list[float] | None = None,
    labels: tuple[str, ...] = ("A", "B", "C", "D"),
) -> pd.DataFrame:
    """给分数分档（默认按分位切成 A/B/C/D，A 最好）。

    Args:
        score: 分数序列
        cutoffs: 自定义分界点（k-1 个），按分位点大小升序或降序均可；
                 None 时按 `labels` 数量等频分位自动生成
        labels: 档位标签，**从最好到最差**排列

    Returns:
        DataFrame，两列：`score` / `grade`
    """
    s = pd.Series(score).astype(float).reset_index(drop=True)
    labels = list(labels)
    k = len(labels)
    if k < 2:
        raise ValueError("labels 至少需要 2 个档位")

    if cutoffs is None:
        clean = s.dropna()
        if len(clean) == 0:
            cutoffs = []
        else:
            qs = np.linspace(0, 1, k + 1)[1:-1]
            cutoffs = [float(v) for v in np.quantile(clean, qs)]
    else:
        cutoffs = [float(c) for c in cutoffs]

    # 统一升序，并补齐/截断到 k-1 个
    cutoffs = sorted(set(cutoffs))
    if len(cutoffs) > k - 1:
        cutoffs = cutoffs[: k - 1]
    while len(cutoffs) < k - 1:
        cutoffs = list(cutoffs)
        cutoffs.append(cutoffs[-1] + 1.0 if cutoffs else 0.0)

    grade = pd.Series(labels[-1], index=s.index, dtype=object)
    assigned = pd.Series(False, index=s.index)
    # 从最高分界往下派发好档位
    for i in range(len(cutoffs)):
        bound = cutoffs[len(cutoffs) - 1 - i]
        mask = (s >= bound) & (~assigned) & s.notna()
        grade[mask] = labels[i]
        assigned = assigned | mask

    return pd.DataFrame({"score": s, "grade": grade})
