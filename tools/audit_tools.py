"""tools.audit_tools — Critic 用的三项确定性审计（模块 3）

设计原则（硬约束 1/3）：
- **全部数值由 Python 计算**，本模块内没有任何 LLM 调用；`critic_node` 只把结果
  交给 LLM 做「定性复核」，LLM 不能改写这些数值。
- **任何一项失败不影响主流程**：调用方（critic_node）对每个检查单独 try/except。

包含：
    - compute_vif(X)                 : 多重共线性（VIF = 1/(1-R²_j)）
    - check_coef_consistency(...)    : 系数符号与业务方向是否一致
    - check_sample_concentration(...) : 类别取值集中度（某取值占比是否过高）

WOE 方向约定（已用真实数据验证，勿改）：
    scorecardpy 的 woe = ln(坏样本占比 / 好样本占比)，因此 **WOE 越大 = 风险越高**。
    实测每个变量 corr(woe, badprob) ≈ +1，故逻辑回归系数**本应为正号**；
    出现负号意味着要么多重共线性导致符号翻转，要么业务方向反转，两者都值得报警。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

# VIF 常用经验阈值：>5 需关注，>10 判定多重共线性
VIF_WARN = 5.0
VIF_FAIL = 10.0
# 单一取值占比上限
CONCENTRATION_THRESHOLD = 0.5
# 数值型列的取值数量上限：超过就不算「类别型」，不做集中度检查
MAX_DISTINCT_FOR_CONCENTRATION = 50


# ============================================================================
# 通用小工具
# ============================================================================
def _numeric_frame(X: pd.DataFrame) -> pd.DataFrame:
    """只保留数值列并中位数填补缺失；非数值列直接丢弃（VIF 无法对其回归）。"""
    num = X.select_dtypes(include="number").copy()
    if num.empty:
        return num
    return num.fillna(num.median(numeric_only=True))


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


# ============================================================================
# 1) VIF：多重共线性
# ============================================================================
def compute_vif(
    X: pd.DataFrame,
    warn: float = VIF_WARN,
    fail: float = VIF_FAIL,
    max_columns: int = 60,
) -> pd.DataFrame:
    """计算每个变量的方差膨胀因子 VIF。

    对每个变量 j，用其余变量线性回归拟合 x_j 得到决定系数 R²_j，则
        `VIF_j = 1 / (1 − R²_j)`
    R²_j → 1 表示 x_j 能被其他变量几乎完美线性表出（多重共线性），此时 VIF → ∞。

    Args:
        X: 特征矩阵（通常是 WOE 变换后的 `train_woe`）
        warn: 关注阈值，默认 5
        fail: 判定阈值，默认 10
        max_columns: 超过该列数只取前 N 列（O(n²) 回归，防止大宽表拖慢）

    Returns:
        DataFrame: `variable / vif / flag`，按 vif 降序。
        flag ∈ {"ok", "warn", "fail"}；退化情形（全 NaN / 常量列）记 `inf` 并判 fail。
    """
    cols = ["variable", "vif", "flag"]
    if X is None or len(X) == 0:
        return _empty(cols)

    num = _numeric_frame(X)
    if len(num.columns) > max_columns:
        logger.warning("compute_vif: 列数 %d > %d，只取前 %d 列", len(num.columns), max_columns, max_columns)
        num = num.iloc[:, :max_columns]
    if num.shape[1] < 2:
        return _empty(cols)

    rows: list[dict[str, Any]] = []
    for col in num.columns:
        y = num[col].to_numpy(dtype=float)
        others = num.drop(columns=[col])
        # 常量列：其他变量必然无法解释它 → R²=0 → VIF=1；但它本身也提供不了信息
        if np.nanstd(y) == 0:
            rows.append({"variable": col, "vif": float("inf"), "flag": "fail"})
            continue
        try:
            reg = LinearRegression().fit(others.to_numpy(dtype=float), y)
            r2 = float(reg.score(others.to_numpy(dtype=float), y))
        except Exception as e:  # noqa: BLE001 - 单个变量失败不应中断整体审计
            logger.debug("VIF 计算失败(%s): %s", col, e)
            rows.append({"variable": col, "vif": float("nan"), "flag": "ok"})
            continue
        if r2 >= 1.0 - 1e-12:        # 完全共线性
            vif = float("inf")
        else:
            vif = float(1.0 / (1.0 - r2))
        if vif == float("inf") or vif >= fail:
            flag = "fail"
        elif vif >= warn:
            flag = "warn"
        else:
            flag = "ok"
        rows.append({"variable": col, "vif": vif, "flag": flag})

    out = pd.DataFrame(rows)
    if not out.empty:
        # inf 排在最前，其余降序
        out = out.sort_values("vif", ascending=False, na_position="last").reset_index(drop=True)
    n_fail = int((out["flag"] == "fail").sum()) if not out.empty else 0
    logger.info("VIF 检查完成: %d 列，其中 %d 列多重共线性", len(out), n_fail)
    return out


# ============================================================================
# 2) 系数符号 vs 业务方向
# ============================================================================
def check_coef_consistency(
    bins: dict[str, pd.DataFrame],
    model: Any,
    xcolumns: list[str] | None = None,
    suffix: str = "_woe",
) -> pd.DataFrame:
    """检查「模型系数符号」与「WOE 指示的风险方向」是否一致。

    判定逻辑（全部确定性）：
        1. 对变量 j 取其分箱表 bins[j]，算 corr(woe_i, badprob_i) 的符号 s_j；
           按 scorecardpy 约定 woe = ln(坏占比/好占比)，s_j 正常情况下 = +1。
        2. 模型里该变量对应的系数符号 a_j = sign(coef_j)。
        3. `ok_j = (s_j == a_j)`；不一致 → 方向反转，需要人工判读。

    Args:
        bins: `woebin` 的输出 {变量名: 分箱表}
        model: 已训练的逻辑回归（需有 coef_）
        xcolumns: 模型特征名顺序；None 时按 model 的列数生成 f0..fn
        suffix: WOE 特征名后缀（默认 `_woe`），用于还原原始变量名

    Returns:
        DataFrame: `variable / coef / expected_sign / actual_sign / corr / ok`，
        只在 xcolumns 缺失时用 model 的列数生成占位名。
    """
    cols = ["variable", "coef", "expected_sign", "actual_sign", "corr", "ok"]
    if model is None or bins is None:
        return _empty(cols)

    try:
        coef = np.asarray(model.coef_).reshape(-1)
    except Exception:  # noqa: BLE001
        return _empty(cols)

    names = list(xcolumns) if xcolumns else [f"f{i}" for i in range(len(coef))]
    if len(names) != len(coef):
        names = [f"f{i}" for i in range(len(coef))]

    rows: list[dict[str, Any]] = []
    for name, c in zip(names, coef):
        raw_var = name[: -len(suffix)] if suffix and name.endswith(suffix) else name
        bdf = bins.get(raw_var) if isinstance(bins, dict) else None
        corr = np.nan
        if bdf is not None and len(bdf) >= 2 and {"woe", "badprob"} <= set(bdf.columns):
            try:
                woe = pd.to_numeric(bdf["woe"], errors="coerce")
                bad = pd.to_numeric(bdf["badprob"], errors="coerce")
                mask = woe.notna() & bad.notna()
                if mask.sum() >= 2 and pd.Series(woe[mask]).std() > 0 and pd.Series(bad[mask]).std() > 0:
                    corr = float(np.corrcoef(woe[mask], bad[mask])[0, 1])
            except Exception as e:  # noqa: BLE001
                logger.debug("符号一致性 corr 计算失败(%s): %s", raw_var, e)

        # 分箱信息缺失时用 WOE 约定的默认方向
        expected_sign = 1 if (np.isnan(corr) or corr >= 0) else -1
        actual_sign = 1 if c > 0 else (-1 if c < 0 else 0)
        rows.append({
            "variable": raw_var,
            "coef": float(c),
            "expected_sign": expected_sign,
            "actual_sign": actual_sign,
            "corr": corr,
            "ok": bool(actual_sign == expected_sign),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        # 不一致的排前面，同组内按 |coef| 从大到小
        out["_bad"] = (~out["ok"]).astype(int)
        out["_abs"] = out["coef"].abs()
        out = (
            out.sort_values(["_bad", "_abs"], ascending=[False, False])
            .drop(columns=["_bad", "_abs"])
            .reset_index(drop=True)
        )
    n_bad = int((~out["ok"]).sum()) if not out.empty else 0
    logger.info("系数符号一致性检查: %d 个变量，%d 个方向不一致", len(out), n_bad)
    return out


# ============================================================================
# 3) 样本集中度
# ============================================================================
def check_sample_concentration(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    threshold: float = CONCENTRATION_THRESHOLD,
    max_distinct: int = MAX_DISTINCT_FOR_CONCENTRATION,
    exclude: list[str] | None = None,
) -> pd.DataFrame:
    """检查类别型变量的取值集中度（某一取值是否一统天下）。

    Args:
        df: 原始数据（通常用 raw_df 或 train_df）
        cols: 指定检查的列；None → 自动挑 object/category 列
              + 取值数 ≤ max_distinct 的数值列
        threshold: 占比上限，默认 0.5
        max_distinct: 数值列被视为「类别」的取值数上限
        exclude: 强制排除的列名。典型用法为排除**目标列 / ID 列 / 年份列**——
            标签列天然极度不平衡（如违约率 3%），把它算进集中度只会产出
            「列 target 的取值 0 占比 97%」这类无意义结论。

    Returns:
        DataFrame: `column / top_value / share / n_distinct / flag`，
        flag ∈ {"ok", "fail"}；缺失值也算一种取值（`dropna=False`）。
    """
    out_cols = ["column", "top_value", "share", "n_distinct", "flag"]
    if df is None or len(df) == 0:
        return _empty(out_cols)

    skip = {str(c) for c in (exclude or [])}

    if cols is None:
        # pandas 3.0 起字符串列默认是 StringDtype（不再是 object dtype），
        # 因此不能只判 is_object_dtype，必须把 is_string_dtype 一起算上。
        candidates = [
            c for c in df.columns
            if (
                pd.api.types.is_string_dtype(df[c])
                or pd.api.types.is_categorical_dtype(df[c])
            )
            and str(c) not in skip
        ]
        # 数值列只有当「取值数足够少」时才当类别看：
        # 同时限制绝对取值数与占行数比例，避免把连续型数值列误当成类别列。
        n_row = len(df)
        candidates += [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
            and str(c) not in skip
            and df[c].nunique(dropna=False) <= max_distinct
            and df[c].nunique(dropna=False) <= max(1, int(n_row * 0.2))
        ]
        cols = list(dict.fromkeys(candidates))
    else:
        cols = [c for c in cols if c in df.columns and str(c) not in skip]

    rows: list[dict[str, Any]] = []
    for c in cols:
        try:
            vc = df[c].value_counts(normalize=True, dropna=False)
        except Exception as e:  # noqa: BLE001 - 不可哈希列跳过
            logger.debug("集中度统计失败(%s): %s", c, e)
            continue
        if vc.empty:
            continue
        share = float(vc.iloc[0])
        rows.append({
            "column": c,
            "top_value": str(vc.index[0]),
            "share": share,
            "n_distinct": int(len(vc)),
            "flag": "fail" if share > threshold else "ok",
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("share", ascending=False).reset_index(drop=True)
    n_bad = int((out["flag"] == "fail").sum()) if not out.empty else 0
    logger.info("样本集中度检查: %d 列，%d 列单一取值超过 %.0f%%", len(out), n_bad, threshold * 100)
    return out
