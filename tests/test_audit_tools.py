"""tests.test_audit_tools — Critic 的三项确定性审计（模块 3）

覆盖：
1. compute_vif：完全共线性→inf、高相关→fail、独立→ok、常量列与非数值列处理
2. check_coef_consistency：符号一致/不一致判定、缺失分箱时的默认方向、名称后缀还原
3. check_sample_concentration：占比超阈值报警、缺失值计入、低基数数值列自动纳入
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tools.audit_tools import (
    check_coef_consistency,
    check_sample_concentration,
    compute_vif,
)

RNG = np.random.default_rng(20260915)


# ============================================================================
# compute_vif
# ============================================================================
def test_vif_detects_perfect_collinearity():
    x = RNG.normal(size=200)
    X = pd.DataFrame({"a": x, "b": x * 2.0 + 1.0, "c": RNG.normal(size=200)})
    out = compute_vif(X)
    flags = dict(zip(out["variable"], out["flag"]))
    assert flags["a"] == "fail" and flags["b"] == "fail"
    assert np.isinf(out.loc[out["variable"] == "a", "vif"].iloc[0])
    assert flags["c"] == "ok"


def test_vif_ok_for_independent_columns():
    X = pd.DataFrame({f"f{i}": RNG.normal(size=500) for i in range(4)})
    out = compute_vif(X)
    assert (out["flag"] == "ok").all()
    assert (out["vif"] < 5).all()


def test_vif_handles_constant_column():
    X = pd.DataFrame({
        "const": np.ones(100),
        "x": RNG.normal(size=100),
        "y": RNG.normal(size=100),
    })
    out = compute_vif(X)
    row = out.loc[out["variable"] == "const"].iloc[0]
    assert row["flag"] == "fail" and np.isinf(row["vif"])


def test_vif_drops_non_numeric():
    X = pd.DataFrame({
        "cat": ["a", "b", "c"] * 50,
        "n1": RNG.normal(size=150),
        "n2": RNG.normal(size=150),
    })
    out = compute_vif(X)
    assert "cat" not in set(out["variable"])
    assert len(out) == 2


def test_vif_fills_nan():
    x = RNG.normal(size=200)
    X = pd.DataFrame({"a": x, "b": x * 3.0})
    X.loc[0:10, "a"] = np.nan
    out = compute_vif(X)
    assert len(out) == 2 and out["vif"].notna().all()


def test_vif_empty_input():
    out = compute_vif(pd.DataFrame())
    assert list(out.columns) == ["variable", "vif", "flag"]
    assert out.empty


def test_vif_single_column_returns_empty():
    out = compute_vif(pd.DataFrame({"a": RNG.normal(size=10)}))
    assert out.empty


def test_vif_respects_max_columns():
    X = pd.DataFrame({f"f{i}": RNG.normal(size=100) for i in range(8)})
    out = compute_vif(X, max_columns=3)
    assert len(out) == 3


# ============================================================================
# check_coef_consistency
# ============================================================================
def _bins(var: str, woe: list[float], badprob: list[float]) -> dict:
    return {var: pd.DataFrame({"woe": woe, "badprob": badprob})}


class _Model:
    def __init__(self, coefs):
        self.coef_ = np.asarray([coefs])


def test_coef_consistent_when_positive():
    bins = _bins("ROE", [-1.0, 0.0, 1.0], [0.01, 0.05, 0.20])
    out = check_coef_consistency(bins, _Model([2.0]), ["ROE_woe"])
    assert out.loc[0, "expected_sign"] == 1
    assert out.loc[0, "actual_sign"] == 1
    assert bool(out.loc[0, "ok"]) is True


def test_coef_flagged_when_negative():
    bins = _bins("ROE", [-1.0, 0.0, 1.0], [0.01, 0.05, 0.20])
    out = check_coef_consistency(bins, _Model([-3.5]), ["ROE_woe"])
    assert bool(out.loc[0, "ok"]) is False
    assert out.loc[0, "actual_sign"] == -1


def test_coef_falls_back_to_woe_convention_without_bins():
    """分箱缺失时按 scorecardpy 约定（WOE 越大越坏）判定期望符号为 +1。"""
    out = check_coef_consistency({}, _Model([-1.0, 2.0]), ["a_woe", "b_woe"])
    assert set(out["expected_sign"]) == {1}
    assert bool(out.loc[out["variable"] == "a", "ok"].iloc[0]) is False
    assert bool(out.loc[out["variable"] == "b", "ok"].iloc[0]) is True


def test_coef_strips_woe_suffix():
    bins = _bins("X", [0.0, 1.0], [0.01, 0.2])
    out = check_coef_consistency(bins, _Model([1.0]), ["X_woe"])
    assert out.loc[0, "variable"] == "X"


def test_coef_custom_suffix():
    bins = _bins("X", [0.0, 1.0], [0.01, 0.2])
    out = check_coef_consistency(bins, _Model([1.0]), ["X__w"], suffix="__w")
    assert out.loc[0, "variable"] == "X"


def test_coef_sorted_inconsistent_first_then_by_abs():
    bins = {
        "weak": pd.DataFrame({"woe": [0.0, 1.0], "badprob": [0.01, 0.2]}),
        "strong": pd.DataFrame({"woe": [0.0, 1.0], "badprob": [0.01, 0.2]}),
        "ok1": pd.DataFrame({"woe": [0.0, 1.0], "badprob": [0.01, 0.2]}),
    }
    out = check_coef_consistency(bins, _Model([-0.5, -9.0, 3.0]),
                                ["weak_woe", "strong_woe", "ok1_woe"])
    assert list(out["variable"]) == ["strong", "weak", "ok1"]


def test_coef_missing_badprob_column():
    bins = {"X": pd.DataFrame({"woe": [0.0, 1.0]})}
    out = check_coef_consistency(bins, _Model([1.0]), ["X_woe"])
    assert bool(out.loc[0, "ok"]) is True      # 无法确定方向 → 按约定判 +，正值即通过


def test_coef_no_model():
    out = check_coef_consistency({}, None, None)
    assert out.empty


# ============================================================================
# check_sample_concentration
# ============================================================================
def test_concentration_flags_dominant_value():
    df = pd.DataFrame({"IndustrySector": ["A"] * 80 + ["B"] * 10 + ["C"] * 10})
    out = check_sample_concentration(df, cols=["IndustrySector"])
    row = out.iloc[0]
    assert row["share"] == 0.8
    assert row["flag"] == "fail"
    assert row["top_value"] == "A"


def test_concentration_ok_when_balanced():
    df = pd.DataFrame({"g": ["A"] * 50 + ["B"] * 50})
    out = check_sample_concentration(df, cols=["g"])
    assert out.iloc[0]["flag"] == "ok"


def test_concentration_counts_nan_as_value():
    df = pd.DataFrame({"g": ["A"] * 60 + [None] * 40})
    out = check_sample_concentration(df, cols=["g"])
    assert out.iloc[0]["share"] == 0.6
    assert out.iloc[0]["flag"] == "fail"


def test_concentration_auto_picks_categorical():
    df = pd.DataFrame({
        "city": ["SH", "BJ", "SZ"] * 10,
        "score": RNG.normal(size=30),          # 连续值（取值数=行数）→ 不检查
        "grade": [1, 2] * 15,                   # 低基数数值 → 检查
    })
    out = check_sample_concentration(df)
    cols = set(out["column"])
    assert "city" in cols and "grade" in cols
    assert "score" not in cols


def test_concentration_includes_low_cardinality_float_flag():
    """形如 0/1 的浮点标志列也算类别（取值数远少于行数）。"""
    n = 1000
    df = pd.DataFrame({"is_st": [0.0] * 900 + [1.0] * 100})
    out = check_sample_concentration(df)
    row = out.loc[out["column"] == "is_st"].iloc[0]
    assert row["share"] == 0.9 and row["flag"] == "fail"
    assert len(df) == n


def test_concentration_custom_threshold():
    df = pd.DataFrame({"g": ["A"] * 40 + ["B"] * 60})
    assert check_sample_concentration(df, cols=["g"], threshold=0.3).iloc[0]["flag"] == "fail"
    assert check_sample_concentration(df, cols=["g"], threshold=0.7).iloc[0]["flag"] == "ok"


def test_concentration_ignores_unknown_cols():
    df = pd.DataFrame({"a": [1, 2, 3]})
    out = check_sample_concentration(df, cols=["not_exist"])
    assert out.empty


def test_concentration_exclude_skips_target_and_id():
    """标签列天然不平衡（违约率 ~3%），必须能排除掉，否则全是噪音结论。"""
    df = pd.DataFrame({
        "is_default": [0] * 970 + [1] * 30,
        "Symbol": ["000001"] * 1000,
        "IndustrySector": ["A"] * 500 + ["B"] * 500,
    })
    out = check_sample_concentration(df, exclude=["is_default", "Symbol"])
    cols = list(out["column"])
    assert "is_default" not in cols
    assert "Symbol" not in cols
    assert "IndustrySector" in cols


def test_concentration_exclude_applies_to_auto_pick():
    df = pd.DataFrame({
        "is_default": [0] * 90 + [1] * 10,     # 低基数数值 → 会被自动挑中
        "city": ["SH"] * 80 + ["BJ"] * 20,
    })
    assert "is_default" in list(check_sample_concentration(df)["column"])
    out = check_sample_concentration(df, exclude=["is_default"])
    assert list(out["column"]) == ["city"]


def test_concentration_exclude_beats_explicit_cols():
    """显式指定的列若命中 exclude，同样被剔除（避免调用方自己忘排除 target）。"""
    df = pd.DataFrame({"is_default": [0] * 90 + [1] * 10, "g": ["A"] * 60 + ["B"] * 40})
    out = check_sample_concentration(df, cols=["is_default", "g"], exclude=["is_default"])
    assert list(out["column"]) == ["g"]


def test_concentration_empty_df():
    out = check_sample_concentration(pd.DataFrame())
    assert out.empty
    assert list(out.columns) == ["column", "top_value", "share", "n_distinct", "flag"]
