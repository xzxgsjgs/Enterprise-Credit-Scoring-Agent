"""tests.test_cutoff_tools — cut-off 择优与分数分档（模块 5）

覆盖：
1. optimize_cutoff 三种 method（ks / approve_rate / bad_rate）
2. 完美分离样本的 KS 切点落在两类之间
3. target 缺失时抛 ValueError
4. 单类别标签 / 空样本降级为 warnings 而非崩溃
5. 语义方向：分数越高越安全，拒绝 score < cutoff
6. grade_scores 默认等频分位、自定义分界、labels 过少报错
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.cutoff_tools import grade_scores, optimize_cutoff

RNG = np.random.default_rng(20260911)


def _two_class_scores(n_good: int = 900, n_bad: int = 100, sep: float = 300.0):
    """构造「高分段全好、低分段全坏」的样本：good ~ N(700,30)，bad ~ N(400,30)。"""
    good = RNG.normal(700.0, 30.0, n_good)
    bad = RNG.normal(700.0 - sep, 30.0, n_bad)
    score = np.concatenate([good, bad])
    y = np.concatenate([np.zeros(n_good), np.ones(n_bad)])
    return y, score


# ============================================================================
# optimize_cutoff — ks
# ============================================================================
def test_ks_method_finds_balanced_cutoff():
    y, s = _two_class_scores()
    res = optimize_cutoff(y, s, method="ks")
    # 切点应落在 400 与 700 之间，且 KS 接近 1（两类几乎完全分离）
    assert 400.0 < res["cutoff"] < 700.0
    assert res["ks_at_cutoff"] > 0.9
    assert res["method"] == "ks"


def test_ks_cutoff_semantics_higher_is_safer():
    """语义：拒绝 score < cutoff，通过 score >= cutoff。"""
    y, s = _two_class_scores()
    res = optimize_cutoff(y, s, method="ks")
    t = res["cutoff"]
    approved = s >= t
    rejected = s < t
    # 通过人群的坏率必须显著低于被拒人群，否则方向就反了
    assert y[approved].mean() < y[rejected].mean()
    # 通过率应与返回的 approve_rate 一致
    assert abs(approved.mean() - res["approve_rate"]) < 1e-9


def test_capture_rate_and_lift_consistent():
    y, s = _two_class_scores()
    res = optimize_cutoff(y, s, method="ks")
    n_bad = y.sum()
    approved = s >= res["cutoff"]
    rejected = ~approved
    expected_capture = y[rejected].sum() / n_bad
    assert abs(res["capture_rate"] - expected_capture) < 1e-9
    base = y.mean()
    after = y[approved].mean()
    if after > 0:
        assert abs(res["lift"] - base / after) < 1e-6
        assert res["lift"] > 1.0  # 筛选有效
    else:
        # 完美分离时通过后坏率为 0，lift 记 inf（报告会附「Lift 数值失真」提示）
        assert res["lift"] == float("inf")


def test_table_has_grid_rows():
    y, s = _two_class_scores()
    res = optimize_cutoff(y, s)
    tbl = res["table"]
    assert isinstance(tbl, pd.DataFrame)
    assert len(tbl) > 10
    # 表格中 KS 最大行必须就是选中的切点
    best = tbl.loc[tbl["ks"].idxmax()]
    assert abs(best["cutoff"] - res["cutoff"]) < 1e-9


# ============================================================================
# optimize_cutoff — approve_rate / bad_rate
# ============================================================================
def test_approve_rate_method_hits_target():
    y, s = _two_class_scores()
    res = optimize_cutoff(y, s, method="approve_rate", target=0.7)
    assert abs(res["approve_rate"] - 0.7) < 0.03
    assert any("目标通过率" in w for w in res["warnings"])


def test_bad_rate_method_hits_target():
    y, s = _two_class_scores()
    base = optimize_cutoff(y, s, method="ks")
    target = float(base["table"].loc[30, "bad_rate_after"])
    res = optimize_cutoff(y, s, method="bad_rate", target=target)
    assert abs(res["bad_rate_after"] - target) < 1e-9
    assert any("目标通过后坏率" in w for w in res["warnings"])


def test_missing_target_raises():
    y, s = _two_class_scores()
    with pytest.raises(ValueError, match="必须提供 target"):
        optimize_cutoff(y, s, method="approve_rate")
    with pytest.raises(ValueError, match="必须提供 target"):
        optimize_cutoff(y, s, method="bad_rate")


def test_ks_method_allows_no_target():
    y, s = _two_class_scores()
    res = optimize_cutoff(y, s, method="ks")
    assert res["cutoff"] is not None


# ============================================================================
# 边界 / 降级
# ============================================================================
def test_single_class_label_degrades():
    y = np.zeros(200)
    s = RNG.normal(600, 30, 200)
    res = optimize_cutoff(y, s)
    assert res["cutoff"] is None
    assert any("单一类别" in w for w in res["warnings"])


def test_empty_sample_degrades():
    res = optimize_cutoff([], [])
    assert res["cutoff"] is None
    assert any("样本为空" in w for w in res["warnings"])


def test_nan_pairs_are_dropped():
    y = np.array([0.0, 1.0, 0.0, 1.0, np.nan, 1.0])
    s = np.array([700.0, 400.0, 680.0, 420.0, 650.0, np.nan])
    res = optimize_cutoff(y, s)
    # NaN 成对剔除后有效样本为 4 行
    tbl = res["table"]
    assert abs(tbl["approve_rate"].max() - 0.75) < 1e-9   # 3 / 4
    # 「全部通过」是退化端点，被有意排除出择优候选，故最多通过 3 行
    assert tbl["n_approved"].max() == 3
    assert res["cutoff"] is not None


def test_custom_grid_size():
    y, s = _two_class_scores()
    res = optimize_cutoff(y, s, n_grid=5)
    assert len(res["table"]) <= 5


def test_pandas_series_input_accepted():
    y, s = _two_class_scores()
    res = optimize_cutoff(pd.Series(y), pd.Series(s))
    assert res["cutoff"] is not None


# ============================================================================
# grade_scores
# ============================================================================
def test_grade_default_quantile():
    s = RNG.normal(600, 50, 1000)
    out = grade_scores(s)
    assert list(out.columns) == ["score", "grade"]
    assert len(out) == 1000
    counts = out["grade"].value_counts()
    assert set(counts.index) <= {"A", "B", "C", "D"}
    # 四档近似等频
    for g in ("A", "B", "C", "D"):
        assert 200 < counts.get(g, 0) < 300


def test_grade_monotonic_high_score_is_best():
    s = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    out = grade_scores(s, cutoffs=[250.0, 350.0, 450.0])
    grades = list(out["grade"])
    assert grades == ["D", "D", "C", "B", "A"]


def test_grade_custom_labels():
    s = np.array([1.0, 2.0, 3.0, 4.0])
    out = grade_scores(s, cutoffs=[2.0, 3.0], labels=("优", "良", "差"))
    assert out["grade"].iloc[-1] == "优"
    assert out["grade"].iloc[0] == "差"
    assert set(out["grade"]) == {"优", "良", "差"}


def test_grade_labels_too_few_raises():
    with pytest.raises(ValueError, match="至少需要 2 个档位"):
        grade_scores([1.0, 2.0], labels=("A",))


def test_grade_handles_nan():
    s = pd.Series([100.0, np.nan, 500.0, 900.0])
    out = grade_scores(s)
    assert len(out) == 4
    # NaN 行保留在表中（grade 为最差档），不静默丢样本
    assert out["score"].isna().sum() == 1


def test_grade_empty_series():
    out = grade_scores(pd.Series([], dtype=float))
    assert len(out) == 0
    assert list(out.columns) == ["score", "grade"]
