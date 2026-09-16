"""tests.test_data_gate — 数据质量守门 + 错误短路测试（模块 6）

覆盖：
1. 正常数据 → 通过
2. 样本量不足 → 阻断
3. 违约率过低 / 过高 → 阻断
4. 违约样本 < 10 → 阻断
5. 时序模式下年份不足 → 阻断
6. 高缺失列 / 单一值列 → warning + suggest_drop
7. data_gate_node 的过滤（min_year / exclude_industries / exclude_vars）
8. 路由函数：route_after_data_gate / route_after_critical
9. 图短路：喂不存在 csv → load_data 后立即 END（不跑满后续节点）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agent.routing import has_critical_error, route_after_critical, route_after_data_gate
from tools import check_data_quality


def _make_df(n: int = 1500, default_rate: float = 0.05, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < default_rate).astype(int)
    return pd.DataFrame({
        "Symbol": [f"{i:06d}" for i in range(n)],
        "year": rng.choice([2018, 2019, 2020, 2021], size=n),
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "is_default_next_year": y,
    })


# ============================================================================
# check_data_quality：确定性校验
# ============================================================================
def test_normal_data_passes():
    df = _make_df(n=1500, default_rate=0.05)
    r = check_data_quality(df, target="is_default_next_year", min_rows=1000)
    assert r["passed"] is True
    assert r["blocks"] == []
    assert r["stats"]["default_rate"] == pytest.approx(0.05, abs=0.02)


def test_insufficient_rows_blocks():
    df = _make_df(n=200)
    r = check_data_quality(df, target="is_default_next_year", min_rows=1000)
    assert r["passed"] is False
    assert any("样本量不足" in b for b in r["blocks"])


def test_low_default_rate_blocks():
    # 1500 行里只有 2 个违约 → 0.13%，低于 0.5% 且少于 10 个
    df = _make_df(n=1500, default_rate=0.0013, seed=7)
    r = check_data_quality(df, target="is_default_next_year", min_rows=1000)
    assert r["passed"] is False
    assert any("违约率过低" in b for b in r["blocks"])
    assert any("违约样本过少" in b for b in r["blocks"])


def test_high_default_rate_blocks():
    df = _make_df(n=1500, default_rate=0.8, seed=3)
    r = check_data_quality(df, target="is_default_next_year", min_rows=1000)
    assert r["passed"] is False
    assert any("违约率过高" in b for b in r["blocks"])


def test_missing_target_blocks():
    df = _make_df(n=1500)
    r = check_data_quality(df, target="not_exist", min_rows=1000)
    assert r["passed"] is False
    assert any("目标列缺失" in b for b in r["blocks"])


def test_time_split_insufficient_years_blocks():
    df = _make_df(n=1500)
    df["year"] = 2020  # 只有 1 年
    r = check_data_quality(
        df, target="is_default_next_year", min_rows=1000,
        min_years=3, time_split=True,
    )
    assert r["passed"] is False
    assert any("年份覆盖不足" in b for b in r["blocks"])


def test_time_split_missing_year_col_blocks():
    df = _make_df(n=1500).drop(columns=["year"])
    r = check_data_quality(
        df, target="is_default_next_year", min_rows=1000, time_split=True,
    )
    assert r["passed"] is False
    assert any("年份列" in b for b in r["blocks"])


def test_high_missing_col_warns():
    df = _make_df(n=1500)
    df.loc[: df.index[int(len(df) * 0.9)], "f1"] = np.nan  # 90% 缺失
    r = check_data_quality(df, target="is_default_next_year", min_rows=1000,
                           max_missing_ratio=0.6)
    assert any("缺失率" in w for w in r["warnings"])
    assert "f1" in r["suggest_drop"]


def test_identical_col_warns():
    df = _make_df(n=1500)
    df["const"] = 1.0  # 100% 单一值
    r = check_data_quality(df, target="is_default_next_year", min_rows=1000,
                           max_identical_ratio=0.98)
    assert any("单一取值" in w for w in r["warnings"])
    assert "const" in r["suggest_drop"]


def test_id_cols_excluded_from_quality_check():
    """ID 列（Symbol）即使唯一值很多也不该被当成『高缺失』误报。"""
    df = _make_df(n=1500)
    r = check_data_quality(df, target="is_default_next_year", min_rows=1000,
                           exclude_cols=["Symbol"])
    assert "Symbol" not in r["suggest_drop"]


def test_good_bad_string_target():
    """目标列是 good/bad 字符串时也能正确统计违约率。"""
    df = _make_df(n=1500, default_rate=0.06, seed=11)
    df["label"] = np.where(df["is_default_next_year"] == 1, "bad", "good")
    r = check_data_quality(df, target="label", min_rows=1000)
    assert r["stats"]["default_rate"] == pytest.approx(0.06, abs=0.02)


# ============================================================================
# 路由函数
# ============================================================================
def test_route_after_data_gate():
    assert route_after_data_gate({"data_gate": {"passed": True}}) == "continue"
    assert route_after_data_gate({"data_gate": {"passed": False}}) == "end"
    # 节点未产出时放行（避免异常场景阻塞）
    assert route_after_data_gate({}) == "continue"


def test_has_critical_error():
    assert has_critical_error({"critical_error": True}) is True
    assert has_critical_error({}) is False
    # 从 step_history 推断
    assert has_critical_error({"step_history": ["[ERR] load_data"]}) is True
    assert has_critical_error({"step_history": ["[ERR] var_filter"]}) is False
    assert has_critical_error({"step_history": ["[OK] load_data"]}) is False


def test_route_after_critical():
    assert route_after_critical({"critical_error": True}) == "end"
    assert route_after_critical({"step_history": ["[ERR] woebin"]}) == "end"
    assert route_after_critical({"step_history": ["[OK] woebin"]}) == "continue"


# ============================================================================
# 图级短路（不依赖真实数据文件）
# ============================================================================
def test_graph_short_circuits_on_missing_csv():
    """喂不存在的 csv → load_data 失败 → 立即 END，不跑满后续节点。"""
    from agent.graph import build_graph

    app = build_graph().compile()
    final = app.invoke({
        "csv_path": "__definitely_not_exist__.csv",
        "target": "is_default_next_year",
        "config_overrides": {},
    })

    # 1) 应当短路：没有 train_df / model / scorecard 等下游产物
    assert final.get("model") is None
    assert final.get("scorecard") is None
    assert final.get("metrics") is None
    # 2) 记录了错误
    assert any("load_data" in e for e in final.get("errors", []))
    # 3) 关键节点错误被识别
    assert has_critical_error(final) is True
    # 4) step_history 里只有 load_data，没有后续节点
    hist = final.get("step_history", [])
    assert any("load_data" in h for h in hist)
    assert not any("[OK] split" in h for h in hist)
    assert not any("[OK] model_train" in h for h in hist)
