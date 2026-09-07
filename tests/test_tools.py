"""端到端冒烟测试 — 不依赖网络下载数据，使用合成数据 + scorecardpy 内置 german 数据。

运行:
    cd /workspace && python3 -m credit_agent.tests.test_tools
或:
    python3 /workspace/credit_agent/tests/test_tools.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 将 /workspace 加入 sys.path，确保导入可解析
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _load_german():
    """优先使用 scorecardpy 内置 germancredit 数据。"""
    try:
        import scorecardpy as sc
        df = sc.germancredit()
        df = df.rename(columns={"creditability": "creditability"})
        df["creditability"] = (df["creditability"] == "good").astype(int)
        return df
    except Exception as e:
        print(f"[warn] scorecardpy.germancredit() 加载失败 ({e})，使用合成数据")
        rng = np.random.default_rng(7)
        df = pd.DataFrame({
            "duration": rng.integers(6, 72, 500),
            "amount": rng.integers(250, 20000, 500),
            "age": rng.integers(19, 75, 500),
            "creditability": rng.choice([0, 1], 500, p=[0.3, 0.7]),
        })
        return df


def test_data_tools():
    from credit_agent.tools import (
        load_data, explore_data, split_dataset,
        handle_missing, handle_outliers,
    )
    df = _load_german()
    print(f"[data] 加载 shape={df.shape}")

    summary = explore_data(df, target="creditability")
    print(f"[data] EDA: rows={summary['n_rows']}, cols={summary['n_cols']}, "
          f"target_dist={summary['target_distribution']}")

    train, test = split_dataset(df, target="creditability", test_size=0.3)
    assert len(train) + len(test) == len(df)
    print(f"[data] split: train={len(train)}, test={len(test)}")

    cleaned = handle_missing(train, strategy="median")
    print(f"[data] 缺失值处理后总缺失数={cleaned.isnull().sum().sum()}")

    num_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != "creditability"]
    capped = handle_outliers(cleaned, columns=num_cols[:2], n_sigma=4.0)
    print(f"[data] 异常值盖帽完成: shape={capped.shape}")


def test_feature_tools():
    from credit_agent.tools import woebin, woebin_ply, compute_iv
    df = _load_german()
    # 仅使用数值列演示
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    sample = df[num_cols].dropna().head(800)
    bins = woebin(sample, target="creditability", max_bins=5, method="quantile")
    print(f"[feature] 分箱完成: {len(bins)} 个变量")
    for var, bdf in list(bins.items())[:2]:
        print(f"[feature] {var}: total_iv={bdf['total_iv'].iloc[0]:.4f}")

    woe = woebin_ply(sample, bins)
    print(f"[feature] WOE 转换: shape={woe.shape}")

    iv = compute_iv(sample, num_cols[0] if num_cols else "amount", "creditability")
    print(f"[feature] 单变量 IV={iv:.4f}")


def test_model_tools():
    from credit_agent.tools import (
        model_train, model_predict, evaluate_performance,
    )
    df = _load_german()
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "creditability"]
    X = df[num_cols].fillna(df[num_cols].median())
    y = df["creditability"]
    model = model_train(X, y)
    proba = model_predict(model, X)
    metrics = evaluate_performance(y, proba)
    print(f"[model] 训练指标: {metrics}")
    assert 0.0 <= metrics["ks"] <= 1.0
    assert 0.5 <= metrics["auc"] <= 1.0


def test_scorecard_tools():
    from credit_agent.tools import (
        woebin, woebin_ply, model_train, build_scorecard, scorecard_ply,
    )
    df = _load_german()
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "creditability"]
    sample = df[num_cols + ["creditability"]].dropna().head(800)

    bins = woebin(sample, target="creditability", max_bins=5)
    woe = woebin_ply(sample, bins)
    y = sample["creditability"]
    model = model_train(woe, y)
    xcols = list(woe.columns)
    card = build_scorecard(
        bins, model,
        xcolumns=xcols,
        base_score=600, pdo=20,
    )
    scored = scorecard_ply(sample.head(10), card)
    print(f"[scorecard] 打分结果: score 列均值={scored['score'].mean():.2f}, "
          f"最小={scored['score'].min():.2f}, 最大={scored['score'].max():.2f}")
    assert "score" in scored.columns


def test_validation_tools():
    from credit_agent.tools import compute_psi, guardrail_check
    rng = np.random.default_rng(0)
    expected = rng.normal(600, 50, 5000)
    actual_stable = rng.normal(600, 50, 5000)
    actual_drift = rng.normal(640, 60, 5000)

    psi_ok = compute_psi(expected, actual_stable)
    psi_bad = compute_psi(expected, actual_drift)
    print(f"[validation] PSI 稳定={psi_ok:.4f}, 漂移={psi_bad:.4f}")
    assert psi_bad > psi_ok

    res_ok = guardrail_check({"ks": 0.45, "auc": 0.8})
    print(f"[validation] 通过: passed={res_ok.passed}, hitl={res_ok.requires_human_review}")

    res_warn = guardrail_check(
        {"ks": 0.25, "auc": 0.75},
        reference_dist=expected, current_dist=actual_drift,
    )
    print(f"[validation] 警告: passed={res_warn.passed}, hitl={res_warn.requires_human_review}, "
          f"warnings={res_warn.warnings}")
    assert res_warn.requires_human_review


if __name__ == "__main__":
    print("=" * 60)
    print("信用评分卡工具集 — 冒烟测试")
    print("=" * 60)
    test_data_tools()
    print("-" * 60)
    test_feature_tools()
    print("-" * 60)
    test_model_tools()
    print("-" * 60)
    test_scorecard_tools()
    print("-" * 60)
    test_validation_tools()
    print("=" * 60)
    print("所有冒烟测试通过")