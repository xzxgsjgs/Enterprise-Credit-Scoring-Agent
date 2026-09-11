"""app.ui.metrics_card —— 指标卡片组件。"""
from __future__ import annotations

import streamlit as st


def render_metrics_cards(metrics: dict, train_key: str = "train", test_key: str = "test") -> None:
    """渲染训练集/测试集 KS、AUC、Gini、PSI 指标卡片。"""
    train = metrics.get(train_key, {}) if isinstance(metrics, dict) else {}
    test = metrics.get(test_key, {}) if isinstance(metrics, dict) else {}
    psi = metrics.get("psi") if isinstance(metrics, dict) else None

    cols = st.columns(4)
    with cols[0]:
        st.metric(
            label="KS (test)",
            value=f"{test.get('ks', 0):.3f}",
            delta=f"train: {train.get('ks', 0):.3f}",
        )
    with cols[1]:
        st.metric(
            label="AUC (test)",
            value=f"{test.get('auc', 0):.3f}",
            delta=f"train: {train.get('auc', 0):.3f}",
        )
    with cols[2]:
        st.metric(
            label="Gini (test)",
            value=f"{test.get('gini', 0):.3f}",
            delta=f"train: {train.get('gini', 0):.3f}",
        )
    with cols[3]:
        st.metric(
            label="PSI",
            value=f"{psi:.3f}" if psi is not None else "N/A",
        )
