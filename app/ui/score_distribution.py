"""app.ui.score_distribution —— 评分分布可视化。"""
from __future__ import annotations

import pandas as pd
import streamlit as st


def render_score_distribution(scored_df: pd.DataFrame, bins: int = 20) -> None:
    """渲染评分分布直方图。"""
    if scored_df is None or "score" not in scored_df.columns:
        st.info("暂无评分数据")
        return

    scores = scored_df["score"].dropna()
    if len(scores) == 0:
        st.info("评分列为空")
        return

    hist, edges = pd.cut(scores, bins=bins, retbins=True)
    dist = hist.value_counts().sort_index().reset_index()
    dist.columns = ["分数区间", "样本数"]
    dist["区间中点"] = dist["分数区间"].apply(lambda x: round(x.mid, 1))

    st.bar_chart(dist.set_index("区间中点")["样本数"])
    st.caption(f"评分范围: {scores.min():.0f} ~ {scores.max():.0f}，均值 {scores.mean():.2f}")
