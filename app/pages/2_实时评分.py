"""app.pages.2_实时评分 / 加载评分卡并对新样本打分。"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.llm import SCORE_EXPLANATION, get_model
from tools import scorecard_ply

st.set_page_config(page_title="实时评分", page_icon="⚡", layout="wide")

st.title("⚡ 实时评分")

# ---- 加载评分卡 ----
scorecard_file = st.sidebar.file_uploader("上传评分卡 (scorecard.pkl)", type=["pkl"])
cutoff = st.sidebar.number_input(
    "通过阈值 cutoff", 300, 1000, 600, 10,
    help="分数大于等于该值时，决策建议「通过」",
)
mode = st.radio("输入模式", ["单笔表单", "批量 CSV"], horizontal=True)

if scorecard_file is None:
    st.info("请在左侧上传训练好的 scorecard.pkl。")
    st.stop()

try:
    scorecard = pickle.loads(scorecard_file.getvalue())
    variables = list(scorecard.keys())
    st.sidebar.caption(f"评分卡变量数: {len(variables)}")
except Exception as e:
    st.error(f"评分卡加载失败: {e}")
    st.stop()


# ---- 单笔表单 ----
if mode == "单笔表单":
    st.subheader("填写申请人信息")
    with st.form("scoring_form"):
        inputs: dict[str, Any] = {}
        # 3-column grid，但每变量自带 helper 区域，整体不是「3 等宽 feature card」
        cols = st.columns(3)
        for i, var in enumerate(variables):
            with cols[i % 3]:
                inputs[var] = st.number_input(
                    var,
                    value=None,
                    placeholder="输入数值",
                    key=f"input_{var}",
                    help=(
                        "根据训练数据的该变量分布填写。数值列输入整数或浮点数；"
                        "类别列请使用训练时的编码值。"
                    ),
                )
        submitted = st.form_submit_button("开始打分", type="primary", use_container_width=True)

    if submitted:
        # 校验必填
        missing = [v for v, val in inputs.items() if val is None]
        if missing:
            st.error(f"以下变量未填写: {missing}")
            st.stop()

        sample_df = pd.DataFrame([inputs])
        scored = scorecard_ply(sample_df, scorecard, only_total_score=False)
        total_score = int(scored["score"].iloc[0])
        passed = total_score >= cutoff

        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("最终评分", total_score)
        col2.metric("建议", "通过" if passed else "拒绝")

        var_score_cols = [c for c in scored.columns if c.startswith("score_")]
        if var_score_cols:
            var_scores = {c.replace("score_", ""): int(scored[c].iloc[0]) for c in var_score_cols}
            st.write("变量得分", pd.DataFrame([var_scores]))

            if st.button("LLM 解释评分", key="explain_score"):
                try:
                    llm = get_model()
                    prompt = SCORE_EXPLANATION.format(
                        variable_scores_json=json.dumps(var_scores, ensure_ascii=False, indent=2),
                        total_score=total_score,
                        cutoff=cutoff,
                    )
                    with st.spinner("LLM 思考中..."):
                        response = llm.invoke(prompt)
                    st.markdown(response.content)
                except Exception as e:
                    st.error(f"LLM 调用失败: {e}")

# ---- 批量 CSV ----
else:
    st.subheader("批量打分")
    batch_file = st.file_uploader("上传待打分 CSV / Excel", type=["csv", "xlsx", "xls"])
    if batch_file is not None:
        suffix = Path(batch_file.name).suffix.lower()
        batch_df = pd.read_csv(batch_file) if suffix == ".csv" else pd.read_excel(batch_file)
        st.write("待打样本", batch_df.head(20))

        if st.button("批量打分", type="primary", use_container_width=True):
            try:
                scored = scorecard_ply(batch_df, scorecard, only_total_score=True)
                result_df = batch_df.copy()
                result_df["score"] = scored["score"].values
                result_df["decision"] = result_df["score"].apply(
                    lambda s: "通过" if s >= cutoff else "拒绝"
                )

                st.write("打分结果", result_df)
                st.download_button(
                    label="下载结果 (CSV)",
                    data=result_df.to_csv(index=False).encode("utf-8"),
                    file_name="scoring_result.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"批量打分失败: {e}")
