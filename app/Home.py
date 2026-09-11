"""app.Home.py —— Streamlit 应用入口。"""
from __future__ import annotations

import os

import streamlit as st

from app.llm import list_models, resolve_default_label

st.set_page_config(
    page_title="Credit Scorecard Agent",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏦 企业信用评分卡自动化建模 Agent")
st.markdown("""
本应用为企业信用评分卡自动化建模项目：
- **页面 1：模型训练** / 上传 CSV，一键完成 WOE 分箱、逻辑回归、评分卡构建与护栏检查
- **页面 2：实时评分** / 加载已训练评分卡，对单笔或批量样本打分

所有数值计算由确定性 Python 工具完成，LLM 仅用于结果解释与建议。
""")

# 侧边栏：模型选择
st.sidebar.title("⚙️ 设置")

if not os.getenv("LLM_PRESETS"):
    st.sidebar.warning("未检测到 `.env` 中的 LLM_PRESETS。请在 credit_agent 目录创建 .env 并配置模型。")
    selected_model = None
else:
    models = list_models()
    default_label = resolve_default_label()
    selected_model = st.sidebar.selectbox(
        "选择 LLM 模型",
        options=models,
        index=models.index(default_label) if default_label in models else 0,
        key="selected_model",
    )
    st.sidebar.caption(f"当前默认: {default_label}")

