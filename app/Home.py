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
st.markdown(
    """
本应用把**自主决策闭环**的评分卡建模 Agent 搬到界面上：所有指标、护栏、cut-off、
报告数值都由确定性 Python 工具计算，LLM 只负责**决策**（诊断重规划）与**文字解释**。

### 页面导航

| 页面 | 用途 |
|---|---|
| **🚂 模型训练与复核** | 上传/选择训练集 → 一键跑 Agent 闭环 → 查看守门结论、自愈履历、Critic 复核、cut-off 分档与 13 章开发报告 |
| **⚡ 实时评分** | 加载已训练评分卡，对单笔或批量样本打分并生成解释 |

### 两个页面都用到的 Agent 能力

| 模块 | 在界面上的位置 |
|---|---|
| 数据质量守门 | 训练页「概览」区的守门面板；不通过直接终止 |
| 自然语言需求 | 训练页顶部输入框，中文描述自动转成受校验的参数 |
| 诊断-重规划 / 分级自愈 | 训练页「自愈履历」页签 + 执行时间线上的 🔁 回路标记 |
| Critic 模型复核 | 训练页「Critic 复核」页签（VIF / 系数符号 / 样本集中度） |
| cut-off 与报告 | 训练页「cut-off 与分档」「开发报告」页签，报告可下载 |

> 提示：训练页提供两种模式 —— **Agent 闭环**（带守门、自愈、复核、报告）与
> **快速训练**（只出模型与指标，支持时序 CV，便于反复调参）。
"""
)

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
    st.sidebar.caption(
        "该模型用于：自然语言需求解析、诊断重规划决策、Critic 定性复核、报告分析段落。"
        "所有环节都有确定性降级路径，LLM 不可用时流程照常跑完。"
    )
