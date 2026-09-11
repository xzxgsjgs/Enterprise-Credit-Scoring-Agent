"""app.pages.1_模型训练 / 上传数据、训练模型、查看结果。"""
from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.core.training import run_training_pipeline
from app.ui import render_guardrail_panel, render_metrics_cards, render_score_distribution

st.set_page_config(page_title="模型训练", page_icon="🚂", layout="wide")

st.title("🚂 模型训练")

# ---- 侧边栏配置（按语义分组，避免 12 控件连成一片）----
st.sidebar.header("训练配置")

with st.sidebar.expander("数据", expanded=True):
    uploaded_file = st.file_uploader("上传训练数据 (CSV / Excel)", type=["csv", "xlsx", "xls"])
    target_col = st.text_input("目标列名", value="is_default_next_year")
    test_size = st.slider("测试集比例", 0.1, 0.5, 0.3, 0.05, help="默认 0.3，划分训练集与测试集的比例")

with st.sidebar.expander("特征与分箱", expanded=False):
    iv_threshold = st.number_input("IV 阈值", 0.0, 0.5, 0.02, 0.01, help="低于该 IV 值的变量将被剔除")
    max_bins = st.slider("最大分箱数", 3, 15, 8, 1)
    regularization = st.number_input(
        "正则化强度 (1/C)", 0.001, 1.0, 0.01, 0.001, format="%.3f",
        help="逻辑回归的正则化强度倒数，越小正则化越强",
    )

with st.sidebar.expander("评分卡", expanded=False):
    base_score = st.number_input("基准分", 300, 1000, 600, 10)
    pdo = st.number_input("PDO", 5, 50, 20, 1, help="Points to Double the Odds")

with st.sidebar.expander("护栏阈值", expanded=False):
    min_ks = st.number_input("最小 KS", 0.0, 1.0, 0.3, 0.05)
    min_auc = st.number_input("最小 AUC", 0.5, 1.0, 0.7, 0.05)
    max_psi = st.number_input("最大 PSI", 0.0, 1.0, 0.1, 0.05)

with st.sidebar.expander("对照模型（LGBM）", expanded=False):
    use_lgbm = st.toggle(
        "启用 LGBM 对照模型",
        value=False,
        help="开启后用 LightGBM 训练对照模型并计算 SHAP 特征重要性（仅参考，不参与评分卡）",
    )
    lgbm_num_leaves = st.slider("num_leaves", 8, 128, 31, 1)
    lgbm_lr = st.number_input("学习率", 0.01, 0.3, 0.05, 0.01, format="%.2f")

config: dict[str, Any] = {
    "test_size": test_size,
    "iv_threshold": iv_threshold,
    "max_bins": max_bins,
    "regularization": regularization,
    "base_score": base_score,
    "pdo": pdo,
    "base_odds": 50,
    "min_ks": min_ks,
    "min_auc": min_auc,
    "max_psi": max_psi,
    # 【v2】时序验证开关
    "time_split": time_split,
    "year_col": year_col,
    "min_train_years": int(min_train_years),
    # 【v2】LGBM 对照模型
    "use_lgbm": use_lgbm,
    "lgbm_num_leaves": int(lgbm_num_leaves),
    "lgbm_lr": float(lgbm_lr),
}

# ---- 主区域 ----
if uploaded_file is None:
    st.info("请在左侧上传训练数据后点击「开始训练」。")
    st.stop()

suffix = Path(uploaded_file.name).suffix.lower()
preview_df = pd.read_csv(uploaded_file) if suffix == ".csv" else pd.read_excel(uploaded_file)
st.write("数据预览", preview_df.head(10))

if target_col not in preview_df.columns:
    st.error(f"目标列 '{target_col}' 不在数据中。可用列: {list(preview_df.columns)}")
    st.stop()

uploaded_file.seek(0)

if st.button("开始训练", type="primary", use_container_width=True):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    progress = st.progress(0, text="加载数据...")
    try:
        result = run_training_pipeline(tmp_path, target=target_col, config=config)
        st.session_state["training_result"] = result
        progress.progress(100, text="训练完成")
        st.success("训练完成")
    except Exception as e:
        progress.empty()
        st.error(f"训练失败: {e}")
        st.stop()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

# ---- 结果展示（用 subheader 自带间距，仅保留 1 次必要 divider）----
if "training_result" in st.session_state:
    result = st.session_state["training_result"]

    st.subheader("模型指标")
    render_metrics_cards(result.metrics)

    render_guardrail_panel(result.guardrail, result.metrics)

    # 【v2】Expanding Window 时间序列 CV 结果（面板数据专用）
    cv_per_fold = getattr(result, "cv_per_fold", None)
    if cv_per_fold is not None and not cv_per_fold.empty:
        st.subheader("Expanding Window CV（时序切分）")
        cv_mean = getattr(result, "cv_mean", None) or {}
        if "error" in cv_mean:
            st.warning(f"CV 失败: {cv_mean['error']}")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("CV 均值 AUC", f"{cv_mean.get('auc', 0):.3f}")
            c2.metric("CV 均值 KS", f"{cv_mean.get('ks', 0):.3f}")
            c3.metric("CV 均值 Gini", f"{cv_mean.get('gini', 0):.3f}")
            c4.metric("Fold 数", int(cv_mean.get("n_folds", 0)))
            st.caption("每行一个 fold：训练集 = year ≤ (t-1)，测试集 = year = t")
            st.dataframe(cv_per_fold, use_container_width=True)

    st.subheader("评分分布（测试集）")
    render_score_distribution(result.scored_test)

    st.subheader("变量重要性")
    # 【v2】使用 coef 绝对值排序的简化 SHAP（结果与 train_woe 列一一对应）
    if hasattr(result, "feature_importance") and result.feature_importance is not None:
        fi = result.feature_importance.rename(columns={"feature": "变量", "coef": "系数", "abs_coef": "abs_系数", "rank": "排名"})
        st.dataframe(fi, use_container_width=True)
        importance_df = fi  # 给下方 download button 使用
    else:
        feature_names = list(result.train_woe.columns)
        importances = result.model.coef_[0]
        importance_df = pd.DataFrame(
            {"变量": feature_names, "系数": importances, "abs_系数": abs(importances)}
        ).sort_values("abs_系数", ascending=False)
        st.dataframe(importance_df, use_container_width=True)

    # 【v2】LGBM 对照模型结果
    lgbm_metrics = getattr(result, "lgbm_metrics", None) or {}
    if lgbm_metrics and "error" not in lgbm_metrics:
        st.subheader("LGBM 对照模型")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("LGBM AUC", f"{lgbm_metrics.get('auc', 0):.3f}")
        c2.metric("LGBM KS", f"{lgbm_metrics.get('ks', 0):.3f}")
        c3.metric("LGBM Gini", f"{lgbm_metrics.get('gini', 0):.3f}")
        c4.metric("vs LR ΔAUC", f"{lgbm_metrics.get('auc', 0) - result.metrics.get('auc', 0):+.3f}")
        st.caption("LGBM 仅作对照；评分卡仍以 LR+WOE 为准（可解释性 + 业务可审计）")
        if getattr(result, "lgbm_shap_importance", None) is not None:
            st.dataframe(
                result.lgbm_shap_importance.rename(columns={"feature": "变量", "mean_abs_shap": "平均|SHAP|", "rank": "排名"}),
                use_container_width=True,
            )
    elif lgbm_metrics and "error" in lgbm_metrics:
        st.warning(f"LGBM 训练失败: {lgbm_metrics['error']}")

    st.divider()
    st.subheader("导出")
    col1, col2, col3 = st.columns(3)

    card_bytes = pickle.dumps(result.scorecard)
    col1.download_button(
        label="下载评分卡 (pickle)",
        data=card_bytes,
        file_name="scorecard.pkl",
        mime="application/octet-stream",
        key="download_scorecard",
    )

    card_json = {k: v.to_dict(orient="records") for k, v in result.scorecard.items()}
    col2.download_button(
        label="下载评分卡 (JSON)",
        data=json.dumps(card_json, ensure_ascii=False, indent=2),
        file_name="scorecard.json",
        mime="application/json",
        key="download_scorecard_json",
    )

    col3.download_button(
        label="下载变量重要性 (CSV)",
        data=importance_df.to_csv(index=False).encode("utf-8"),
        file_name="variable_importance.csv",
        mime="text/csv",
        key="download_importance",
    )
