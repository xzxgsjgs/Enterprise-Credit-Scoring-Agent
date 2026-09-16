"""app.pages.1_模型训练 / 上传数据、训练模型、查看结果。

UX 流程：
1. 顶部「训练集」选择器：默认数据集（项目内置 CSMAR 全量面板）或用户上传过的历史数据集
2. 上传新训练集 → 自动保存到 DATASETS_DIR → 出现在历史下拉里
3. 「数据」配置区只需设置目标列、测试集比例、时序切分开关
4. 主区域预览 + 训练按钮 + 进度条
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.core.dataset_store import (
    delete_dataset,
    list_uploaded_datasets,
    save_uploaded_dataset,
)
from app.core.paths import DEFAULT_DATASET_PATH, LATEST_SCORECARD_PATH
from app.core.training import run_training_pipeline
from app.ui import render_guardrail_panel, render_metrics_cards, render_score_distribution

st.set_page_config(page_title="模型训练", page_icon="🚂", layout="wide")
st.title("🚂 模型训练")

# ---- 常量 ----
DEFAULT_DATASET_LABEL = "⭐ 默认数据集 (CSMAR 全量面板)"

# ---- 缓存数据集列表（rerun 之间保持）----
if "_uploaded_datasets" not in st.session_state:
    st.session_state._uploaded_datasets = list_uploaded_datasets()


def _refresh_datasets() -> None:
    st.session_state._uploaded_datasets = list_uploaded_datasets()


# ---- 侧边栏：训练集选择（顶部 expanded=True）----
st.sidebar.header("训练配置")

with st.sidebar.expander("训练集", expanded=True):
    options = [DEFAULT_DATASET_LABEL] + [
        f"📁 {d['label']} · {d['rows']} 行" for d in st.session_state._uploaded_datasets
    ]

    if "selected_dataset" not in st.session_state:
        st.session_state.selected_dataset = DEFAULT_DATASET_LABEL
    if st.session_state.selected_dataset not in options:
        st.session_state.selected_dataset = DEFAULT_DATASET_LABEL

    selected_label = st.selectbox("当前训练集", options, key="selected_dataset")

    # 元信息展示
    if selected_label == DEFAULT_DATASET_LABEL:
        st.caption("📍 路径: data store/csmar_enterprise_panel.csv")
        st.caption("📊 42,998 行 × 53 列")
        st.caption("📅 年份范围: 2015 – 2024")
        st.caption("🎯 目标列: is_default_next_year（违约率 ~2.9%）")
        st.caption("🏷️ 来源: CSMAR 企业财务面板（公开学术数据集）")
    else:
        for d in st.session_state._uploaded_datasets:
            if f"📁 {d['label']} · {d['rows']} 行" == selected_label:
                st.caption(f"📍 路径: {d['path']}")
                st.caption(f"📊 {d['rows']} 行 × {d['cols']} 列")
                st.caption(f"⏰ 上传于 {d['uploaded_at']}")
                st.caption(f"💾 大小 {d['size_bytes'] / 1024:.1f} KB")
                break

    # 上传新数据集
    st.divider()
    st.caption("**上传新训练集**")
    with st.form("upload_dataset_form", clear_on_submit=True):
        new_file = st.file_uploader(
            "选择 CSV / Excel",
            type=["csv", "xlsx", "xls"],
            key="upload_ds_file",
        )
        new_label = st.text_input(
            "训练集别名（必填）",
            placeholder="例：my_test_v1",
            key="upload_ds_label",
        )
        submit_upload = st.form_submit_button("保存到历史", use_container_width=True)
        if submit_upload:
            if new_file is None:
                st.error("请先选择文件")
            elif not new_label.strip():
                st.error("请填写别名")
            else:
                meta = save_uploaded_dataset(
                    new_file,
                    new_label.strip(),
                    Path(new_file.name).suffix.lower(),
                )
                _refresh_datasets()
                # 自动切到新上传的训练集
                st.session_state.selected_dataset = (
                    f"📁 {meta['label']} · {meta['rows']} 行"
                )
                st.success(f"已保存: {meta['label']} ({meta['rows']} 行)")
                time.sleep(0.3)
                st.rerun()

    # 历史训练集列表（每条带删除按钮）
    if st.session_state._uploaded_datasets:
        st.divider()
        st.caption("**已上传的训练集**")
        for d in st.session_state._uploaded_datasets:
            c1, c2 = st.columns([4, 1])
            c1.caption(f"📁 {d['label']}\n{d['rows']} 行 · {d['uploaded_at']}")
            if c2.button("🗑", key=f"del_{d['id']}", help=f"删除 {d['label']}"):
                delete_dataset(d['id'])
                _refresh_datasets()
                # 如果删的是当前选中的，回到默认
                if (
                    st.session_state.selected_dataset
                    == f"📁 {d['label']} · {d['rows']} 行"
                ):
                    st.session_state.selected_dataset = DEFAULT_DATASET_LABEL
                st.rerun()


# ---- 解析当前训练集路径 ----
current_dataset_path: Path | None = None
current_dataset_meta: dict[str, Any] | None = None
if selected_label == DEFAULT_DATASET_LABEL:
    current_dataset_path = DEFAULT_DATASET_PATH
    current_dataset_meta = {
        "label": "默认数据集",
        "path": str(DEFAULT_DATASET_PATH),
        "source": "default",
    }
else:
    for d in st.session_state._uploaded_datasets:
        if f"📁 {d['label']} · {d['rows']} 行" == selected_label:
            current_dataset_path = Path(d['path'])
            current_dataset_meta = d
            break

if current_dataset_path is None or not current_dataset_path.exists():
    st.error(f"训练集文件不存在: {current_dataset_path}")
    st.stop()


# ---- 「数据」配置区（去掉了 file_uploader，源数据由顶部训练集选择器控制）----
with st.sidebar.expander("数据", expanded=True):
    st.caption(f"已选: **{current_dataset_meta['label']}**")
    target_col = st.text_input("目标列名", value="is_default_next_year")
    test_size = st.slider(
        "测试集比例", 0.1, 0.5, 0.3, 0.05,
        help="默认 0.3，划分训练集与测试集的比例",
    )
    time_split = st.toggle(
        "启用时序切分 (Expanding Window CV)", value=False,
        help="需数据包含年份列",
    )
    year_col = st.text_input("年份列名", value="year", disabled=not time_split)
    min_train_years = st.number_input(
        "最小训练年数", 1, 10, 2, 1, disabled=not time_split,
    )

with st.sidebar.expander("特征与分箱", expanded=False):
    iv_threshold = st.number_input(
        "IV 阈值", 0.0, 0.5, 0.02, 0.01,
        help="低于该 IV 值的变量将被剔除",
    )
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
    "time_split": time_split,
    "year_col": year_col,
    "min_train_years": int(min_train_years),
    "use_lgbm": use_lgbm,
    "lgbm_num_leaves": int(lgbm_num_leaves),
    "lgbm_lr": float(lgbm_lr),
}


# ---- 主区域：预览数据 ----
suffix = current_dataset_path.suffix.lower()
preview_df = (
    pd.read_csv(current_dataset_path, encoding="utf-8-sig")
    if suffix == ".csv"
    else pd.read_excel(current_dataset_path)
)
st.write(f"📊 数据预览（{len(preview_df)} 行 × {preview_df.shape[1]} 列）", preview_df.head(10))

if target_col not in preview_df.columns:
    st.error(f"目标列 '{target_col}' 不在数据中。可用列: {list(preview_df.columns)}")
    st.stop()

# ---- 预计耗时估算 ----
n_rows_estimate = max(len(preview_df), 1)
is_small = n_rows_estimate <= 1000
cv_on = bool(config.get("time_split"))
lgbm_on = bool(config.get("use_lgbm"))

if cv_on:
    eta_text = "1-3 分钟（小样本）" if is_small else "30-60 分钟（大样本）"
elif lgbm_on:
    eta_text = "10-30 秒（小样本）" if is_small else "2-5 分钟（大样本）"
else:
    eta_text = "10-30 秒（小样本）" if is_small else "2-5 分钟（大样本）"

st.caption(
    f"数据集 {current_dataset_meta['label']} · {n_rows_estimate} 行 · "
    f"{'开启' if cv_on else '关闭'}时序切分 · {'开启' if lgbm_on else '关闭'}LGBM 对照  |  预计耗时 {eta_text}"
)


# ---- 阶段映射：stage -> (pct, 中文阶段) ----
STAGE_PROGRESS = {
    "load_data": (5, "加载数据"),
    "eda": (10, "数据探索"),
    "cv_start": (12, "时序切分启动"),
    "cv_fold": (35, "时序切分"),
    "cv_end": (55, "时序切分收尾"),
    "split": (60, "数据划分"),
    "preprocess": (62, "预处理"),
    "var_filter": (68, "变量筛选 (IV)"),
    "woebin": (78, "WOE 分箱"),
    "train_lr": (88, "逻辑回归训练"),
    "evaluate": (92, "评估指标"),
    "scorecard": (96, "生成评分卡"),
    "lgbm_start": (96, "LGBM 对照模型"),
    "lgbm_end": (99, "LGBM 完成"),
    "done": (100, "完成"),
}


# ---- 训练按钮 ----
if st.button("开始训练", type="primary", use_container_width=True):
    progress = st.progress(0, text="准备开始 ...")
    start_ts = time.time()

    def _on_progress(stage: str, detail: str = "") -> None:
        pct, stage_label = STAGE_PROGRESS.get(stage, (0, stage))
        elapsed = time.time() - start_ts
        msg = f"{stage_label} · 已耗时 {elapsed:.1f}s"
        if detail:
            msg = f"{msg} · {detail}"
        progress.progress(pct, text=msg)

    try:
        result = run_training_pipeline(
            str(current_dataset_path),
            target=target_col,
            config=config,
            progress_callback=_on_progress,
        )
        st.session_state["training_result"] = result
        LATEST_SCORECARD_PATH.write_bytes(pickle.dumps(result.scorecard))
        progress.progress(100, text=f"完成 · 总耗时 {time.time() - start_ts:.1f}s")
        st.success(f"训练完成 · 评分卡已自动保存到 {LATEST_SCORECARD_PATH}")
    except Exception as e:
        progress.empty()
        st.error(f"训练失败: {e}")
        st.stop()


# ---- 结果展示 ----
if "training_result" in st.session_state:
    result = st.session_state["training_result"]

    st.subheader("模型指标")
    render_metrics_cards(result.metrics)

    render_guardrail_panel(result.guardrail, result.metrics)

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
    if hasattr(result, "feature_importance") and result.feature_importance is not None:
        fi = result.feature_importance.rename(columns={"feature": "变量", "coef": "系数", "abs_coef": "abs_系数", "rank": "排名"})
        st.dataframe(fi, use_container_width=True)
        importance_df = fi
    else:
        feature_names = list(result.train_woe.columns)
        importances = result.model.coef_[0]
        importance_df = pd.DataFrame(
            {"变量": feature_names, "系数": importances, "abs_系数": abs(importances)}
        ).sort_values("abs_系数", ascending=False)
        st.dataframe(importance_df, use_container_width=True)

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