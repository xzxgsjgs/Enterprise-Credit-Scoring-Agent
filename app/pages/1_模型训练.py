"""app.pages.1_模型训练 —— 模型训练与 Agent 复核工作台。

页面设计（V2 重构）：
    侧边栏  ① 训练集  ② 执行模式  ③ 数据  ④ 特征与分箱  ⑤ 评分卡
            ⑥ 护栏阈值  ⑦ 数据守门阈值  ⑧ Critic 阈值  ⑨ 对照模型
    主区    A 自然语言需求（模块 4）  B 数据概览  C 执行
            D 执行时间线（18 节点实时）  E 结果分区（Tabs）  F 人工复核

两种执行模式：
- **🤖 Agent 闭环**（默认）：跑 18 节点 LangGraph 图，依次经过
  数据守门 → 建模 → Critic 复核 → 护栏 → （必要时）诊断重规划回跳 → 报告与 cut-off。
  产出《模型开发报告》13 章 + cut-off 建议 + 分档。耗时更长但带自愈与复核。
- **⚡ 快速训练**：跑确定性 pipeline，只出模型与指标，支持 Expanding Window CV 与 LGBM。
  适合反复调参试跑。
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.core.agent_runner import (
    NODE_ORDER,
    build_agent_config,
    parse_requirement,
    run_agent,
    save_scorecard,
    summarize_state,
)
from app.core.dataset_store import (
    delete_dataset,
    list_uploaded_datasets,
    save_uploaded_dataset,
)
from app.core.paths import DEFAULT_DATASET_PATH, LATEST_SCORECARD_PATH
from app.core.training import run_training_pipeline
from app.ui import (
    apply_event,
    init_timeline,
    render_critic_panel,
    render_cutoff_panel,
    render_data_gate_panel,
    render_guardrail_panel,
    render_hitl_panel,
    render_metrics_cards,
    render_report_panel,
    render_retry_panel,
    render_run_log,
    render_run_summary,
    render_score_distribution,
    render_timeline,
)

st.set_page_config(page_title="模型训练与复核", page_icon="🚂", layout="wide")

DEFAULT_DATASET_LABEL = "⭐ 默认数据集 (CSMAR 全量面板)"
MODE_AGENT = "🤖 Agent 闭环"
MODE_QUICK = "⚡ 快速训练"


# ============================================================================
# 会话状态
# ============================================================================
def _ss(key: str, default: Any) -> Any:
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


_ss("_uploaded_datasets", list_uploaded_datasets())
_ss("agent_result", None)          # dict: summary/state/thread_id/status/elapsed/report_name
_ss("quick_result", None)
_ss("nl_patch", {})
_ss("nl_warnings", [])


def _refresh_datasets() -> None:
    st.session_state._uploaded_datasets = list_uploaded_datasets()


# ============================================================================
# 侧边栏
# ============================================================================
def _render_dataset_picker() -> tuple[str, Path | None, dict[str, Any] | None]:
    st.sidebar.header("训练配置")

    with st.sidebar.expander("① 训练集", expanded=True):
        options = [DEFAULT_DATASET_LABEL] + [
            f"📁 {d['label']} · {d['rows']} 行" for d in st.session_state._uploaded_datasets
        ]
        if "selected_dataset" not in st.session_state:
            st.session_state.selected_dataset = DEFAULT_DATASET_LABEL
        if st.session_state.selected_dataset not in options:
            st.session_state.selected_dataset = DEFAULT_DATASET_LABEL

        selected = st.selectbox("当前训练集", options, key="selected_dataset")

        if selected == DEFAULT_DATASET_LABEL:
            st.caption("📍 data store/csmar_enterprise_panel.csv")
            st.caption("📊 42,998 行 × 53 列 · 2015–2024")
            st.caption("🎯 is_default_next_year（违约率 ~2.9%）")
        else:
            for d in st.session_state._uploaded_datasets:
                if f"📁 {d['label']} · {d['rows']} 行" == selected:
                    st.caption(f"📍 {d['path']}")
                    st.caption(f"📊 {d['rows']} 行 × {d['cols']} 列")
                    st.caption(f"⏰ {d['uploaded_at']} · 💾 {d['size_bytes'] / 1024:.1f} KB")
                    break

        st.divider()
        st.caption("**上传新训练集**")
        with st.form("upload_dataset_form", clear_on_submit=True):
            new_file = st.file_uploader("CSV / Excel", type=["csv", "xlsx", "xls"], key="upload_ds_file")
            new_label = st.text_input("训练集别名（必填）", placeholder="例：my_test_v1", key="upload_ds_label")
            if st.form_submit_button("保存到历史", width="stretch"):
                if new_file is None:
                    st.error("请先选择文件")
                elif not new_label.strip():
                    st.error("请填写别名")
                else:
                    meta = save_uploaded_dataset(
                        new_file, new_label.strip(), Path(new_file.name).suffix.lower()
                    )
                    _refresh_datasets()
                    st.session_state.selected_dataset = f"📁 {meta['label']} · {meta['rows']} 行"
                    st.success(f"已保存: {meta['label']}（{meta['rows']} 行）")
                    time.sleep(0.3)
                    st.rerun()

        if st.session_state._uploaded_datasets:
            st.divider()
            st.caption("**已上传的训练集**")
            for d in st.session_state._uploaded_datasets:
                c1, c2 = st.columns([4, 1])
                c1.caption(f"📁 {d['label']}\n{d['rows']} 行 · {d['uploaded_at']}")
                if c2.button("🗑", key=f"del_{d['id']}", help=f"删除 {d['label']}"):
                    delete_dataset(d["id"])
                    _refresh_datasets()
                    if st.session_state.selected_dataset == f"📁 {d['label']} · {d['rows']} 行":
                        st.session_state.selected_dataset = DEFAULT_DATASET_LABEL
                    st.rerun()

    # 解析路径
    if selected == DEFAULT_DATASET_LABEL:
        return selected, DEFAULT_DATASET_PATH, {"label": "默认数据集", "source": "default"}
    for d in st.session_state._uploaded_datasets:
        if f"📁 {d['label']} · {d['rows']} 行" == selected:
            return selected, Path(d["path"]), d
    return selected, None, None


def _render_mode_and_agent_opts() -> tuple[
    str, dict[str, Any], dict[str, Any], dict[str, Any], str
]:
    """返回 (mode, 基础 config, agent 专属 config, 展示用阈值, 目标列名)。"""
    with st.sidebar.expander("② 执行模式", expanded=True):
        mode = st.radio(
            "执行模式",
            [MODE_AGENT, MODE_QUICK],
            captions=[
                "18 节点闭环：守门 → 建模 → Critic → 护栏 → 自愈 → 报告",
                "确定性 pipeline：只出模型指标，支持时序 CV",
            ],
            label_visibility="collapsed",
        )

        agent_opts: dict[str, Any] = {}
        if mode == MODE_AGENT:
            st.caption("**Agent 自主决策**")
            max_retry = st.slider(
                "最大自愈轮数 max_retry", 0, 5, 3,
                help="护栏未通过时最多自动诊断+回跳重训几次；0 = 关闭自愈直接转人工",
            )
            allow_diag = st.toggle(
                "LLM 参与诊断决策", value=True,
                help="关闭则完全用规则阶梯兜底（确定性，无 LLM 依赖）",
            )
            allow_critic = st.toggle("LLM 参与 Critic 复核", value=True,
                                     help="关闭则只用三项确定性审计结论")
            allow_report = st.toggle("LLM 撰写报告分析段", value=True,
                                     help="关闭则报告的分析段落用占位文本（离线可用）")
        else:
            max_retry, allow_diag, allow_critic, allow_report = 0, False, False, False

    with st.sidebar.expander("③ 数据", expanded=(mode == MODE_QUICK)):
        target_col = st.text_input("目标列名", value="is_default_next_year")
        test_size = st.slider("测试集比例", 0.1, 0.5, 0.3, 0.05)
        if mode == MODE_QUICK:
            time_split = st.toggle("启用时序切分 (Expanding Window CV)", value=False,
                                   help="需数据含年份列")
            year_col = st.text_input("年份列名", value="year", disabled=not time_split)
            min_train_years = st.number_input("最小训练年数", 1, 10, 2, 1, disabled=not time_split)
        else:
            time_split, year_col, min_train_years = False, "year", 2
            st.info(
                "Agent 闭环当前使用随机切分（时序 CV 只在快速训练模式提供）。"
                "如需时间外推验证，请切换到「快速训练」。",
                icon="ℹ️",
            )

    with st.sidebar.expander("④ 特征与分箱", expanded=False):
        iv_threshold = st.number_input("IV 阈值", 0.0, 0.5, 0.02, 0.01,
                                       help="低于该 IV 值的变量将被剔除")
        max_bins = st.slider("最大分箱数", 3, 15, 8, 1)
        regularization = st.number_input("正则化强度 (1/C)", 0.001, 1.0, 0.01, 0.001,
                                         format="%.3f")

    with st.sidebar.expander("⑤ 评分卡", expanded=False):
        base_score = st.number_input("基准分", 300, 1000, 600, 10)
        pdo = st.number_input("PDO", 5, 50, 20, 1)

    with st.sidebar.expander("⑥ 护栏阈值", expanded=False):
        min_ks = st.number_input("最小 KS", 0.0, 1.0, 0.3, 0.05)
        min_auc = st.number_input("最小 AUC", 0.5, 1.0, 0.7, 0.05)
        max_psi = st.number_input("最大 PSI", 0.0, 1.0, 0.1, 0.05)

    gate_opts: dict[str, Any] = {}
    critic_opts: dict[str, Any] = {}
    if mode == MODE_AGENT:
        with st.sidebar.expander("⑦ 数据守门阈值", expanded=False):
            st.caption("模块 6：不满足即短路终止，不进入建模")
            min_rows = st.number_input("最小样本量", 10, 100000, 1000, 10)
            min_default_rate = st.number_input("违约率下限", 0.0, 0.5, 0.005, 0.001, format="%.3f")
            max_default_rate = st.number_input("违约率上限", 0.05, 1.0, 0.5, 0.05)
            max_missing_ratio = st.slider("单列缺失率上限", 0.0, 1.0, 0.6, 0.05)
            max_identical_ratio = st.slider("单一值占比上限", 0.5, 1.0, 0.98, 0.01)
        gate_opts = {
            "min_rows": int(min_rows),
            "min_default_rate": float(min_default_rate),
            "max_default_rate": float(max_default_rate),
            "max_missing_ratio": float(max_missing_ratio),
            "max_identical_ratio": float(max_identical_ratio),
        }

        with st.sidebar.expander("⑧ Critic 阈值", expanded=False):
            st.caption("模块 3：三项确定性审计的判定阈值")
            vif_threshold = st.number_input("VIF 高危阈值", 2.0, 50.0, 10.0, 0.5,
                                            help="VIF > 该值判高危；> 5 判中等")
            concentration_threshold = st.slider("单一取值占比上限", 0.3, 1.0, 0.5, 0.05)
        critic_opts = {
            "vif_threshold": float(vif_threshold),
            "concentration_threshold": float(concentration_threshold),
        }

        with st.sidebar.expander("⑨ cut-off 与报告", expanded=False):
            cutoff_label = st.selectbox(
                "cut-off 择优口径", ["ks（最大化 KS）", "approve_rate（目标通过率）", "bad_rate（目标坏率）"],
            )
            cutoff_method = cutoff_label.split("（")[0]
            cutoff_target = None
            if cutoff_method == "approve_rate":
                cutoff_target = st.slider("目标通过率", 0.1, 0.99, 0.7, 0.01)
            elif cutoff_method == "bad_rate":
                cutoff_target = st.slider("目标通过后坏率", 0.001, 0.2, 0.02, 0.001, format="%.3f")
            generate_report = st.toggle("生成并落盘开发报告", value=True)
            report_dir = st.text_input("报告目录", value="reports")
    else:
        cutoff_method, cutoff_target, generate_report, report_dir = "ks", None, False, "reports"
        vif_threshold, concentration_threshold = 10.0, 0.5

    with st.sidebar.expander("⑩ 对照模型（LGBM）", expanded=False):
        st.caption("仅作对照，不替换评分卡主模型")
        use_lgbm = st.toggle("启用 LGBM 对照模型", value=False)
        lgbm_num_leaves = st.slider("num_leaves", 8, 128, 31, 1)
        lgbm_lr = st.number_input("学习率", 0.01, 0.3, 0.05, 0.01, format="%.2f")

    base_cfg: dict[str, Any] = {
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
    agent_cfg: dict[str, Any] = {
        "max_retry": int(max_retry),
        "allow_llm_diagnosis": bool(allow_diag),
        "allow_llm_critic": bool(allow_critic),
        "allow_llm_report": bool(allow_report),
        "generate_report": bool(generate_report),
        "report_dir": report_dir,
        "cutoff_method": cutoff_method,
        "cutoff_target": cutoff_target,
        **gate_opts,
        **critic_opts,
    }
    display = {
        "vif_threshold": vif_threshold,
        "concentration_threshold": concentration_threshold,
    }
    return mode, base_cfg, agent_cfg, display, target_col


# ============================================================================
# 主区：自然语言需求（模块 4）
# ============================================================================
def _render_nl_box() -> dict[str, Any]:
    with st.container(border=True):
        st.markdown("##### 💬 自然语言建模需求（模块 4）")
        st.caption(
            "直接用中文描述要求，Agent 把它解析成受白名单约束的参数（LLM 解析 + 规则兜底 + 范围校验）。"
            "例：`用 2018 年以后数据、最多分 6 箱、排除房地产行业、AUC 目标 0.95`"
        )
        c1, c2 = st.columns([4, 1])
        text = c1.text_area(
            "需求描述", placeholder="用 2018 年以后数据、最多分 6 箱、排除房地产行业",
            height=80, label_visibility="collapsed", key="nl_text",
        )
        with c2:
            parse_clicked = st.button("解析需求", width="stretch")
            clear_clicked = st.button("清空", width="stretch")

        if clear_clicked:
            st.session_state.nl_patch = {}
            st.session_state.nl_warnings = []
            st.rerun()

        if parse_clicked:
            if not text.strip():
                st.warning("请输入需求描述")
            else:
                with st.spinner("解析中（LLM + 规则兜底）..."):
                    patch, warns = parse_requirement(text)
                st.session_state.nl_patch = patch
                st.session_state.nl_warnings = warns

        patch = st.session_state.nl_patch
        warns = st.session_state.nl_warnings
        if patch:
            st.success(f"已解析出 {len(patch)} 项参数，将覆盖侧边栏同项设置")
            st.dataframe(
                pd.DataFrame([{"参数": k, "值": str(v)} for k, v in patch.items()]),
                width="stretch", hide_index=True,
            )
        for w in warns:
            st.warning(w)
        if patch and not warns:
            st.caption("✅ 全部参数通过白名单与范围校验（越界值会被自动夹取）")
    return patch


# ============================================================================
# 执行：Agent 闭环
# ============================================================================
def _run_agent_mode(csv_path: Path, target: str, cfg: dict[str, Any],
                    resume_payload: dict[str, Any] | None = None,
                    thread_id: str | None = None) -> None:
    timeline = init_timeline()
    if resume_payload is None:
        st.markdown("**Agent 执行时间线**")
    else:
        st.markdown("**Agent 执行时间线（人工复核后继续）**")
    slot = st.empty()

    def _paint(current: str | None = None) -> None:
        with slot.container():
            render_timeline(timeline, current=current)

    _paint()

    def _on_event(ev: Any) -> None:
        apply_event(timeline, ev)
        # 猜测下一个待执行节点，用于显示「运行中」（回跳决策不猜）
        nxt = None
        if ev.node in NODE_ORDER and ev.status != "retry":
            i = NODE_ORDER.index(ev.node)
            for cand in NODE_ORDER[i + 1:]:
                if timeline.get(cand, {}).get("status") == "pending":
                    nxt = cand
                    break
        _paint(current=nxt)

    with st.spinner("Agent 运行中，请稍候……"):
        result = run_agent(
            csv_path=str(csv_path),
            target=target,
            config=cfg,
            thread_id=thread_id,
            on_event=_on_event,
            resume_payload=resume_payload,
        )
    _paint()

    summary = summarize_state(result.state)
    # 评分卡落盘，供「实时评分」页使用
    if result.state.get("scorecard"):
        summary["scorecard_saved"] = save_scorecard(result.state, LATEST_SCORECARD_PATH)

    st.session_state.agent_result = {
        "summary": summary,
        "state": result.state,
        "thread_id": result.thread_id,
        "status": result.status,
        "elapsed": result.elapsed,
        "error": result.error,
        "report_name": f"model_report_{result.thread_id}.md",
    }


# ============================================================================
# 结果：Agent 闭环
# ============================================================================
def _render_agent_results(payload: dict[str, Any], display_cfg: dict[str, Any]) -> None:
    summary = payload["summary"]
    state = payload["state"]
    status = payload["status"]

    st.divider()

    reached_modeling = summary.get("auc") is not None or summary.get("guardrail_passed") is not None

    # ---- 状态横幅 ----
    if status == "interrupted":
        st.warning(f"⏸ 流程在「人工复核」处暂停 · 已耗时 {payload['elapsed']:.1f}s")
    elif status == "failed":
        st.error(f"❌ 流程终止：{payload.get('error') or '不可恢复节点失败'}")
    elif summary.get("gate_passed") is False:
        st.error(
            f"⛔ 数据守门未通过，流程已短路终止（未进入建模）· 耗时 {payload['elapsed']:.1f}s"
        )
    elif not reached_modeling:
        st.warning(f"⚠️ 流程提前结束，未产出模型 · 耗时 {payload['elapsed']:.1f}s")
    else:
        st.success(f"✅ 流程完成 · 总耗时 {payload['elapsed']:.1f}s · thread `{payload['thread_id']}`")

    render_run_summary(summary)

    # ---- 提前终止的两种终局：守门拦截 / 关键节点失败 ----
    if summary.get("critical_error"):
        st.error("不可恢复节点失败，流程已短路终止。请检查数据格式与目标列后重试。")
        render_run_log(summary)
        return
    if summary.get("gate_passed") is False:
        render_data_gate_panel(state.get("data_gate"))
        st.markdown(
            "**下一步建议**：调整数据（补足样本量 / 处理异常违约率 / 剔除高缺失列）"
            "或放宽侧边栏「⑦ 数据守门阈值」，然后重新启动。"
        )
        render_run_log(summary)
        return

    # ---- 结果分区 ----
    tabs = st.tabs([
        "📋 概览",
        "🔁 自愈履历",
        "🧪 Critic 复核",
        "🎯 cut-off 与分档",
        "📄 开发报告",
        "📈 模型明细",
        "📦 导出",
    ])

    with tabs[0]:
        render_data_gate_panel(state.get("data_gate"))
        st.divider()
        if state.get("guardrail") is not None:
            render_guardrail_panel(state["guardrail"], state.get("metrics") or {})
        else:
            st.info("未产出护栏结论。")
        st.divider()
        render_run_log(summary)

    with tabs[1]:
        render_retry_panel(summary, state.get("diagnosis"), reached_modeling=reached_modeling)

    with tabs[2]:
        render_critic_panel(
            state.get("critic_report"),
            vif_threshold=display_cfg.get("vif_threshold", 10.0),
            concentration_threshold=display_cfg.get("concentration_threshold", 0.5),
        )

    with tabs[3]:
        render_cutoff_panel(
            state.get("cutoff_info"),
            grades=state.get("score_grades"),
            detail=state.get("scored_detail"),
        )

    with tabs[4]:
        render_report_panel(
            state.get("report_md") or "",
            report_path=state.get("report_path"),
            report_name=payload.get("report_name", "model_report.md"),
        )

    with tabs[5]:
        _render_model_details(state)

    with tabs[6]:
        _render_exports(state)


def _render_model_details(state: dict[str, Any]) -> None:
    metrics = state.get("metrics") or {}
    if metrics:
        st.markdown("##### 模型指标")
        render_metrics_cards(metrics)

    scored = state.get("scored_test")
    if scored is not None:
        st.markdown("##### 评分分布（测试集）")
        render_score_distribution(scored)

    fi = state.get("feature_importance")
    if fi is not None and hasattr(fi, "columns"):
        st.markdown("##### 变量重要性（LR+WOE 的 |coef| 排序）")
        st.caption("由 `tools.coefficient_importance` 确定性计算，非 LLM 产出")
        st.dataframe(
            fi.rename(columns={"feature": "变量", "coef": "系数",
                               "abs_coef": "abs_系数", "rank": "排名"}),
            width="stretch", hide_index=True,
        )

    lgbm = state.get("lgbm_metrics") or {}
    if lgbm and "error" not in lgbm:
        st.markdown("##### LGBM 对照模型")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("LGBM AUC", f"{lgbm.get('auc', 0):.3f}")
        c2.metric("LGBM KS", f"{lgbm.get('ks', 0):.3f}")
        c3.metric("LGBM Gini", f"{lgbm.get('gini', 0):.3f}")
        c4.metric("vs LR ΔAUC", f"{lgbm.get('auc', 0) - (metrics.get('test') or {}).get('auc', 0):+.3f}")
        st.caption("LGBM 仅作对照；评分卡仍以 LR+WOE 为准（可解释性 + 业务可审计）")
    elif lgbm and "error" in lgbm:
        st.warning(f"LGBM 训练失败: {lgbm['error']}")


def _render_exports(state: dict[str, Any]) -> None:
    card = state.get("scorecard")
    if not card:
        st.info("无评分卡可导出。")
        return

    fi = state.get("feature_importance")
    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "下载评分卡 (pickle)", data=pickle.dumps(card),
        file_name="scorecard.pkl", mime="application/octet-stream",
        key="dl_card_agent",
    )
    card_json = {k: v.to_dict(orient="records") for k, v in card.items()}
    c2.download_button(
        "下载评分卡 (JSON)",
        data=json.dumps(card_json, ensure_ascii=False, indent=2),
        file_name="scorecard.json", mime="application/json",
        key="dl_card_json_agent",
    )
    if fi is not None and hasattr(fi, "to_csv"):
        c3.download_button(
            "下载变量重要性 (CSV)",
            data=fi.to_csv(index=False).encode("utf-8"),
            file_name="variable_importance.csv", mime="text/csv",
            key="dl_importance_agent",
        )
    st.caption(f"评分卡已同步保存到 `{LATEST_SCORECARD_PATH}`，可直接到「实时评分」页使用。")


# ============================================================================
# 执行与结果：快速训练（确定性 pipeline）
# ============================================================================
QUICK_STAGE_PROGRESS: dict[str, tuple[int, str]] = {
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


def _run_quick_mode(csv_path: Path, target: str, base_cfg: dict[str, Any]) -> None:
    progress = st.progress(0, text="准备开始 ...")
    start_ts = time.time()

    def _on_progress(stage: str, detail: str = "") -> None:
        pct, label = QUICK_STAGE_PROGRESS.get(stage, (0, stage))
        msg = f"{label} · 已耗时 {time.time() - start_ts:.1f}s"
        if detail:
            msg = f"{msg} · {detail}"
        progress.progress(pct, text=msg)

    try:
        result = run_training_pipeline(
            str(csv_path), target=target, config=base_cfg, progress_callback=_on_progress,
        )
    except Exception as e:  # noqa: BLE001
        progress.empty()
        st.error(f"训练失败: {e}")
        return

    st.session_state.quick_result = result
    LATEST_SCORECARD_PATH.write_bytes(pickle.dumps(result.scorecard))
    progress.progress(100, text=f"完成 · 总耗时 {time.time() - start_ts:.1f}s")
    st.success(f"训练完成 · 评分卡已保存到 {LATEST_SCORECARD_PATH}")


def _render_quick_results() -> None:
    result = st.session_state.quick_result
    st.divider()
    st.info(
        "快速训练模式只产出模型与指标。若需要数据守门、自愈重规划、Critic 复核与开发报告，"
        "请切换到「🤖 Agent 闭环」。",
        icon="⚡",
    )

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
            st.dataframe(cv_per_fold, width="stretch")

    st.subheader("评分分布（测试集）")
    render_score_distribution(result.scored_test)

    st.subheader("变量重要性")
    fi = getattr(result, "feature_importance", None)
    if fi is not None:
        st.dataframe(
            fi.rename(columns={"feature": "变量", "coef": "系数",
                               "abs_coef": "abs_系数", "rank": "排名"}),
            width="stretch", hide_index=True,
        )
    else:
        feature_names = list(result.train_woe.columns)
        imp = pd.DataFrame({
            "变量": feature_names, "系数": result.model.coef_[0],
            "abs_系数": abs(result.model.coef_[0]),
        }).sort_values("abs_系数", ascending=False)
        st.dataframe(imp, width="stretch", hide_index=True)

    lgbm_metrics = getattr(result, "lgbm_metrics", None) or {}
    if lgbm_metrics and "error" not in lgbm_metrics:
        st.subheader("LGBM 对照模型")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("LGBM AUC", f"{lgbm_metrics.get('auc', 0):.3f}")
        c2.metric("LGBM KS", f"{lgbm_metrics.get('ks', 0):.3f}")
        c3.metric("LGBM Gini", f"{lgbm_metrics.get('gini', 0):.3f}")
        c4.metric("vs LR ΔAUC", f"{lgbm_metrics.get('auc', 0) - result.metrics.get('auc', 0):+.3f}")
    elif lgbm_metrics and "error" in lgbm_metrics:
        st.warning(f"LGBM 训练失败: {lgbm_metrics['error']}")

    st.divider()
    st.subheader("导出")
    col1, col2, col3 = st.columns(3)
    col1.download_button(
        "下载评分卡 (pickle)", data=pickle.dumps(result.scorecard),
        file_name="scorecard.pkl", mime="application/octet-stream", key="dl_card_quick",
    )
    col2.download_button(
        "下载评分卡 (JSON)",
        data=json.dumps({k: v.to_dict(orient="records") for k, v in result.scorecard.items()},
                        ensure_ascii=False, indent=2),
        file_name="scorecard.json", mime="application/json", key="dl_card_json_quick",
    )
    if fi is not None:
        col3.download_button(
            "下载变量重要性 (CSV)", data=fi.to_csv(index=False).encode("utf-8"),
            file_name="variable_importance.csv", mime="text/csv", key="dl_importance_quick",
        )


# ============================================================================
# 页面主体
# ============================================================================
st.title("🚂 模型训练与复核")
st.caption(
    "Agent 闭环 = 数据守门 → 建模 → Critic 复核 → 合规护栏 → 失败自愈重规划 → 开发报告与 cut-off。"
    "所有数值由确定性 Python 工具计算，LLM 只负责决策与文字解释。"
)

selected_label, current_dataset_path, current_dataset_meta = _render_dataset_picker()
mode, base_cfg, agent_cfg, display_cfg, target_col = _render_mode_and_agent_opts()

# ---- 顶部状态条 ----
if current_dataset_path is None or not current_dataset_path.exists():
    st.error(f"训练集文件不存在: {current_dataset_path}")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.markdown("**当前训练集**")
c1.markdown(f"`{current_dataset_meta['label']}`")
c2.markdown("**执行模式**")
c2.markdown(mode)
c3.markdown("**最近一次运行**")
_ar = st.session_state.agent_result
_qr = st.session_state.quick_result
if _ar:
    _st_map = {"completed": "✅ 完成", "interrupted": "⏸ 待人工复核", "failed": "❌ 终止"}
    c3.markdown(f"{_st_map.get(_ar['status'], _ar['status'])} · Agent 闭环")
elif _qr is not None:
    c3.markdown("✅ 完成 · 快速训练")
else:
    c3.markdown("_尚未运行_")

st.divider()

# ---- A. 自然语言需求 ----
nl_patch = _render_nl_box() if mode == MODE_AGENT else {}

# ---- B. 数据概览 ----
suffix = current_dataset_path.suffix.lower()
try:
    preview_df = (
        pd.read_csv(current_dataset_path, encoding="utf-8-sig")
        if suffix == ".csv"
        else pd.read_excel(current_dataset_path)
    )
except Exception as e:  # noqa: BLE001
    st.error(f"训练集读取失败: {e}")
    st.stop()

target_col = target_col.strip()
with st.expander(
    f"📊 数据概览 · {len(preview_df)} 行 × {preview_df.shape[1]} 列", expanded=False
):
    st.dataframe(preview_df.head(20), width="stretch")

# 目标列取值校验（提前暴露低级错误，避免跑完才失败）
if target_col not in preview_df.columns:
    st.error(f"目标列 `{target_col}` 不在数据中。可用列: {list(preview_df.columns)[:40]}")
    st.stop()

_rate = float(pd.to_numeric(preview_df[target_col], errors="coerce").mean() or 0)

# ---- C. 执行 ----
n_rows = max(len(preview_df), 1)
is_small = n_rows <= 2000
if mode == MODE_AGENT:
    eta = "1-3 分钟（小样本）" if is_small else "5-20 分钟（大样本）"
else:
    eta = "10-30 秒（小样本）" if is_small else "2-5 分钟（大样本）"
    if base_cfg.get("time_split"):
        eta = "1-3 分钟（小样本）" if is_small else "30-60 分钟（大样本）"

st.caption(
    f"数据集 `{current_dataset_meta['label']}` · {n_rows} 行 · 违约率 {_rate:.2%} · "
    f"{'开启' if base_cfg.get('use_lgbm') else '关闭'} LGBM 对照  |  预计耗时 {eta}"
)

_run_col, _hint_col = st.columns([3, 2])
label = "🚀 启动 Agent 闭环" if mode == MODE_AGENT else "⚡ 开始快速训练"
clicked = _run_col.button(label, type="primary", width="stretch")
_hint_col.caption(
    "Agent 闭环会依次经过守门 → 建模 → Critic → 护栏 →（必要时）自愈 → 报告，"
    "失败节点会自动短路或回跳，过程中不要关闭页面。"
    if mode == MODE_AGENT else
    "快速训练只跑一次 pipeline，不经过守门与复核，适合反复调参试跑。"
)

if clicked:
    st.session_state.agent_result = None
    st.session_state.quick_result = None
    if mode == MODE_AGENT:
        cfg = build_agent_config(base_cfg, agent_cfg, nl_patch)
        _run_agent_mode(current_dataset_path, target_col, cfg)
    else:
        _run_quick_mode(current_dataset_path, target_col, base_cfg)

# ---- F. 人工复核（Agent 中断时） ----
_ar = st.session_state.agent_result
if _ar and _ar["status"] == "interrupted":
    st.divider()
    render_hitl_panel(_ar["summary"])
    st.markdown("**人工决策**")
    d1, d2, d3 = st.columns([1, 2, 1])
    action = d1.radio("决策", ["approve（采纳）", "reject（否决）"], key="hitl_action")
    note = d2.text_input("复核意见", value="", key="hitl_note",
                         placeholder="例：KS 偏低但业务可接受，先上线观察")
    if d3.button("提交决策并继续", width="stretch"):
        payload = {"action": action.split("（")[0], "note": note}
        cfg = build_agent_config(base_cfg, agent_cfg, nl_patch)
        _run_agent_mode(
            current_dataset_path, target_col, cfg,
            resume_payload=payload, thread_id=_ar["thread_id"],
        )
        st.rerun()

# ---- E. 结果 ----
_ar = st.session_state.agent_result
if _ar:
    _render_agent_results(_ar, display_cfg)
elif st.session_state.quick_result is not None:
    _render_quick_results()
