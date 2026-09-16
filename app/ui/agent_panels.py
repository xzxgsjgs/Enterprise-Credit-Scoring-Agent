"""app.ui.agent_panels —— Agent V2 六个模块的展示面板。

面板与模块的对应关系：
- render_data_gate_panel   → 模块 6 数据质量前置守门
- render_retry_panel       → 模块 1 诊断-重规划回路 / 模块 2 护栏失败分级自愈
- render_critic_panel      → 模块 3 Critic 模型复核
- render_report_panel      → 模块 5 开发报告（13 章）
- render_cutoff_panel      → 模块 5 cut-off 择优与分档
- render_hitl_panel        → 人审（护栏终局）
- render_run_summary       → 顶部三卡概览

原则：面板只读 state，不做任何数值计算（数值一律来自 tools/ 的确定性结果）。
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

_SEV_ICON = {"high": "🔴", "mid": "🟠", "low": "🟡"}
_SEV_LABEL = {"high": "高危", "mid": "中等", "low": "轻微"}


def _fmt(v: Any, digits: int = 4, pct: bool = False) -> str:
    if v is None:
        return "-"
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)
    if fv != fv:  # NaN
        return "-"
    return f"{fv * 100:.2f}%" if pct else f"{fv:.{digits}f}"


# ============================================================================
# 顶部概览
# ============================================================================
def render_run_summary(summary: dict[str, Any]) -> None:
    """三张卡：模型表现 / 护栏 / Agent 自愈与复核。"""
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**模型表现（测试集）**")
        st.markdown(
            f"<div style='font-size:22px;font-weight:600'>AUC {_fmt(summary.get('auc'))}"
            f" · KS {_fmt(summary.get('ks'))}</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"Gini {_fmt(summary.get('gini'))} · PSI {_fmt(summary.get('psi'), 4)}")

    with c2:
        st.markdown("**合规护栏**")
        passed = summary.get("guardrail_passed")
        review = summary.get("needs_human_review")
        if passed is None:
            st.markdown("_未执行_")
        elif not passed:
            st.error("❌ 未通过护栏")
        elif review:
            st.warning("⚠️ 通过，建议人工复核")
        else:
            st.success("✅ 通过护栏")
        st.caption(
            f"数据守门：{'通过' if summary.get('gate_passed') else '未通过'}"
            if summary.get("gate_passed") is not None else "数据守门：未执行"
        )

    with c3:
        st.markdown("**Agent 自主决策**")
        n_retry = int(summary.get("n_retry") or 0)
        critic = summary.get("critic") or {}
        st.markdown(
            f"<div style='font-size:22px;font-weight:600'>自愈 {n_retry} 轮</div>",
            unsafe_allow_html=True,
        )
        if critic:
            n_issue = critic.get("n_issues", 0)
            source = critic.get("source", "rule")
            tag = "🔴 需人工确认" if summary.get("critic_has_high") else (
                "⚠️ 有中等问题" if n_issue else "✅ 无问题"
            )
            st.caption(f"Critic：{tag} · {n_issue} 项（来源 {source}）")
        else:
            st.caption("Critic：未执行")


# ============================================================================
# 模块 6：数据质量前置守门
# ============================================================================
def render_data_gate_panel(gate: dict[str, Any] | None) -> None:
    st.markdown("#### 模块 6 · 数据质量前置守门")
    st.caption("在切分与建模之前先校验数据是否值得建模；不通过则直接短路终止，不浪费后续算力。")

    if not gate:
        st.info("未执行数据守门（快速训练模式，或流程在守门之前终止）。")
        return

    passed = bool(gate.get("passed"))
    with st.container(border=True):
        if passed:
            st.success("✅ 通过守门")
        else:
            st.error("❌ 未通过守门 —— 流程已在此终止")

        stats = gate.get("stats") or {}
        if stats:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("样本行数", f"{stats.get('n_rows', '-')}")
            c2.metric("特征列数", f"{stats.get('n_cols', '-')}")
            c3.metric("违约样本", f"{stats.get('n_default', '-')}")
            c4.metric("违约率", _fmt(stats.get("default_rate"), pct=True))

        if gate.get("suggestion"):
            st.markdown(f"**Agent 建议**：{gate['suggestion']}")

        blocks = gate.get("blocks") or []
        if blocks:
            st.markdown("**阻断原因**")
            for b in blocks:
                st.markdown(f"- ❌ {b}")

        warns = gate.get("warnings") or []
        if warns:
            st.markdown("**警告**")
            for w in warns:
                st.markdown(f"- ⚠️ {w}")

        filters = gate.get("applied_filters") or []
        if filters:
            st.markdown("**已自动执行的过滤**")
            for f in filters:
                st.markdown(f"- {f}")

        drop = gate.get("suggest_drop") or []
        if drop:
            st.markdown(f"**建议剔除的高缺失/单值列**：{', '.join(f'`{c}`' for c in drop)}")


# ============================================================================
# 模块 1/2：诊断-重规划回路 + 护栏分级自愈
# ============================================================================
def render_retry_panel(summary: dict[str, Any], diagnosis: dict[str, Any] | None = None,
                       reached_modeling: bool = True) -> None:
    st.markdown("#### 模块 1 / 2 · 诊断-重规划回路与分级自愈")
    st.caption(
        "护栏未通过时，Agent 先自己诊断（LLM 决策，失败降级为规则阶梯），"
        "改参数后回跳到最合适的上游节点重跑；超过 max_retry 才转人工。"
    )

    hist = summary.get("retry_history") or []
    if not hist:
        if not reached_modeling:
            st.info("流程在建模之前就终止了（数据守门未通过或关键节点失败），没有产生重规划记录。")
        else:
            st.success("✅ 一次通过，未触发自愈回路（retry = 0）")
        return

    st.info(f"共触发 **{len(hist)}** 轮自动重规划")

    rows = []
    for r in hist:
        patch = r.get("param_patch") or {}
        rows.append({
            "轮次": r.get("round"),
            "动作": r.get("action"),
            "回跳到": r.get("goto"),
            "参数调整": ", ".join(f"{k}={v}" for k, v in patch.items()) or "-",
            "决策来源": r.get("source"),
            "KS": r.get("ks"),
            "AUC": r.get("auc"),
            "PSI": r.get("psi"),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if diagnosis:
        with st.expander("最近一次诊断详情", expanded=True):
            st.markdown(f"- **动作**：`{diagnosis.get('action')}` → 回跳 `{diagnosis.get('goto')}`")
            st.markdown(f"- **决策来源**：`{diagnosis.get('source')}`"
                        f"（`llm` = LLM 决策；`rule` = 规则阶梯兜底；`llm(invalid)` = LLM 输出非法被降级）")
            st.markdown(f"- **参数调整**：`{diagnosis.get('param_patch')}`")
            st.markdown(f"- **诊断理由**：{diagnosis.get('reason', '-')}")


# ============================================================================
# 模块 3：Critic 模型复核
# ============================================================================
def render_critic_panel(critic: dict[str, Any] | None, vif_threshold: float = 10.0,
                        concentration_threshold: float = 0.5) -> None:
    st.markdown("#### 模块 3 · Critic 模型复核")
    st.caption(
        "三项**确定性**审计（VIF 多重共线性 / 系数符号与 WOE 风险方向是否一致 / 样本取值集中度），"
        "LLM 只做定性复核：不得改写结论、不得新增清单外条目、建议里禁止出现数字。"
    )

    if not critic:
        st.info("未执行 Critic 复核。")
        return

    passed = bool(critic.get("passed"))
    n_high = int(critic.get("n_high") or 0)
    with st.container(border=True):
        if passed:
            st.success("✅ 三项审计未发现超限项")
        elif n_high:
            st.error(f"❌ 发现 {n_high} 项高危问题，需人工确认后再上线")
        else:
            st.warning("⚠️ 存在中等问题，建议人工评估影响")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("问题总数", int(critic.get("n_issues") or 0))
        c2.metric("高危 / 中等 / 轻微",
                  f"{n_high} / {critic.get('n_mid', 0)} / {critic.get('n_low', 0)}")
        c3.metric("共线性超限", f"{critic.get('vif_fail', 0)} 列",
                  delta=f"{critic.get('vif_warn', 0)} 列 VIF>5" if critic.get("vif_warn") else None,
                  delta_color="off")
        c4.metric("复核来源", "LLM 已参与" if critic.get("source") == "rule+llm" else "纯规则")

        st.caption(
            f"阈值口径：VIF > {vif_threshold} 判高危、> 5 判中等；"
            f"单一取值占比 > {concentration_threshold:.0%} 判集中度超限"
        )

        issues = critic.get("issues") or []
        if issues:
            st.markdown("**问题明细**")
            df = pd.DataFrame([
                {
                    "严重度": f"{_SEV_ICON.get(i.get('severity'), '')} {_SEV_LABEL.get(i.get('severity'), i.get('severity'))}",
                    "审计项": i.get("check"),
                    "问题": i.get("item"),
                    "处置建议": i.get("suggestion"),
                }
                for i in issues
            ])
            st.dataframe(df, width="stretch", hide_index=True)
        else:
            st.caption("无问题明细。")


# ============================================================================
# 模块 5：cut-off 择优与分档
# ============================================================================
def render_cutoff_panel(cutoff: dict[str, Any] | None, grades: Any = None,
                        detail: Any = None) -> None:
    st.markdown("#### 模块 5 · cut-off 择优与分数分档")
    st.caption(
        "在测试集上按指定口径网格搜索最优分数临界值。**语义：分数越高越安全** —— "
        "拒绝 `score < cut-off`，通过 `score ≥ cut-off`。"
    )

    if not cutoff or cutoff.get("cutoff") is None:
        st.info("未产出 cut-off 结果（可能样本无违约或在择优点退化）。")
        return

    with st.container(border=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("推荐 cut-off", f"{_fmt(cutoff.get('cutoff'), 2)}")
        c2.metric("通过率", _fmt(cutoff.get("approve_rate"), pct=True))
        c3.metric("通过后坏率", _fmt(cutoff.get("bad_rate_after"), pct=True),
                  delta=f"整体 {_fmt(cutoff.get('base_bad_rate'), pct=True)}", delta_color="inverse")
        c4.metric("违约捕获率", _fmt(cutoff.get("capture_rate"), pct=True))
        c5.metric("Lift", f"{_fmt(cutoff.get('lift'), 2)}×")

        st.caption(
            f"择优口径：`{cutoff.get('method')}` · 该点 KS = {_fmt(cutoff.get('ks_at_cutoff'))} · "
            f"误伤率（好客户被拒）= {_fmt(cutoff.get('fpr'), pct=True)}"
        )

        if cutoff.get("warnings"):
            for w in cutoff["warnings"]:
                st.warning(w)

        table = cutoff.get("table")
        if isinstance(table, pd.DataFrame) and not table.empty:
            with st.expander(f"全网格明细（{len(table)} 个候选点）", expanded=False):
                st.dataframe(table, width="stretch", hide_index=True)

    if isinstance(grades, pd.DataFrame) and not grades.empty:
        st.markdown("**分数分档分布（测试集）**")
        counts = (
            grades["grade"].value_counts().sort_index()
            .rename_axis("档位").reset_index(name="样本数")
        )
        counts["占比"] = (counts["样本数"] / counts["样本数"].sum()).map(lambda v: f"{v:.2%}")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(counts, width="stretch", hide_index=True)
        with c2:
            st.bar_chart(counts.set_index("档位")["样本数"])

    if isinstance(detail, pd.DataFrame) and not detail.empty:
        st.markdown("**逐变量分值明细（前 20 行）**")
        st.caption("可用于生成《拒贷理由书》：取 `*_points` 负向贡献最大的变量作为拒绝依据。")
        st.dataframe(detail.head(20), width="stretch", hide_index=True)


# ============================================================================
# 模块 5：开发报告
# ============================================================================
def render_report_panel(report_md: str, report_path: str | None = None,
                        report_name: str = "model_report.md") -> None:
    st.markdown("#### 模块 5 · 《模型开发报告》")
    st.caption(
        "13 章固定结构：概述 / 数据 / 标签 / 特征工程 / 模型参数 / 性能 / 特征重要性 / "
        "评分卡刻度 / cut-off / 护栏 / **Critic 复核** / 风险与局限 / 分箱附录。"
        "章节内全部数值由 Python 插值，LLM 只写三段定性分析且禁止出现数字。"
    )

    if not report_md:
        st.info("未生成报告（快速训练模式，或报告生成被关闭）。")
        return

    if report_path:
        st.caption(f"已落盘：`{report_path}`")

    st.download_button(
        "⬇️ 下载报告（Markdown）",
        data=report_md.encode("utf-8"),
        file_name=report_name,
        mime="text/markdown",
        width="content",
    )
    with st.expander("报告全文", expanded=False):
        st.markdown(report_md)


# ============================================================================
# 护栏终局：人工复核
# ============================================================================
def render_hitl_panel(summary: dict[str, Any]) -> None:
    st.markdown("#### 人工复核（护栏终局）")
    st.caption(
        "Agent 自愈次数已达上限或护栏判死，流程已暂停等待你的决策。"
        "决策不会改变后续路径（都会汇流到报告），但会写入运行记录。"
    )

    gr = summary.get("guardrail_passed")
    if gr is False:
        st.error("护栏未通过")
    if summary.get("metrics_note"):
        st.caption(summary["metrics_note"])

    with st.container(border=True):
        for w in (summary.get("warnings") or [])[-5:]:
            st.markdown(f"- ⚠️ {w}")


# ============================================================================
# 运行摘要（底部：日志与错误）
# ============================================================================
def render_run_log(summary: dict[str, Any]) -> None:
    errs = summary.get("errors") or []
    warns = summary.get("warnings") or []
    st.markdown("#### 运行记录")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**错误（{len(errs)}）**")
        if errs:
            for e in errs:
                st.markdown(f"- ❌ {e}")
        else:
            st.caption("无")
    with c2:
        st.markdown(f"**警告（{len(warns)}）**")
        if warns:
            for w in warns:
                st.markdown(f"- ⚠️ {w}")
        else:
            st.caption("无")
