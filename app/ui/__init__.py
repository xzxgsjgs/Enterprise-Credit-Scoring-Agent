"""app.ui —— Streamlit 可复用 UI 组件。"""
from app.ui.agent_panels import (
    render_critic_panel,
    render_cutoff_panel,
    render_data_gate_panel,
    render_hitl_panel,
    render_report_panel,
    render_retry_panel,
    render_run_log,
    render_run_summary,
)
from app.ui.agent_timeline import (
    apply_event,
    init_timeline,
    render_timeline,
    timeline_summary,
)
from app.ui.guardrail_panel import render_guardrail_panel
from app.ui.metrics_card import render_metrics_cards
from app.ui.score_distribution import render_score_distribution

__all__ = [
    # 基础组件
    "render_metrics_cards",
    "render_guardrail_panel",
    "render_score_distribution",
    # Agent V2 组件
    "init_timeline",
    "apply_event",
    "render_timeline",
    "timeline_summary",
    "render_run_summary",
    "render_data_gate_panel",
    "render_retry_panel",
    "render_critic_panel",
    "render_cutoff_panel",
    "render_report_panel",
    "render_hitl_panel",
    "render_run_log",
]
