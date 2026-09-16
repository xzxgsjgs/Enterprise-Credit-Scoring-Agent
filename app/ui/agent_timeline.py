"""app.ui.agent_timeline —— Agent 18 节点执行时间线。

设计目标：让用户一眼看出「Agent 走到哪一步、哪一步出的问题、有没有回跳重试」，
而不是像 CLI 那样刷一屏 `[OK] xxx` 日志。

状态语义：
- ○ 待执行   / ● 运行中   / ✅ 成功   / ⚠️ 有告警   / ❌ 失败   / 🔁 回跳重规划
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from app.core.agent_runner import (  # noqa: F401  (NODE_ORDER 由页面复用)
    CRITICAL_NODES,
    NODE_ORDER,
    node_label,
    node_module,
)

_STATUS_ICON = {
    "pending": "○",
    "running": "●",
    "ok": "✅",
    "warn": "⚠️",
    "fail": "❌",
    "retry": "🔁",
}

_STATUS_COLOR = {
    "pending": "#9aa5b1",
    "running": "#2b6cb0",
    "ok": "#2f855a",
    "warn": "#b7791f",
    "fail": "#c53030",
    "retry": "#6b46c1",
}

_STATUS_TEXT = {
    "pending": "待执行",
    "running": "运行中",
    "ok": "完成",
    "warn": "警告",
    "fail": "失败",
    "retry": "回跳重规划",
}


def init_timeline() -> dict[str, dict[str, Any]]:
    """初始化时间线状态（每个节点一条）。"""
    return {
        n: {"status": "pending", "detail": "", "runs": 0}
        for n in NODE_ORDER
    }


def apply_event(timeline: dict[str, dict[str, Any]], event: Any) -> dict[str, dict[str, Any]]:
    """把一次节点事件合并进时间线（原地修改并返回）。"""
    node = getattr(event, "node", None) or (event.get("node") if isinstance(event, dict) else None)
    if not node:
        return timeline
    status = getattr(event, "status", None) or (event.get("status") if isinstance(event, dict) else "ok")
    detail = getattr(event, "detail", "") or (event.get("detail", "") if isinstance(event, dict) else "")

    slot = timeline.setdefault(str(node), {"status": "pending", "detail": "", "runs": 0})
    slot["status"] = status
    slot["detail"] = detail
    slot["runs"] = int(slot.get("runs", 0)) + 1
    return timeline


def render_timeline(
    timeline: dict[str, dict[str, Any]],
    current: str | None = None,
    compact: bool = False,
) -> None:
    """渲染时间线。

    Args:
        timeline: init_timeline() 的结构，已被 apply_event 更新过
        current: 当前正在执行的节点（渲染成「运行中」）
        compact: True → 只渲染一行摘要 + 折叠明细
    """
    if compact:
        done = sum(1 for v in timeline.values() if v["status"] not in ("pending", "running"))
        total = len(timeline)
        high = [k for k, v in timeline.items() if v["status"] == "fail"]
        label = f"执行时间线 · {done}/{total} 节点已执行"
        if high:
            label += f" · ❌ {len(high)} 个节点失败"
        with st.expander(label, expanded=False):
            _render_rows(timeline, current)
        return
    _render_rows(timeline, current)


def _render_rows(timeline: dict[str, dict[str, Any]], current: str | None) -> None:
    rows: list[str] = []
    for node in NODE_ORDER:
        slot = timeline.get(node) or {"status": "pending", "detail": "", "runs": 0}
        status = "running" if (current == node and slot["status"] in ("pending", "running")) else slot["status"]
        icon = _STATUS_ICON.get(status, "○")
        color = _STATUS_COLOR.get(status, "#9aa5b1")
        label = node_label(node)
        module = node_module(node)
        badge = f"<span style='color:#718096;font-size:11px'>{module}</span>"
        if node in CRITICAL_NODES:
            badge += " <span style='color:#c53030;font-size:11px'>关键</span>"
        runs = slot.get("runs", 0)
        run_tag = (
            f" <span style='color:#6b46c1;font-size:11px'>×{runs}</span>"
            if runs > 1 else ""
        )
        detail = slot.get("detail") or ("运行中…" if status == "running" else "")
        detail_html = (
            f"<div style='color:#4a5568;font-size:12px;margin-left:26px;"
            f"word-break:break-all'>{_esc(detail)}</div>" if detail else ""
        )
        rows.append(
            f"<div style='padding:3px 0'>"
            f"<span style='color:{color};font-weight:600;font-size:13px'>{icon}</span>"
            f"<span style='margin-left:8px;color:{color};font-size:13px;font-weight:600'>{label}</span>"
            f"{run_tag} <span style='margin-left:6px'>{badge}</span>"
            f"<span style='float:right;color:{color};font-size:11px'>{_STATUS_TEXT.get(status, status)}</span>"
            f"{detail_html}"
            f"</div>"
        )
    st.markdown(
        "<div style='border:1px solid rgba(128,128,128,.25);border-radius:8px;"
        "padding:10px 14px;'>" + "".join(rows) + "</div>",
        unsafe_allow_html=True,
    )


def _esc(s: Any) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def timeline_summary(timeline: dict[str, dict[str, Any]]) -> dict[str, int]:
    """统计各状态节点数，供顶部指标卡使用。"""
    out: dict[str, int] = {}
    for v in timeline.values():
        out[v["status"]] = out.get(v["status"], 0) + 1
    return out
