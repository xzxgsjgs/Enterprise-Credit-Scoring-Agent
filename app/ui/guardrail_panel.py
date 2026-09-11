"""app.ui.guardrail_panel —— 护栏结果展示与 LLM 解释。"""
from __future__ import annotations

import json
from typing import Any

import streamlit as st

from app.llm import GUARDRAIL_INTERPRETATION, get_model


def render_guardrail_panel(guardrail: Any, metrics: dict) -> None:
    """渲染护栏判定面板，并提供「LLM 解释」按钮。"""
    gr = guardrail
    passed = getattr(gr, "passed", False)
    requires_human_review = getattr(gr, "requires_human_review", False)
    warnings = getattr(gr, "warnings", []) or []
    rationale = getattr(gr, "rationale", "")

    if not passed:
        status_color = "error"
        status_text = "❌ 未通过护栏，禁止上线"
    elif requires_human_review:
        status_color = "warning"
        status_text = "⚠️ 通过但建议人工复核"
    else:
        status_color = "success"
        status_text = "✅ 通过护栏"

    with st.container(border=True):
        st.subheader("合规护栏")
        getattr(st, status_color)(status_text)
        st.caption(rationale)

        if warnings:
            with st.expander("警告详情"):
                for w in warnings:
                    st.write(f"- {w}")

        if st.button("🧠 让 LLM 解释护栏结果", key="interpret_guardrail"):
            try:
                llm = get_model()
                prompt = GUARDRAIL_INTERPRETATION.format(
                    guardrail_json=json.dumps(
                        {
                            "passed": passed,
                            "requires_human_review": requires_human_review,
                            "warnings": warnings,
                            "rationale": rationale,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    metrics_json=json.dumps(metrics, ensure_ascii=False, indent=2, default=float),
                )
                with st.spinner("LLM 思考中..."):
                    response = llm.invoke(prompt)
                st.markdown(response.content)
            except Exception as e:
                st.error(f"LLM 调用失败: {e}")
