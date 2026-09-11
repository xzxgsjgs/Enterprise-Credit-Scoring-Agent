"""app.llm —— Streamlit 侧 LLM 客户端与提示词。"""
from app.llm.client import get_model, list_models, list_presets, resolve_default_label
from app.llm.prompts import (
    GUARDRAIL_INTERPRETATION,
    HITL_ADVICE,
    SCORE_EXPLANATION,
    VARIABLE_IMPORTANCE,
)

__all__ = [
    "get_model",
    "list_models",
    "list_presets",
    "GUARDRAIL_INTERPRETATION",
    "HITL_ADVICE",
    "SCORE_EXPLANATION",
    "VARIABLE_IMPORTANCE",
]
