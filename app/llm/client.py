"""app.llm.client —— 多模型切换客户端。

设计原则：
- 模型列表与密钥全部来自 .env，本文件不读取真实 key 的值，只通过 os.getenv 按名引用。
- LLM_DEFAULT 通过「去 emoji + 大小写不敏感」与 label 模糊匹配。
- 支持任意 OpenAI 兼容接口（百炼、硅基流动、本地 vLLM 等）。
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from importlib import import_module
from typing import Any

try:
    _chat_models = import_module("langchain.chat_models")
    init_chat_model = _chat_models.init_chat_model
except (ImportError, AttributeError) as exc:
    _langchain_import_error = exc

    def init_chat_model(*args: Any, **kwargs: Any):
        raise RuntimeError(
            "未安装或版本不兼容的 LangChain，请安装支持 init_chat_model 的 langchain。"
        ) from _langchain_import_error


@dataclass(frozen=True)
class ModelPreset:
    label: str
    provider: str
    model: str
    base_url: str
    api_key_env: str


def _strip_emoji(text: str) -> str:
    """移除 emoji 与特殊符号，保留字母/数字/空格/点/横线，用于模糊匹配。"""
    cleaned = re.sub(r"[^\w\s.-]", "", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def _load_presets() -> list[ModelPreset]:
    raw = os.getenv("LLM_PRESETS", "[]")
    if not raw or raw.strip() == "":
        return []
    try:
        items = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"LLM_PRESETS 不是合法 JSON: {e}\n"
            "常见原因：.env 中 LLM_PRESETS 被拆成了多行，或对象之间缺少逗号。\n"
            "正确格式：单行 JSON array，例如：\n"
            'LLM_PRESETS=[{"label":"Qwen Turbo","provider":"openai","model":"qwen-turbo","base_url":"https://dashscope.aliyuncs.com/compatible-mode/v1","api_key_env":"DASHSCOPE_API_KEY"}]'
        ) from e
    if not isinstance(items, list):
        raise RuntimeError("LLM_PRESETS 必须是 JSON array")
    presets = []
    for idx, it in enumerate(items):
        try:
            presets.append(ModelPreset(**it))
        except TypeError as e:
            raise RuntimeError(f"LLM_PRESETS[{idx}] 字段缺失: {e}") from e
    return presets


def list_presets() -> list[ModelPreset]:
    """返回所有模型预设。"""
    return _load_presets()


def list_models() -> list[str]:
    """返回所有模型 label 列表，用于 Streamlit 下拉框。"""
    return [p.label for p in _load_presets()]


def resolve_default_label(presets: list[ModelPreset] | None = None) -> str:
    """根据 LLM_DEFAULT 解析默认 label；匹配失败时回退第一条。"""
    presets = presets or _load_presets()
    if not presets:
        return ""
    default = os.getenv("LLM_DEFAULT", "")
    default_clean = _strip_emoji(default).lower()
    if not default_clean:
        return presets[0].label

    # 1) 精确匹配（去 emoji后）
    for p in presets:
        if _strip_emoji(p.label).lower() == default_clean:
            return p.label

    # 2) 子串匹配
    for p in presets:
        if default_clean in _strip_emoji(p.label).lower():
            return p.label

    # 3) 回退第一条
    return presets[0].label


def get_model(
    label: str | None = None,
    temperature: float | None = None,
    timeout: int | None = None,
    **kwargs: Any,
):
    """初始化并返回 LangChain chat model。

    Args:
        label: 模型 label；None 时使用 LLM_DEFAULT 解析结果。
        temperature: 覆盖 LLM_TEMPERATURE（诊断类决策用 0，保证可复现）。
        timeout: 覆盖 LLM_TIMEOUT（秒）。
        **kwargs: 额外传给 init_chat_model 的参数。

    Returns:
        BaseChatModel 实例。
    """
    presets = _load_presets()
    if not presets:
        raise RuntimeError("LLM_PRESETS 未配置，请先配置 .env")

    target_label = label or resolve_default_label(presets)
    target_clean = _strip_emoji(target_label).lower()

    preset = None
    for p in presets:
        if _strip_emoji(p.label).lower() == target_clean:
            preset = p
            break
    if preset is None:
        for p in presets:
            if target_clean in _strip_emoji(p.label).lower():
                preset = p
                break
    if preset is None:
        preset = presets[0]

    api_key = os.getenv(preset.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"环境变量 {preset.api_key_env} 未设置（当前模型: {preset.label}）"
        )

    temperature = float(
        temperature if temperature is not None else os.getenv("LLM_TEMPERATURE", "0.2")
    )
    timeout = int(timeout if timeout is not None else os.getenv("LLM_TIMEOUT", "30"))

    return init_chat_model(
        model=preset.model,
        model_provider=preset.provider,
        base_url=preset.base_url,
        api_key=api_key,
        temperature=temperature,
        timeout=timeout,
        **kwargs,
    )
