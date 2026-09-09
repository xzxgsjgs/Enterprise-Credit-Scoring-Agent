"""agent.serde — 自定义 LangGraph 序列化器

默认 JsonPlusSerializer 用 ormsgpack，不能处理 DataFrame / numpy array / sklearn model。
本模块提供一个包装版：先尝试 JsonPlusSerializer，失败则用 pickle fallback。

接口规范参考 langgraph.checkpoint.serde.base.SerializerProtocol:
    dumps_typed(obj) -> (type_str, data_bytes)
    loads_typed((type_str, data_bytes)) -> obj
"""
from __future__ import annotations

import pickle
from typing import Any

from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


class PickleFallbackSerializer(SerializerProtocol):
    """serde 包装：失败 → pickle。

    优先用 JsonPlusSerializer（兼容 LangChain/LangGraph 常见类型且更紧凑），
    DataFrame / numpy / sklearn model 等 ormsgpack 不支持的，fallback 到 pickle。
    """

    def __init__(self) -> None:
        self._primary = JsonPlusSerializer()

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        try:
            return self._primary.dumps_typed(obj)
        except Exception:
            return ("pickle", pickle.dumps(obj))

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        type_, raw = data
        if type_ == "pickle":
            return pickle.loads(raw)
        return self._primary.loads_typed(data)