"""agent.graph — StateGraph 构建与 checkpointer 装配

拓扑：
    START
      → load_data → explore_data → split → preprocess
      → var_filter → woebin → woebin_apply → model_train
      → model_predict → evaluate_performance → build_scorecard
      → scorecard_ply → guardrail_check
      ├─ passed → END
      └─ critical / warning → hitl_review → END

持久化：SqliteSaver（单机 POC），文件落盘到 ./checkpoints.db
"""
from __future__ import annotations

import logging
from pathlib import Path

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.nodes import (
    evaluate_node,
    explore_data_node,
    guardrail_node,
    hitl_review_node,
    load_data_node,
    predict_node,
    preprocess_node,
    route_after_guardrail,
    route_after_hitl,
    scorecard_apply_node,
    scorecard_node,
    split_node,
    train_node,
    var_filter_node,
    woebin_apply_node,
    woebin_node,
)
from agent.serde import PickleFallbackSerializer
from agent.state import AgentState

logger = logging.getLogger(__name__)


# ============================================================================
# 图构建
# ============================================================================
def build_graph() -> StateGraph:
    """构建 StateGraph（未编译），便于测试时替换 checkpointer。"""
    builder = StateGraph(AgentState)

    # ---- 节点注册 ----
    builder.add_node("load_data", load_data_node)
    builder.add_node("explore_data", explore_data_node)
    builder.add_node("split", split_node)
    builder.add_node("preprocess", preprocess_node)
    builder.add_node("var_filter", var_filter_node)
    builder.add_node("woebin", woebin_node)
    builder.add_node("woebin_apply", woebin_apply_node)
    builder.add_node("model_train", train_node)
    builder.add_node("model_predict", predict_node)
    builder.add_node("evaluate_performance", evaluate_node)
    builder.add_node("build_scorecard", scorecard_node)
    builder.add_node("scorecard_ply", scorecard_apply_node)
    builder.add_node("guardrail_check", guardrail_node)
    builder.add_node("hitl_review", hitl_review_node)

    # ---- 边 ----
    builder.add_edge(START, "load_data")
    builder.add_edge("load_data", "explore_data")
    builder.add_edge("explore_data", "split")
    builder.add_edge("split", "preprocess")
    builder.add_edge("preprocess", "var_filter")
    builder.add_edge("var_filter", "woebin")
    builder.add_edge("woebin", "woebin_apply")
    builder.add_edge("woebin_apply", "model_train")
    builder.add_edge("model_train", "model_predict")
    builder.add_edge("model_predict", "evaluate_performance")
    builder.add_edge("evaluate_performance", "build_scorecard")
    builder.add_edge("build_scorecard", "scorecard_ply")
    builder.add_edge("scorecard_ply", "guardrail_check")

    # ---- 条件边：guardrail → END 或 hitl_review ----
    builder.add_conditional_edges(
        "guardrail_check",
        route_after_guardrail,
        {"end": END, "hitl": "hitl_review"},
    )

    # ---- 条件边：hitl_review → END ----
    builder.add_conditional_edges(
        "hitl_review",
        route_after_hitl,
        {"end": END},
    )

    return builder


def compile_with_sqlite(db_path: str | Path = "checkpoints.db"):
    """编译 StateGraph，绑定 SqliteSaver checkpointer。

    Args:
        db_path: SQLite 文件路径，目录不存在会自动创建

    Returns:
        CompiledStateGraph（含 checkpointer）
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    builder = build_graph()
    # SqliteSaver 是上下文管理器，编译时直接传入实例
    # 编译后的 app 在使用时需保持 saver 上下文
    # 这里采用 in-memory 连接的简单写法（适合 POC）
    import sqlite3
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    serde = PickleFallbackSerializer()
    checkpointer = SqliteSaver(conn, serde=serde)
    app = builder.compile(checkpointer=checkpointer)
    logger.info("图已编译: SqliteSaver → %s (serde=pickle-fallback)", db_path)
    return app, conn


# ============================================================================
# 便捷工厂
# ============================================================================
def get_app(db_path: str | Path = "checkpoints.db"):
    """获取已编译的 app 与底层连接（调用方负责 close 连接）。"""
    return compile_with_sqlite(db_path)