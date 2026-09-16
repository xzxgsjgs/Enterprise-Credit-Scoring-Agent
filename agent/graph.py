"""agent.graph — StateGraph 构建与 checkpointer 装配

【V2 拓扑】（相比 v1 新增：data_gate 守门 + 错误短路 + diagnose 回跳 + 分级自愈）
    START
      → load_data ──✗(critical)──→ END
      → explore_data → data_gate ──✗(未过守门)──→ END
      → split ──✗(critical)──→ END
      → preprocess → var_filter → woebin ──✗(critical)──→ END
      → woebin_apply → model_train ──✗(critical)──→ END
      → model_predict → evaluate_performance → build_scorecard
      → critic（审计，不阻塞）→ scorecard_ply → guardrail_check
      ├─ 通过且无需人审                 → reporter → END
      ├─ 未通过且 retry_count < max_retry → diagnose ──┐
      └─ 其他 / 已达上限                 → hitl_review ─┴→ reporter → END
                                                        │
        diagnose 决策回跳 ◄─────────────────────────────┘
          goto ∈ {preprocess, var_filter, woebin, model_train, build_scorecard}
          或 escalate/abort → hitl_review

持久化：SqliteSaver（单机 POC），文件落盘到 ./checkpoints.db
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    critic_node,
    data_gate_node,
    diagnose_node,
    evaluate_node,
    explore_data_node,
    guardrail_node,
    hitl_review_node,
    load_data_node,
    predict_node,
    preprocess_node,
    reporter_node,
    scorecard_apply_node,
    scorecard_node,
    split_node,
    train_node,
    var_filter_node,
    woebin_apply_node,
    woebin_node,
)
from agent.routing import (
    route_after_critical,
    route_after_data_gate,
    route_after_diagnose,
    route_after_guardrail,
    route_after_hitl,
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
    builder.add_node("data_gate", data_gate_node)          # 【V2】数据质量守门
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
    builder.add_node("diagnose", diagnose_node)                 # 【V2】诊断-重规划
    builder.add_node("reporter", reporter_node)                 # 【V2】报告 + cut-off
    builder.add_node("critic", critic_node)                     # 【V2】Critic 复核（不阻塞）
    builder.add_node("hitl_review", hitl_review_node)

    # ---- 边（不可恢复节点后挂短路条件边）----
    builder.add_edge(START, "load_data")
    builder.add_conditional_edges(
        "load_data", route_after_critical, {"end": END, "continue": "explore_data"}
    )
    builder.add_edge("explore_data", "data_gate")

    # 【V2】data_gate：未通过直接 END，不再硬跑后续节点
    builder.add_conditional_edges(
        "data_gate", route_after_data_gate, {"end": END, "continue": "split"}
    )

    builder.add_conditional_edges(
        "split", route_after_critical, {"end": END, "continue": "preprocess"}
    )
    builder.add_edge("preprocess", "var_filter")
    builder.add_edge("var_filter", "woebin")
    builder.add_conditional_edges(
        "woebin", route_after_critical, {"end": END, "continue": "woebin_apply"}
    )
    builder.add_edge("woebin_apply", "model_train")
    builder.add_conditional_edges(
        "model_train", route_after_critical, {"end": END, "continue": "model_predict"}
    )
    builder.add_edge("model_predict", "evaluate_performance")
    builder.add_edge("evaluate_performance", "build_scorecard")
    # 【V2 模块 3】Critic 审计（规则先行，LLM 可降级复核；失败只记 warning，不阻塞）
    builder.add_edge("build_scorecard", "critic")
    builder.add_edge("critic", "scorecard_ply")
    builder.add_edge("scorecard_ply", "guardrail_check")

    # ---- 条件边：guardrail → reporter / diagnose / hitl_review ----
    # 【V2 模块 3】「end」分支先过 reporter（汇总报告 + cut-off），再由 reporter → END
    builder.add_conditional_edges(
        "guardrail_check",
        route_after_guardrail,
        {"end": "reporter", "diagnose": "diagnose", "hitl": "hitl_review"},
    )

    # ---- 条件边：diagnose 回跳到指定重跑节点（或转人工）----
    builder.add_conditional_edges(
        "diagnose",
        route_after_diagnose,
        {
            "preprocess": "preprocess",
            "var_filter": "var_filter",
            "woebin": "woebin",
            "model_train": "model_train",
            "build_scorecard": "build_scorecard",
            "hitl": "hitl_review",
        },
    )

    # ---- 条件边：hitl 之后同样汇流到 reporter 再 END ----
    builder.add_conditional_edges(
        "hitl_review",
        route_after_hitl,
        {"end": "reporter"},
    )

    # ---- reporter 为终局节点 ----
    builder.add_edge("reporter", END)

    return builder


def compile_with_sqlite(db_path: str | Path = "checkpoints.db"):
    """编译 StateGraph，绑定 SqliteSaver checkpointer。

    Args:
        db_path: SQLite 文件路径，目录不存在会自动创建

    Returns:
        (CompiledStateGraph, sqlite3.Connection)；调用方负责 close 连接
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    builder = build_graph()
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


# 保留 route_after_diagnose 的 re-export，便于批次 B 直接引用
__all__ = ["build_graph", "compile_with_sqlite", "get_app", "route_after_diagnose"]