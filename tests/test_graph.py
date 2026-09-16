"""阶段二测试 — LangGraph 工作流编排

覆盖：
    1. 端到端：合成数据 → 全流程 → 断言 metrics + scorecard
    2. HITL 中断：构造护栏必失败场景 → 断言停在 hitl_review
    3. HITL 恢复：approve → 断言流程走到 END
    4. 持久化：新连接读 SQLite → 断言可恢复中断状态
"""
from __future__ import annotations

import os
import tempfile
import uuid

import numpy as np
import pandas as pd
import pytest
from langgraph.types import Command

from agent.graph import compile_with_sqlite


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def german_credit_df() -> pd.DataFrame:
    """优先 scorecardpy.germancredit()，失败兜底合成数据。"""
    try:
        import scorecardpy as sc
        df = sc.germancredit()
        # scorecardpy 返回字符串标签，转 0/1
        if df["creditability"].dtype == object:
            df["creditability"] = (df["creditability"] == "good").astype(int)
        return df
    except Exception:
        rng = np.random.default_rng(42)
        n = 800
        return pd.DataFrame({
            "duration_in_month": rng.integers(4, 72, n),
            "credit_amount": rng.integers(250, 18000, n),
            "age": rng.integers(19, 75, n),
            "installment_rate": rng.integers(1, 5, n),
            "present_residence_since": rng.integers(1, 5, n),
            "number_of_existing_credits": rng.integers(1, 5, n),
            "creditability": rng.integers(0, 2, n),
        })


@pytest.fixture
def csv_path(tmp_path, german_credit_df) -> str:
    p = tmp_path / "data.csv"
    german_credit_df.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def db_path(tmp_path) -> str:
    p = tmp_path / "checkpoints.db"
    return str(p)


@pytest.fixture
def fresh_thread() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# 测试 1: 端到端（默认配置）
# ---------------------------------------------------------------------------
def test_end_to_end_default(csv_path, db_path, fresh_thread):
    app, conn = compile_with_sqlite(db_path)
    try:
        cfg = {"configurable": {"thread_id": fresh_thread}}
        result = app.invoke(
            {
                "csv_path": csv_path,
                "target": "creditability",
                "config_overrides": {"generate_report": False},  # 避免测试污染真实 reports/ 目录
            },
            config=cfg,
        )
        assert "metrics" in result, "metrics 缺失"
        assert result["metrics"]["auc"] > 0.5, f"AUC={result['metrics']['auc']} 应 > 0.5"
        assert "scorecard" in result, "scorecard 缺失"
        scored = result.get("scored_test")
        assert scored is not None and "score" in scored.columns, "scored_test 缺 score 列"
        assert result.get("guardrail") is not None, "guardrail 缺失"
        print(f"\n[test_end_to_end_default] metrics={result['metrics']}")
        print(f"[test_end_to_end_default] guardrail.passed={result['guardrail'].passed}")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 测试 1b: 【V2 批次 C】护栏通过 → 必经 reporter → 产出报告/cut-off/分档
# ---------------------------------------------------------------------------
def test_end_to_end_reaches_reporter(csv_path, db_path, fresh_thread):
    """放宽护栏阈值 + 关闭人审 → 图应走到 reporter 并落产出 report_md。"""
    app, conn = compile_with_sqlite(db_path)
    try:
        cfg = {"configurable": {"thread_id": fresh_thread}}
        result = app.invoke(
            {
                "csv_path": csv_path,
                "target": "creditability",
                "config_overrides": {
                    "min_ks": 0.1, "min_auc": 0.6, "max_psi": 0.9,
                    "require_human_review": False,
                    "generate_report": False,   # 避免测试污染真实 reports/ 目录
                },
            },
            config=cfg,
        )
        assert result.get("report_md"), "reporter 未产出 report_md"
        assert "一、项目概述" in result["report_md"]
        assert "十三、附录：分箱明细" in result["report_md"]
        assert "十一、模型复核结论（Critic）" in result["report_md"]
        cutoff = result.get("cutoff_info")
        assert cutoff and cutoff.get("cutoff") is not None, "cut-off 择优缺失"
        assert result.get("score_grades") is not None, "分数分档缺失"
        assert result.get("feature_importance") is not None, "特征重要性缺失"
        print(f"\n[reporter] cutoff={cutoff['cutoff']:.1f} "
              f"通过率={cutoff['approve_rate']:.2%} 报告字数={len(result['report_md'])}")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 测试 2: HITL 中断（构造护栏必失败）
# ---------------------------------------------------------------------------
def test_hitl_interrupt(csv_path, db_path, fresh_thread):
    """min_ks/min_auc 设到不可能达到 → guardrail.passed=False → 必触发 HITL。"""
    app, conn = compile_with_sqlite(db_path)
    try:
        cfg = {"configurable": {"thread_id": fresh_thread}}
        app.invoke(
            {
                "csv_path": csv_path,
                "target": "creditability",
                "config_overrides": {"min_ks": 0.99, "min_auc": 0.99, "generate_report": False},
            },
            config=cfg,
        )
        snap = app.get_state(cfg)
        assert snap.next, f"应在 HITL 处暂停，但 next 为空: values={snap.values}"
        # next 应包含 hitl_review
        next_nodes = list(snap.next)
        assert any("hitl" in str(n) for n in next_nodes), f"next 不含 hitl_review: {next_nodes}"
        print(f"\n[test_hitl_interrupt] next={next_nodes}")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 测试 3: HITL 恢复 approve → 走完
# ---------------------------------------------------------------------------
def test_hitl_resume_approve(csv_path, db_path, fresh_thread):
    app, conn = compile_with_sqlite(db_path)
    try:
        cfg = {"configurable": {"thread_id": fresh_thread}}
        app.invoke(
            {
                "csv_path": csv_path,
                "target": "creditability",
                "config_overrides": {"min_ks": 0.99, "min_auc": 0.99, "generate_report": False},
            },
            config=cfg,
        )
        # resume approve
        result = app.invoke(
            Command(resume={"action": "approve", "note": "approved by test"}),
            config=cfg,
        )
        assert result["hitl_decision"] == "approve", f"got {result.get('hitl_decision')}"
        assert result["hitl_note"] == "approved by test", f"got {result.get('hitl_note')}"
        snap = app.get_state(cfg)
        assert not snap.next, "resume 后应走到 END"
        print(f"\n[test_hitl_resume_approve] hitl_decision={result['hitl_decision']}")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 测试 4: 持久化（新连接读 SQLite 应能恢复中断状态）
# ---------------------------------------------------------------------------
def test_state_persistence(csv_path, db_path, fresh_thread):
    """第一次 invoke 后关闭连接，模拟进程重启，再用新连接 resume。"""
    cfg = {"configurable": {"thread_id": fresh_thread}}
    # 第一次跑
    app1, conn1 = compile_with_sqlite(db_path)
    try:
        app1.invoke(
            {
                "csv_path": csv_path,
                "target": "creditability",
                "config_overrides": {"min_ks": 0.99, "min_auc": 0.99, "generate_report": False},
            },
            config=cfg,
        )
        snap1 = app1.get_state(cfg)
        assert snap1.next, "应在 HITL 处暂停"
    finally:
        conn1.close()

    # 模拟重启：新连接
    app2, conn2 = compile_with_sqlite(db_path)
    try:
        snap2 = app2.get_state(cfg)
        assert snap2.next, "重启后应能恢复中断状态"
        # resume
        result = app2.invoke(
            Command(resume={"action": "approve", "note": "resumed after restart"}),
            config=cfg,
        )
        assert result["hitl_decision"] == "approve"
        print(f"\n[test_state_persistence] 重启后成功 resume, hitl_note={result['hitl_note']}")
    finally:
        conn2.close()


# ---------------------------------------------------------------------------
# 测试 5: 节点错误捕获（构造非法 csv）
# ---------------------------------------------------------------------------
def test_error_capture(tmp_path, db_path, fresh_thread):
    """CSV 不存在 → load_data 应捕获错误并写入 errors。"""
    app, conn = compile_with_sqlite(db_path)
    try:
        cfg = {"configurable": {"thread_id": fresh_thread}}
        result = app.invoke(
            {
                "csv_path": str(tmp_path / "nonexistent.csv"),
                "target": "creditability",
                "config_overrides": {"generate_report": False},  # 避免测试污染真实 reports/ 目录
            },
            config=cfg,
        )
        errors = result.get("errors", [])
        assert any("load_data" in e for e in errors), f"应有 load_data 错误: {errors}"
        print(f"\n[test_error_capture] errors={errors}")
    finally:
        conn.close()