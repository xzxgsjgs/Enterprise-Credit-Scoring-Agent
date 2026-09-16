"""批次 B 测试 — 诊断-重规划回路（模块 1）与护栏失败分级自愈（模块 2）

分层：
    A. _validate_diagnosis 净化层（白名单 / 夹取 / choices / 非法 goto）
    B. _rule_based_diagnosis 规则急救阶梯
    C. diagnose_node 降级与计数
    D. route_after_diagnose 回跳路由
    E. 端到端：护栏必失败场景下的自动回跳 + 超限转人工

所有测试不依赖 LLM（LLM 路径用 monkeypatch 强制失败来覆盖降级分支）。
"""
from __future__ import annotations

import uuid

import numpy as np
import pandas as pd
import pytest
from langgraph.types import Command

from agent.nodes import (
    _merge_config,
    _plan_signature,
    _rule_based_diagnosis,
    _validate_diagnosis,
    diagnose_node,
)
from agent.routing import route_after_diagnose


# ---------------------------------------------------------------------------
# Fixtures（与 test_graph.py 同源，独立声明以免跨模块 fixture 解析歧义）
# ---------------------------------------------------------------------------
@pytest.fixture
def csv_path(tmp_path) -> str:
    """1000 行 german credit 合成数据。"""
    rng = np.random.default_rng(42)
    n = 1000
    df = pd.DataFrame({
        "duration_in_month": rng.integers(4, 72, n),
        "credit_amount": rng.integers(250, 18000, n),
        "age": rng.integers(19, 75, n),
        "installment_rate": rng.integers(1, 5, n),
        "present_residence_since": rng.integers(1, 5, n),
        "number_of_existing_credits": rng.integers(1, 5, n),
    })
    # 构造有区分度的标签，保证 KS/AUC 真实可算
    logit = (
        0.02 * df["duration_in_month"]
        - 0.00006 * df["credit_amount"]
        - 0.03 * df["age"]
        + 0.4 * df["installment_rate"]
        + rng.normal(0, 0.6, n)
    )
    df["creditability"] = (logit > logit.mean()).astype(int)
    p = tmp_path / "loop_data.csv"
    df.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def db_path(tmp_path) -> str:
    return str(tmp_path / "checkpoints.db")


@pytest.fixture
def fresh_thread() -> str:
    return str(uuid.uuid4())


@pytest.fixture
def no_llm(monkeypatch):
    """强制 LLM 不可用，覆盖 diagnose 的降级分支。"""
    import app.llm.client as cli

    def boom(*a, **k):
        raise RuntimeError("no api key")

    monkeypatch.setattr(cli, "get_model", boom, raising=False)
    return None


# ---------------------------------------------------------------------------
# A. _validate_diagnosis 净化层
# ---------------------------------------------------------------------------
def test_validate_diagnosis_accepts_valid_retune():
    raw = {
        "action": "retune",
        "goto": "woebin",
        "param_patch": {"max_bins": 6},
        "reason": "PSI 超限",
    }
    diag, warns = _validate_diagnosis(raw, {})
    assert diag["action"] == "retune"
    assert diag["goto"] == "woebin"
    assert diag["param_patch"] == {"max_bins": 6}
    assert diag["source"] == "llm"
    assert warns == []


def test_validate_diagnosis_drops_unknown_param():
    raw = {"action": "retune", "goto": "woebin", "param_patch": {"drop_all_data": True, "max_bins": 5}}
    diag, warns = _validate_diagnosis(raw, {})
    assert "max_bins" in diag["param_patch"]
    assert "drop_all_data" not in diag["param_patch"]
    assert any("不在白名单" in w for w in warns)


def test_validate_diagnosis_clamps_out_of_range():
    raw = {"action": "retune", "goto": "woebin", "param_patch": {"max_bins": 999, "min_bin_size": 0.9}}
    diag, warns = _validate_diagnosis(raw, {})
    assert diag["param_patch"]["max_bins"] == 12          # 上界
    assert diag["param_patch"]["min_bin_size"] == 0.2     # 上界
    assert len(warns) == 2
    assert all("夹取" in w for w in warns)


def test_validate_diagnosis_rejects_bad_choices():
    raw = {
        "action": "retune", "goto": "woebin",
        "param_patch": {"binning_method": "magic", "missing_strategy": "magic"},
    }
    diag, warns = _validate_diagnosis(raw, {})
    # 两个非法枚举都被丢弃 → patch 为空 → retune 降级为 escalate
    assert diag["action"] == "escalate"
    assert diag["goto"] == "hitl_review"
    assert len(warns) >= 3  # 2 个枚举 + 1 个降级


def test_validate_diagnosis_invalid_goto_downgrades():
    raw = {"action": "retune", "goto": "drop_database", "param_patch": {"max_bins": 6}}
    diag, warns = _validate_diagnosis(raw, {})
    assert diag["action"] == "escalate"
    assert diag["goto"] == "hitl_review"
    assert diag["param_patch"] == {}
    assert any("goto" in w for w in warns)


def test_validate_diagnosis_non_dict():
    diag, warns = _validate_diagnosis("oops", {})
    assert diag["action"] == "escalate"
    assert warns


def test_validate_diagnosis_escalate_keeps_no_patch():
    raw = {"action": "escalate", "goto": "var_filter", "param_patch": {"max_bins": 4}, "reason": "无解"}
    diag, _ = _validate_diagnosis(raw, {})
    assert diag["action"] == "escalate"
    assert diag["goto"] == "hitl_review"
    assert diag["param_patch"] == {}


def test_validate_diagnosis_bool_and_list_normalization():
    raw = {
        "action": "retune", "goto": "model_train",
        "param_patch": {"use_lgbm": "true", "exclude_vars": "A, B"},
    }
    diag, _ = _validate_diagnosis(raw, {})
    assert diag["param_patch"]["use_lgbm"] is True
    assert diag["param_patch"]["exclude_vars"] == ["A", "B"]


# ---------------------------------------------------------------------------
# B. _rule_based_diagnosis 规则阶梯
# ---------------------------------------------------------------------------
def _state_with(ks=0.5, auc=0.8, psi=0.05, **extra):
    from tools.validation_tools import GuardrailResult

    gr = GuardrailResult(
        passed=False, metrics={"ks": ks, "auc": auc, "psi": psi},
        warnings=[], requires_human_review=True,
    )
    st = {
        "metrics": {"ks": ks, "auc": auc, "gini": 2 * auc - 1},
        "guardrail": gr,
        "warnings": [],
        "retry_history": [],
        "target": "creditability",
        # 默认全数值特征，保证 KS<0.2 时能走到 chimerge 分支
        "train_df": pd.DataFrame({
            "f1": np.random.default_rng(11).normal(size=120),
            "f2": np.random.default_rng(12).normal(size=120),
            "creditability": np.random.default_rng(13).integers(0, 2, 120),
        }),
    }
    st.update(extra)
    return st


def test_rule_high_psi_concentrated_excludes_top_var(monkeypatch):
    """PSI>0.25 且 CSI 集中在单变量 → 剔除该变量并重跑 var_filter。"""
    import agent.nodes as nodes

    monkeypatch.setattr(
        nodes, "_csi_top_variables",
        lambda st: ("debt_ratio", 0.42, 0.05, 0.03),
    )
    st = _state_with(ks=0.5, auc=0.8, psi=0.31)
    diag = _rule_based_diagnosis(st, {"min_ks": 0.3, "min_auc": 0.7})
    assert diag["action"] == "retune"
    assert diag["goto"] == "var_filter"
    assert diag["param_patch"]["exclude_vars"] == ["debt_ratio"]
    assert diag["source"] == "rule"
    assert "CSI" in diag["reason"]


def test_rule_high_psi_diffuse_coarsens_bins(monkeypatch):
    """PSI>0.25 但 CSI 分散 → 粗化到 6 箱。"""
    import agent.nodes as nodes

    monkeypatch.setattr(
        nodes, "_csi_top_variables",
        lambda st: ("x1", 0.12, 0.11, 0.10),   # 最高仅 2× 不到中位数 → 判为分散
    )
    st = _state_with(ks=0.5, auc=0.8, psi=0.31)
    diag = _rule_based_diagnosis(st, {"min_ks": 0.3, "min_auc": 0.7})
    assert diag["goto"] == "woebin"
    assert diag["param_patch"] == {"max_bins": 6}


def test_rule_low_ks_switches_to_chimerge():
    """纯数值特征 + KS<0.2 → 换 chimerge。"""
    st = _state_with(ks=0.15, auc=0.75, psi=0.02)
    st["train_df"] = pd.DataFrame({          # 全数值 → 允许 chimerge
        "f1": np.random.default_rng(0).normal(size=100),
        "f2": np.random.default_rng(1).normal(size=100),
        "creditability": np.random.default_rng(2).integers(0, 2, 100),
    })
    diag = _rule_based_diagnosis(st, {"min_ks": 0.3, "min_auc": 0.7})
    assert diag["goto"] == "woebin"
    assert diag["param_patch"] == {"binning_method": "chimerge"}


def test_rule_low_ks_with_categorical_avoids_chimerge():
    """含类别型特征 → 必须避开 chimerge（否则所有变量分箱失败）。"""
    st = _state_with(ks=0.15, auc=0.75, psi=0.02)
    st["train_df"] = pd.DataFrame({
        "num": np.random.default_rng(0).normal(size=100),
        "cat": np.random.default_rng(1).choice(["A", "B", "C"], size=100),  # object 列
        "creditability": np.random.default_rng(2).integers(0, 2, 100),
    })
    diag = _rule_based_diagnosis(st, {"min_ks": 0.3, "min_auc": 0.7})
    assert diag["goto"] == "woebin"
    assert diag["param_patch"].get("binning_method") != "chimerge"
    assert diag["param_patch"]["max_bins"] == 10


def test_rule_ks_below_min_but_auc_ok_merges_small_bins():
    st = _state_with(ks=0.25, auc=0.80, psi=0.02)
    diag = _rule_based_diagnosis(st, {"min_ks": 0.3, "min_auc": 0.7})
    assert diag["goto"] == "woebin"
    assert diag["param_patch"] == {"min_bin_size": 0.08}


def test_rule_low_auc_enables_lgbm_once():
    st = _state_with(ks=0.35, auc=0.62, psi=0.02)
    diag = _rule_based_diagnosis(st, {"min_ks": 0.3, "min_auc": 0.7})
    assert diag["goto"] == "model_train"
    assert diag["param_patch"] == {"use_lgbm": True}
    # 已开 LGBM → 不再重复该方案
    st2 = _state_with(ks=0.35, auc=0.62, psi=0.02)
    diag2 = _rule_based_diagnosis(st2, {"min_ks": 0.3, "min_auc": 0.7, "use_lgbm": True})
    assert diag2["action"] == "escalate"


def test_rule_skips_already_tried_plan():
    tried = {
        "round": 1, "action": "retune", "goto": "woebin",
        "param_patch": {"min_bin_size": 0.08},
    }
    st = _state_with(ks=0.25, auc=0.80, psi=0.02)
    st["retry_history"] = [tried]
    diag = _rule_based_diagnosis(st, {"min_ks": 0.3, "min_auc": 0.7})
    # 该方案已试过 → 应跳过；此场景无其它可用方案 → escalate
    assert diag["action"] == "escalate"
    assert diag["source"] == "rule"


def test_rule_non_actionable_single_class_fold():
    st = _state_with(ks=0.5, auc=0.8, psi=0.02)
    st["warnings"] = ["[WARN] fold 2024: This solver needs samples of at least 2 classes (单类别)"]
    diag = _rule_based_diagnosis(st, {"min_ks": 0.3, "min_auc": 0.7})
    assert diag["action"] == "escalate"
    assert diag.get("non_actionable") is True
    assert "单类别" in diag["reason"]


def test_rule_fallback_escalates_when_all_pass():
    """指标全部达标（不可能出现在 diagnose 里，但兜底路径要稳定）→ escalate。"""
    st = _state_with(ks=0.55, auc=0.85, psi=0.02)
    diag = _rule_based_diagnosis(st, {"min_ks": 0.3, "min_auc": 0.7})
    assert diag["action"] == "escalate"
    assert diag["goto"] == "hitl_review"


def test_plan_signature_stable():
    assert _plan_signature("woebin", {"a": 1, "b": 2}) == _plan_signature("woebin", {"b": 2, "a": 1})
    assert _plan_signature("woebin", {"a": 1}) != _plan_signature("var_filter", {"a": 1})


# ---------------------------------------------------------------------------
# C. diagnose_node：降级 + 计数 + config 合并
# ---------------------------------------------------------------------------
def test_diagnose_node_falls_back_to_rule_when_llm_dead(no_llm):
    st = _state_with(ks=0.15, auc=0.75, psi=0.02)
    cmd = diagnose_node(st)
    assert isinstance(cmd, Command)
    update = cmd.update
    diag = update["diagnosis"]
    assert diag["source"] == "rule"
    assert diag["action"] == "retune"
    assert diag["goto"] == "woebin"
    assert update["retry_count"] == 1
    assert len(update["retry_history"]) == 1
    assert any("已回退规则诊断" in w for w in update["warnings"])


def test_diagnose_node_merges_patch_into_config(no_llm):
    st = _state_with(ks=0.15, auc=0.75, psi=0.02)
    st["config_overrides"] = {"min_ks": 0.3, "exclude_vars": ["old_var"]}
    cmd = diagnose_node(st)
    cfg = cmd.update["config_overrides"]
    assert cfg["binning_method"] == "chimerge"
    assert cfg["exclude_vars"] == ["old_var"]


def test_diagnose_node_respects_max_retry(no_llm):
    st = _state_with(ks=0.15, auc=0.75, psi=0.02)
    st["config_overrides"] = {"max_retry": 2}
    st["retry_count"] = 2
    cmd = diagnose_node(st)
    diag = cmd.update["diagnosis"]
    assert diag["action"] == "escalate"
    assert diag["goto"] == "hitl_review"
    assert diag["source"] == "guard"
    # 超限不再自增
    assert cmd.update["retry_count"] == 2


def test_diagnose_node_disable_llm_flag(no_llm):
    st = _state_with(ks=0.15, auc=0.75, psi=0.02)
    st["config_overrides"] = {"allow_llm_diagnosis": False}
    cmd = diagnose_node(st)
    assert cmd.update["diagnosis"]["source"] == "rule"
    assert any("确定性规则诊断" in w for w in cmd.update["warnings"])


def test_diagnose_node_rejects_duplicate_llm_plan(monkeypatch):
    """LLM 重复给出试过的方案 → 被拦截，改由规则阶梯挑新方案。"""
    import agent.nodes as nodes

    monkeypatch.setattr(
        nodes, "_llm_json",
        lambda *a, **k: {
            "action": "retune", "goto": "var_filter",
            "param_patch": {"iv_threshold": 0.05}, "reason": "再试一次",
        },
    )
    st = _state_with(ks=0.15, auc=0.75, psi=0.02)
    st["retry_history"] = [{
        "round": 1, "action": "retune", "goto": "var_filter",
        "param_patch": {"iv_threshold": 0.05},
    }]
    cmd = diagnose_node(st)
    diag = cmd.update["diagnosis"]
    assert diag["source"] == "rule", "重复方案应被拦截并降级到规则"
    assert diag["param_patch"] != {"iv_threshold": 0.05}
    assert any("重复已试过" in w for w in cmd.update["warnings"])


def test_var_filter_auto_relax_when_too_few_vars():
    """iv_threshold 过严导致变量被剔光 → 必须自动兜底保住 ≥ MIN_KEPT_VARS 个变量。"""
    from agent.nodes import MIN_KEPT_VARS, _var_filter_with_relax

    rng = np.random.default_rng(3)
    n = 200
    df = pd.DataFrame({
        "target": rng.integers(0, 2, n),
        "weak1": rng.normal(size=n),      # 弱变量，IV≈0
        "weak2": rng.normal(size=n),
        "strong1": rng.normal(size=n) + np.arange(n) % 2 * 0.0,
    })
    df["bad_ratio"] = np.where(df.index % 2 == 0, 0.5, 0.2)

    out, _ = _var_filter_with_relax(df, "target", iv_t=0.5, missing_t=0.5, identical_t=0.95)
    kept = [c for c in out.columns if c != "target"]
    assert len(kept) >= min(MIN_KEPT_VARS, df.shape[1] - 1), f"变量被剔光: {kept}"
    assert "target" in out.columns


def test_woebin_node_falls_back_to_tree(monkeypatch):
    """分箱算法不适配（返回空 bins）→ 自动用 tree 重试，不允许把空 bins 传下去。"""
    import agent.nodes as nodes

    calls: list[str] = []

    def fake_woebin(df, target=None, method="tree", **kw):
        calls.append(method)
        return {} if method == "chimerge" else {"v1": pd.DataFrame({"bin": ["[-inf,inf)"]})}

    monkeypatch.setattr(nodes, "woebin", fake_woebin)
    st = {"train_df": pd.DataFrame({"a": [1, 2, 3]}), "target": "y",
          "config_overrides": {"binning_method": "chimerge"}}
    out = nodes.woebin_node(st)
    assert calls == ["chimerge", "tree"], f"应先按配置再回退 tree，实际 {calls}"
    assert out["bins"], "回退后必须拿到非空 bins"


def test_woebin_node_raises_when_all_methods_fail(monkeypatch):
    import agent.nodes as nodes

    monkeypatch.setattr(nodes, "woebin", lambda *a, **k: {})
    st = {"train_df": pd.DataFrame({"a": [1, 2, 3]}), "target": "y", "config_overrides": {}}
    out = nodes.woebin_node(st)
    # @safe 会把异常转成 errors；这里断言没有把空 bins 放行下去
    assert not out.get("bins")
    assert any("woebin" in e for e in out.get("errors", []))


def test_merge_config_exclude_vars_union():
    cfg = {"exclude_vars": ["a"], "max_bins": 8}
    out = _merge_config(cfg, {"exclude_vars": ["b", "a"], "max_bins": 6})
    assert out["exclude_vars"] == ["a", "b"]
    assert out["max_bins"] == 6
    # 不污染原 dict
    assert cfg["max_bins"] == 8


# ---------------------------------------------------------------------------
# D. route_after_diagnose
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("goto", ["preprocess", "var_filter", "woebin", "model_train", "build_scorecard"])
def test_route_after_diagnose_allowed(goto):
    assert route_after_diagnose({"diagnosis": {"goto": goto}}) == goto


def test_route_after_diagnose_invalid_or_missing():
    assert route_after_diagnose({"diagnosis": {"goto": "format_c_drive"}}) == "hitl"
    assert route_after_diagnose({"diagnosis": {}}) == "hitl"
    assert route_after_diagnose({}) == "hitl"


# ---------------------------------------------------------------------------
# E. 端到端：自动回跳 + 超限转人工
# ---------------------------------------------------------------------------
def test_graph_auto_retry_then_hitl(csv_path, db_path, fresh_thread, no_llm):
    """min_ks/min_auc 设到不可能达到 → 必须自动回跳 ≥1 次，最终停在 hitl_review。"""
    from agent.graph import compile_with_sqlite

    app, conn = compile_with_sqlite(db_path)
    try:
        cfg = {"configurable": {"thread_id": fresh_thread}}
        result = app.invoke(
            {
                "csv_path": csv_path,
                "target": "creditability",
                "config_overrides": {"min_ks": 0.99, "min_auc": 0.99, "max_retry": 3},
            },
            config=cfg,
        )
        hist = result.get("retry_history") or []
        assert len(hist) >= 1, f"至少应自动回跳 1 次，实际 {len(hist)} 次"
        assert result.get("retry_count", 0) >= 1
        # 至少一条 DIAG 记录写进了 step_history
        assert any(h.startswith("[DIAG#") for h in result.get("step_history", []))
        # 每跳一次都应回跳到具体节点，而不是直接放弃
        first = hist[0]
        assert first["goto"] in {
            "preprocess", "var_filter", "woebin", "model_train", "build_scorecard",
        }, f"首次回跳目标非法: {first}"
        print(f"\n[test_graph_auto_retry_then_hitl] retry_history={hist}")

        snap = app.get_state(cfg)
        assert snap.next and any("hitl" in str(n) for n in snap.next), f"最终应停在 hitl_review: {snap.next}"
    finally:
        conn.close()


def test_graph_max_retry_cap_1(csv_path, db_path, fresh_thread, no_llm):
    """max_retry=1：只允许回跳一次，第二次必须转人工。"""
    from agent.graph import compile_with_sqlite

    app, conn = compile_with_sqlite(db_path)
    try:
        cfg = {"configurable": {"thread_id": fresh_thread}}
        result = app.invoke(
            {
                "csv_path": csv_path,
                "target": "creditability",
                "config_overrides": {"min_ks": 0.99, "min_auc": 0.99, "max_retry": 1},
            },
            config=cfg,
        )
        assert result.get("retry_count", 0) == 1, f"retry_count 应为 1，实际 {result.get('retry_count')}"
        snap = app.get_state(cfg)
        assert snap.next and any("hitl" in str(n) for n in snap.next)
        print(f"\n[test_graph_max_retry_cap_1] retry_count={result['retry_count']} next={snap.next}")
    finally:
        conn.close()


def test_graph_pass_goes_to_end_without_diagnose(csv_path, db_path, fresh_thread, no_llm):
    """放宽护栏到极易通过 → 应直接 END，不进 diagnose、不触发 HITL。"""
    from agent.graph import compile_with_sqlite

    app, conn = compile_with_sqlite(db_path)
    try:
        cfg = {"configurable": {"thread_id": fresh_thread}}
        result = app.invoke(
            {
                "csv_path": csv_path,
                "target": "creditability",
                "config_overrides": {
                    "min_ks": 0.05, "min_auc": 0.5, "max_psi": 0.5,
                    "require_human_review": False, "max_retry": 3,
                },
            },
            config=cfg,
        )
        assert result.get("retry_count", 0) == 0, "通过场景不应产生重试"
        assert not (result.get("retry_history") or []), "通过场景不应有 retry_history"
        snap = app.get_state(cfg)
        assert not snap.next, f"应直接走到 END，实际 next={snap.next}"
        print("\n[test_graph_pass_goes_to_end] 流程直接结束，无重试")
    finally:
        conn.close()
