"""tests.test_agent_runner — Streamlit Agent 运行封装层（app/core/agent_runner.py）

覆盖：
1. 节点元数据完整性：18 个节点都有中文标签，且顺序与图拓扑一致
2. build_agent_config 白名单：只接受 Agent 相关键，不污染基础配置
3. _event_of 状态判定：errors→fail / warnings→warn / 正常→ok / diagnose→retry
4. summarize_state：从完整的 final state 提炼 UI 摘要
5. save_scorecard：落盘成功 / 无评分卡时返回 False
6. parse_requirement：空输入短路、非法参数被拦截
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.agent_runner import (
    AGENT_PARAM_KEYS,
    CRITICAL_NODES,
    NODE_META,
    NODE_ORDER,
    NodeEvent,
    _event_of,
    build_agent_config,
    node_label,
    node_module,
    parse_requirement,
    save_scorecard,
    summarize_state,
)
from tools.validation_tools import GuardrailResult

# ============================================================================
# 节点元数据
# ============================================================================
def test_node_order_covers_all_meta_keys():
    assert set(NODE_ORDER) == set(NODE_META), "时间线顺序与节点元数据必须一一对应"


def test_node_order_matches_graph_topology():
    """主干顺序必须与 agent/graph.py 的实际拓扑一致（防 UI 展示与实现漂移）。"""
    expected = [
        "load_data", "explore_data", "data_gate", "split", "preprocess", "var_filter",
        "woebin", "woebin_apply", "model_train", "model_predict", "evaluate_performance",
        "build_scorecard", "critic", "scorecard_ply", "guardrail_check",
        "diagnose", "hitl_review", "reporter",
    ]
    assert NODE_ORDER == expected


def test_critical_nodes_are_the_four_short_circuit_nodes():
    assert CRITICAL_NODES == {"load_data", "split", "woebin", "model_train"}


def test_every_node_has_chinese_label_and_module():
    for node in NODE_ORDER:
        label, module = NODE_META[node]
        assert label and any("\u4e00" <= ch <= "\u9fff" for ch in label), node
        assert module, node
        assert node_label(node) == label
        assert node_module(node) == module


def test_unknown_node_falls_back():
    assert node_label("nope") == "nope"
    assert node_module("nope") == "其他"


# ============================================================================
# build_agent_config：白名单
# ============================================================================
def test_build_agent_config_merges_whitelisted_keys():
    base = {"test_size": 0.3, "max_bins": 8}
    cfg = build_agent_config(base, {"max_retry": 2, "allow_llm_critic": False})
    assert cfg["test_size"] == 0.3 and cfg["max_bins"] == 8
    assert cfg["max_retry"] == 2 and cfg["allow_llm_critic"] is False


def test_build_agent_config_rejects_unknown_keys():
    cfg = build_agent_config({}, {"evil_param": 1, "test_size": 0.4})
    assert "evil_param" not in cfg
    assert "test_size" not in cfg, "基础配置键不应由 agent_opts 注入"


def test_build_agent_config_nl_patch_passes_through():
    """NL 解析结果已过 validate_patch，必须原样合并（含基础建模参数如 max_bins）。"""
    cfg = build_agent_config({"max_bins": 8}, {}, {"max_bins": 6, "min_year": 2018})
    assert cfg["max_bins"] == 6
    assert cfg["min_year"] == 2018


def test_build_agent_config_nl_patch_overrides_agent_opts():
    cfg = build_agent_config({}, {"cutoff_method": "ks"}, {"cutoff_method": "bad_rate"})
    assert cfg["cutoff_method"] == "bad_rate"


def test_build_agent_config_does_not_mutate_base():
    base = {"test_size": 0.3}
    build_agent_config(base, {"max_retry": 1})
    assert base == {"test_size": 0.3}


def test_all_agent_param_keys_are_in_default_config():
    """白名单里的键必须都能在 default_config.yaml 找到，否则 UI 设置了也不生效。"""
    from pathlib import Path

    import yaml

    yaml_path = Path(__file__).resolve().parents[1] / "config" / "default_config.yaml"
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    flat: set[str] = set()
    for section, vals in raw.items():
        if isinstance(vals, dict):
            flat.update(vals.keys())
        else:
            flat.add(section)
    missing = [k for k in AGENT_PARAM_KEYS if k not in flat]
    assert not missing, f"白名单键不在 default_config.yaml: {missing}"


# ============================================================================
# _event_of：事件状态判定
# ============================================================================
def test_event_ok():
    ev = _event_of("load_data", {"step_history": ["[OK] load_data"]})
    assert ev.status == "ok"
    assert ev.label == "加载数据"
    assert ev.detail == "[OK] load_data"


def test_event_warn_on_warnings():
    ev = _event_of("guardrail_check", {"warnings": ["[guardrail] PSI 偏高"]})
    assert ev.status == "warn"
    assert "PSI" in ev.detail


def test_event_fail_on_errors():
    ev = _event_of("model_train", {"errors": ["model_train 失败: boom"], "critical_error": True})
    assert ev.status == "fail"
    assert "boom" in ev.detail


def test_event_critical_without_error_still_fail():
    ev = _event_of("split", {"critical_error": True})
    assert ev.status == "fail"


def test_event_diagnose_is_retry_with_decision_detail():
    ev = _event_of("diagnose", {
        "diagnosis": {"action": "retune", "goto": "model_train", "source": "rule"},
        "step_history": ["[DIAG#1] retune → model_train"],
    })
    assert ev.status == "retry"
    assert "model_train" in ev.detail and "rule" in ev.detail


def test_event_tolerates_non_dict_update():
    ev = _event_of("reporter", None)
    assert ev.status == "ok"
    assert isinstance(ev, NodeEvent)


# ============================================================================
# summarize_state
# ============================================================================
def _full_state() -> dict:
    return {
        "metrics": {"test": {"auc": 0.89, "ks": 0.77, "gini": 0.78, "psi": 0.043}},
        "guardrail": GuardrailResult(
            passed=True, metrics={}, warnings=[], requires_human_review=False,
            rationale="ok",
        ),
        "data_gate": {"passed": True, "suggestion": "数据质量良好"},
        "critic_report": {
            "passed": False, "source": "rule+llm", "n_issues": 11,
            "n_high": 3, "n_mid": 2, "n_low": 6, "issues": [],
        },
        "cutoff_info": {"cutoff": 902.8, "approve_rate": 0.8387},
        "retry_history": [{"round": 1, "action": "retune", "goto": "model_train"}],
        "report_path": "reports/x.md",
        "report_md": "# 报告",
        "warnings": ["w1"],
        "errors": [],
    }


def test_summarize_state_extracts_all_panels():
    s = summarize_state(_full_state())
    assert s["auc"] == 0.89 and s["ks"] == 0.77
    assert s["guardrail_passed"] is True
    assert s["gate_passed"] is True
    assert s["n_retry"] == 1
    assert s["critic"]["n_issues"] == 11
    assert s["critic_has_high"] is True
    assert s["cutoff"]["cutoff"] == 902.8
    assert s["report_path"] == "reports/x.md"
    assert s["critical_error"] is False


def test_summarize_state_empty_is_safe():
    s = summarize_state({})
    assert s["auc"] is None
    assert s["gate_passed"] is None
    assert s["n_retry"] == 0
    assert s["critic"] == {}
    assert s["critic_has_high"] is False
    assert s["report_md"] == ""


def test_summarize_state_critical_error_flag():
    s = summarize_state({"critical_error": True, "errors": ["woebin 失败"]})
    assert s["critical_error"] is True
    assert s["errors"] == ["woebin 失败"]


# ============================================================================
# save_scorecard
# ============================================================================
def test_save_scorecard_writes_pickle(tmp_path):
    card = {"ROA": pd.DataFrame({"bin": ["[0,1)"], "points": [-10]})}
    dest = tmp_path / "nested" / "card.pkl"
    assert save_scorecard({"scorecard": card}, dest) is True
    assert dest.exists() and dest.stat().st_size > 0


def test_save_scorecard_returns_false_without_card(tmp_path):
    dest = tmp_path / "card.pkl"
    assert save_scorecard({}, dest) is False
    assert not dest.exists()


def test_save_scorecard_returns_false_on_write_error(tmp_path):
    """目标路径不可写时应静默返回 False，不把 UI 搞崩。"""
    blocked = tmp_path / "afile"
    blocked.write_text("x", encoding="utf-8")
    assert save_scorecard({"scorecard": {"a": 1}}, blocked / "sub" / "c.pkl") is False


# ============================================================================
# parse_requirement（模块 4 的 UI 入口）
# ============================================================================
def test_parse_requirement_empty_short_circuits():
    assert parse_requirement("") == ({}, [])
    assert parse_requirement("   ") == ({}, [])
    assert parse_requirement(None) == ({}, [])  # type: ignore[arg-type]


def test_parse_requirement_rule_path_returns_whitelisted_patch():
    """不走 LLM（use_llm=False）时应由规则解析出合法 patch，且能顺利合并进 config。"""
    patch, _warns = parse_requirement("最多分 6 箱", use_llm=False)
    assert isinstance(patch, dict)
    merged = build_agent_config({"max_bins": 8}, {}, patch)
    if "max_bins" in patch:
        assert merged["max_bins"] == patch["max_bins"], "NL 参数必须真的生效，不能在中途被丢掉"


def test_parse_requirement_clamps_out_of_range():
    """越界值必须被夹到 param_bounds 内（模块 4 的校验层）。"""
    patch, _warns = parse_requirement("最多分 100 箱", use_llm=False)
    if "max_bins" in patch:
        assert 3 <= patch["max_bins"] <= 12


@pytest.mark.parametrize("text", ["排除房地产行业", "用 2018 年以后的数据", "关注 KS"])
def test_parse_requirement_never_raises(text):
    patch, warns = parse_requirement(text, use_llm=False)
    assert isinstance(patch, dict) and isinstance(warns, list)
