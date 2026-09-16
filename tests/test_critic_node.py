"""tests.test_critic_node — Critic 节点：规则审计汇总 + LLM 复核降级（模块 3）

覆盖：
1. 三项检查各自产生 findings（VIF / 系数符号 / 集中度）
2. 严重度排序与计数、passed 判定
3. LLM 不可用 → source="rule"；LLM 返回清单外 item 或含数字建议 → 被拦截
4. 数据缺失 / 检查内部异常 → 只记 step_history，不抛错（不阻塞主流程）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agent.nodes import critic_node

RNG = np.random.default_rng(20260915)


class _Model:
    def __init__(self, coefs):
        self.coef_ = np.asarray([coefs])


def _base_state(**override) -> dict:
    """最小可用 state：两项 VIF fail + 一项系数反转 + 一项集中度报警。"""
    x = RNG.normal(size=120)
    X = pd.DataFrame({
        "a_woe": x,
        "b_woe": x * 2.0,              # 与 a 完全共线 → VIF fail
        "c_woe": RNG.normal(size=120),
    })
    state = {
        "target": "y",
        "train_woe": X,
        "model": _Model([1.5, -2.0, 0.4]),
        "bins": {
            "a": pd.DataFrame({"woe": [-1.0, 1.0], "badprob": [0.01, 0.3]}),
            "b": pd.DataFrame({"woe": [-1.0, 1.0], "badprob": [0.01, 0.3]}),
            "c": pd.DataFrame({"woe": [-1.0, 1.0], "badprob": [0.01, 0.3]}),
        },
        "feature_importance": pd.DataFrame({
            "feature": ["b_woe", "a_woe", "c_woe"], "coef": [-2.0, 1.5, 0.4],
            "abs_coef": [2.0, 1.5, 0.4], "rank": [1, 2, 3],
        }),
        "raw_df": pd.DataFrame({"IndustrySector": ["A"] * 90 + ["B"] * 30}),
        "config_overrides": {"allow_llm_critic": False},
    }
    state.update(override)
    return state


# ============================================================================
# 规则路径
# ============================================================================
def test_rule_path_detects_all_three_checks():
    out = critic_node(_base_state())
    rep = out["critic_report"]
    assert rep["source"] == "rule"
    checks = {i["check"] for i in rep["issues"]}
    assert {"vif", "coef_sign", "concentration"} == checks
    assert rep["vif_fail"] >= 2
    assert rep["coef_flip"] == 1              # b 的系数为负 → 方向反转
    assert rep["concentration_fail"] == 1


def test_rule_path_severity_ordering_and_counts():
    out = critic_node(_base_state())
    rep = out["critic_report"]
    sev = [i["severity"] for i in rep["issues"]]
    rank = {"high": 3, "mid": 2, "low": 1}
    assert [rank[s] for s in sev] == sorted([rank[s] for s in sev], reverse=True)
    assert rep["n_issues"] == len(rep["issues"])
    assert rep["n_high"] + rep["n_mid"] + rep["n_low"] == rep["n_issues"]
    assert rep["passed"] is False


def test_clean_state_passes():
    X = pd.DataFrame({f"f{i}_woe": RNG.normal(size=200) for i in range(3)})
    state = _base_state(
        train_woe=X,
        model=_Model([1.0, 2.0, 3.0]),
        bins={f"f{i}": pd.DataFrame({"woe": [-1.0, 1.0], "badprob": [0.01, 0.3]}) for i in range(3)},
        feature_importance=pd.DataFrame({
            "feature": ["f0_woe", "f1_woe", "f2_woe"], "coef": [1.0, 2.0, 3.0],
            "abs_coef": [1.0, 2.0, 3.0], "rank": [1, 2, 3],
        }),
        raw_df=pd.DataFrame({"g": ["A"] * 50 + ["B"] * 50}),
    )
    rep = critic_node(state)["critic_report"]
    assert rep["passed"] is True
    assert rep["issues"] == []
    assert rep["n_issues"] == 0
    assert "critic_findings" not in critic_node(state)


def test_high_severity_emits_warning():
    out = critic_node(_base_state())
    assert any("高危问题" in w for w in out.get("warnings", []))


def test_finding_items_are_rendered_in_report():
    """critic_report 必须能被 reporter 直接消费（item/suggestion 文本非空）。"""
    rep = critic_node(_base_state())["critic_report"]
    for it in rep["issues"]:
        assert it["item"] and it["suggestion"]


# ============================================================================
# 降级 / 容错
# ============================================================================
def test_missing_everything_does_not_raise():
    out = critic_node({"target": "y", "config_overrides": {}})
    assert "[OK] critic" in out["step_history"]
    assert out["critic_report"]["source"] == "rule"
    assert out["critic_report"]["issues"] == []


def test_broken_model_is_tolerated():
    state = _base_state(model=object())     # 没有 coef_ → 符号检查失败但整体不崩
    out = critic_node(state)
    assert "[OK] critic" in out["step_history"]
    assert "critic_report" in out


def test_broken_bins_is_tolerated():
    out = critic_node(_base_state(bins={"a": "not-a-dataframe"}))
    assert "critic_report" in out


def test_non_numeric_train_woe_is_tolerated():
    out = critic_node(_base_state(train_woe=pd.DataFrame({"s": ["x"] * 10})))
    assert "critic_report" in out


# ============================================================================
# LLM 复核路径
# ============================================================================
def test_llm_success_switches_source(monkeypatch):
    from agent import nodes

    captured = {}

    def fake_llm_json(prompt, temperature=None):
        captured["prompt"] = prompt
        # 只复核第一条：改 severity + 给中文建议（无数字）
        return {"issues": [{"item": None, "severity": "low", "suggestion": "影响有限，可保留观察"}]}

    base = critic_node(_base_state())["critic_report"]
    first_item = base["issues"][0]["item"]

    def patched(prompt, temperature=None):
        captured["prompt"] = prompt
        return {"issues": [{"item": first_item, "severity": "low", "suggestion": "影响有限，可保留观察"}]}

    monkeypatch.setattr(nodes, "_llm_json", patched)
    state = _base_state()
    state["config_overrides"]["allow_llm_critic"] = True
    rep = critic_node(state)["critic_report"]
    assert rep["source"] == "rule+llm"
    assert rep["issues"][0]["item"] == first_item
    assert rep["issues"][0]["severity"] == "low"
    assert len(rep["issues"]) == 1
    # prompt 里不得塞具体指标数值（防诱导 LLM 复述数字）
    assert isinstance(captured["prompt"], str)


def test_llm_returns_empty_falls_back_to_rule(monkeypatch):
    from agent import nodes

    monkeypatch.setattr(nodes, "_llm_json", lambda *a, **k: {"issues": []})
    state = _base_state()
    state["config_overrides"]["allow_llm_critic"] = True
    rep = critic_node(state)["critic_report"]
    assert rep["source"] == "rule"
    assert rep["n_issues"] > 0


def test_llm_failure_falls_back_to_rule(monkeypatch):
    from agent import nodes

    monkeypatch.setattr(nodes, "_llm_json", lambda *a, **k: None)
    state = _base_state()
    state["config_overrides"]["allow_llm_critic"] = True
    rep = critic_node(state)["critic_report"]
    assert rep["source"] == "rule"


def test_llm_hallucinated_item_is_dropped(monkeypatch):
    from agent import nodes

    def fake(prompt, temperature=None):
        return {"issues": [
            {"item": "某个不存在的变量 X", "severity": "high", "suggestion": "请删除"},
        ]}

    monkeypatch.setattr(nodes, "_llm_json", fake)
    state = _base_state()
    state["config_overrides"]["allow_llm_critic"] = True
    rep = critic_node(state)["critic_report"]
    assert rep["source"] == "rule"          # 全部被丢净 → 回退规则结论
    assert all("X" not in i["item"] or i["item"] in {f["item"] for f in rep["issues"]}
               for i in rep["issues"])


def test_llm_suggestion_with_digits_is_replaced(monkeypatch):
    from agent import nodes

    captured = {}

    def fake(prompt, temperature=None):
        if "snapshot" in captured:
            return None
        # 第一条不动，需要拿到规则项 → 先取一次规则结论
        return {"issues": [{"item": captured["item"], "severity": "high",
                            "suggestion": "VIF 达到 12.5，建议剔除"}]}

    # 先拿规则 item
    rule_item = critic_node(_base_state())["critic_report"]["issues"][0]["item"]
    captured["item"] = rule_item
    monkeypatch.setattr(nodes, "_llm_json", fake)
    state = _base_state()
    state["config_overrides"]["allow_llm_critic"] = True
    rep = critic_node(state)["critic_report"]
    assert rep["source"] == "rule+llm"
    suggestion = rep["issues"][0]["suggestion"]
    assert not any(c.isdigit() for c in suggestion), "含数字的建议必须被规则原文替换"


def test_allow_llm_critic_false_skips_llm(monkeypatch):
    from agent import nodes

    def boom(*a, **k):
        raise AssertionError("不应调用 LLM")

    monkeypatch.setattr(nodes, "_llm_json", boom)
    rep = critic_node(_base_state())["critic_report"]
    assert rep["source"] == "rule"


@pytest.mark.parametrize("bad_key", ["vif_threshold", "concentration_threshold"])
def test_custom_thresholds_respected(bad_key):
    cfg = {"allow_llm_critic": False, bad_key: 1e-9 if bad_key == "vif_threshold" else 0.01}
    rep = critic_node(_base_state(config_overrides=cfg))["critic_report"]
    assert rep["passed"] is False
    assert rep["n_issues"] > 0


# ============================================================================
# item 对齐（LLM 回传整行时的三级降级匹配）
# ============================================================================
def test_match_known_item_exact():
    from agent.nodes import _match_known_item

    items = {"变量 A 存在多重共线性（VIF=12.30）"}
    assert _match_known_item("变量 A 存在多重共线性（VIF=12.30）", items) == (
        "变量 A 存在多重共线性（VIF=12.30）")


def test_match_known_item_strips_index_and_key_prefix():
    """实际踩坑：LLM 把 prompt 里的整行 `[8] item=... | 规则初判=low` 抄回来。"""
    from agent.nodes import _match_known_item

    known = "变量 ROE 系数符号（-1）与 WOE 风险方向（+1）相反"
    items = {known}
    raw = f"[8] item={known} | 规则初判=low"
    assert _match_known_item(raw, items) == known
    assert _match_known_item(f"item={known}", items) == known
    assert _match_known_item(f"{known} | 规则初判=low", items) == known
    assert _match_known_item(f'"{known}"', items) == known


def test_match_known_item_substring_and_ambiguity():
    from agent.nodes import _match_known_item

    items = {"变量 A 存在多重共线性（VIF=12.30）", "变量 A 存在多重共线性（VIF=5.10）"}
    assert _match_known_item("前缀噪声 变量 A 存在多重共线性（VIF=12.30） 后缀", items) == (
        "变量 A 存在多重共线性（VIF=12.30）")
    # 有歧义时取最长匹配，不会抛错
    assert _match_known_item("变量 A 存在多重共线性", items) is not None


def test_match_known_item_rejects_unknown_and_empty():
    from agent.nodes import _match_known_item

    items = {"变量 A 存在多重共线性（VIF=12.30）"}
    assert _match_known_item("完全无关的一段话", items) is None
    assert _match_known_item(None, items) is None
    assert _match_known_item("", items) is None
    assert _match_known_item("  ", items) is None


def test_llm_copying_whole_marker_lines_still_reviewed(monkeypatch):
    """端到端：LLM 按 marker 行回传 → source 应为 rule+llm（而不是退化成 rule）。"""
    from agent import nodes

    base = critic_node(_base_state())["critic_report"]
    target_item = base["issues"][0]["item"]

    def patched(prompt, temperature=None):
        # 模拟小模型「照抄整行」的典型输出
        return {"issues": [{"item": f"[3] item={target_item} | 规则初判=high",
                            "severity": "low", "suggestion": "影响有限，可保留观察"}]}

    monkeypatch.setattr(nodes, "_llm_json", patched)
    state = _base_state()
    state["config_overrides"]["allow_llm_critic"] = True
    rep = critic_node(state)["critic_report"]
    assert rep["source"] == "rule+llm"
    assert rep["issues"][0]["item"] == target_item
    assert rep["issues"][0]["severity"] == "low"


def test_critic_excludes_target_from_concentration():
    """target / id / year 列不得进入集中度审计（否则全是「0 占 97%」噪音）。"""
    state = _base_state(
        target="is_default_next_year",
        raw_df=pd.DataFrame({
            "is_default_next_year": [0] * 194 + [1] * 6,
            "IndustrySector": ["A"] * 100 + ["B"] * 100,
        }),
    )
    state["config_overrides"].update({"id_col": "Symbol", "year_col": "year"})
    rep = critic_node(state)["critic_report"]
    items = [i["item"] for i in rep["issues"]]
    assert not any("is_default_next_year" in s for s in items)
