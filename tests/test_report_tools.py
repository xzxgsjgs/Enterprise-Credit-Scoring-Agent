"""tests.test_report_tools — 报告渲染与拒贷理由书（模块 3 Reporter / 模块 5 可解释）

覆盖：
1. render_model_report 空 ctx 不崩溃，章节齐全
2. 全字段 ctx 渲染出真实数值（且数值来自 ctx 而非 LLM）
3. LLM 段落缺失 → 占位符；注入 → 出现在报告里
4. cut-off / 分档 / 重规划轮次章节正确渲染
5. render_reject_letter 负向贡献排序、cutoff 决策建议、缺列降级
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tools.report_tools import (
    DEFAULT_PLACEHOLDER,
    SECTIONS,
    render_model_report,
    render_reject_letter,
)
from tools.validation_tools import GuardrailResult

RNG = np.random.default_rng(20260911)


# ============================================================================
# 最小 ctx
# ============================================================================
def test_empty_ctx_renders_all_sections():
    md = render_model_report({})
    assert isinstance(md, str) and len(md) > 500
    for sec in SECTIONS:
        assert sec in md, f"缺章节: {sec}"


def test_empty_ctx_uses_placeholder_for_analysis():
    md = render_model_report({})
    assert DEFAULT_PLACEHOLDER in md
    assert "N/A" in md  # thread_id / csv_path 缺省


def test_missing_numeric_values_render_as_dash():
    md = render_model_report({
        "metrics": {"test": {"ks": None, "auc": float("nan")}},
    })
    assert "-" in md


# ============================================================================
# 完整 ctx
# ============================================================================
def _full_ctx() -> dict:
    y = np.concatenate([np.zeros(900), np.ones(100)])
    score = np.concatenate([RNG.normal(700, 30, 900), RNG.normal(400, 30, 100)])
    return {
        "thread_id": "test-thread-001",
        "generated_at": "2026-09-11 12:00:00",
        "csv_path": "credit_agent_cache/loop_verify_panel.csv",
        "target": "is_default_next_year",
        "config_overrides": {
            "missing_strategy": "median", "outlier_method": "cap", "outlier_sigma": 5.0,
            "iv_threshold": 0.02, "binning_method": "tree", "max_bins": 8,
            "base_score": 600, "pdo": 20, "base_odds": 50, "use_lgbm": True,
            "min_ks": 0.3, "min_auc": 0.7, "max_psi": 0.1,
            "exclude_vars": ["Symbol", "ShortName"],
        },
        "data_gate": {
            "passed": True,
            "applied_filters": ["剔除单类别列 EndDate"],
            "warnings": ["年份 2024 样本量偏少"],
            "stats": {"n_rows": 1136, "n_cols": 53, "n_default": 35, "default_rate": 0.0308},
        },
        "eda": {"stats": {"n_rows": 1136}},
        "metrics": {
            "train": {"ks": 0.71, "auc": 0.93, "gini": 0.86},
            "test": {"ks": 0.64, "auc": 0.90, "gini": 0.80, "psi": 0.05},
        },
        "cv_per_fold": pd.DataFrame({
            "year": [2021, 2022, 2023],
            "auc": [0.88, 0.90, 0.91],
            "ks": [0.61, 0.64, 0.66],
        }),
        "cv_mean": {"auc": 0.897, "ks": 0.637, "gini": 0.794, "n_folds": 3},
        "feature_importance": pd.DataFrame({
            "feature": ["ROA_3y_avg", "ROE", "CurrentRatio"],
            "coef": [-1.2, -0.8, -0.4],
            "abs_coef": [1.2, 0.8, 0.4],
            "rank": [1, 2, 3],
        }),
        "scorecard": {"ROA_3y_avg": {"bins": 3}, "ROE": {"bins": 3}, "CurrentRatio": {"bins": 2}},
        "guardrail": GuardrailResult(
            passed=False,
            metrics={"ks": 0.64, "auc": 0.90, "psi": 0.05},
            warnings=["PSI 接近阈值上限"],
            requires_human_review=True,
            rationale="KS/AUC 达标但 PSI 偏高，建议人工复核",
        ),
        "critic_report": {
            "passed": False, "source": "rule",
            "issues": [{"severity": "high", "item": "样本集中度",
                        "suggestion": "扩充行业覆盖"}],
        },
        "retry_history": [
            {"round": 1, "action": "retune", "goto": "model_train",
             "param_patch": {"regularization": 0.1}, "source": "rule",
             "ks": 0.60, "auc": 0.88},
        ],
        "cutoff": {
            "method": "ks", "cutoff": 552.3, "approve_rate": 0.9012,
            "bad_rate_after": 0.0123, "ks_at_cutoff": 0.81,
            "capture_rate": 0.88, "lift": 2.5, "base_bad_rate": 0.0308,
            "table": pd.DataFrame({"cutoff": [500.0, 552.3], "ks": [0.6, 0.81]}),
        },
        "grades": pd.DataFrame({
            "score": score,
            "grade": ["A"] * 250 + ["B"] * 250 + ["C"] * 250 + ["D"] * 250,
        }),
        "bins": {"ROA_3y_avg": pd.DataFrame({
            "bin": ["[-inf,0)", "[0,0.05)", "[0.05,inf)"],
            "woe": [-0.9, 0.1, 0.8],
        })},
        "scored_test": pd.DataFrame({"score": score}),
        "analysis": {
            "overview": "本模型面向 t+1 年违约风险排序。",
            "performance": "测试集表现与训练集接近，未见明显过拟合。",
            "risk": "需持续监控 PSI。",
        },
    }


def test_full_ctx_renders_metrics_and_env():
    md = render_model_report(_full_ctx())
    assert "test-thread-001" in md
    assert "is_default_next_year" in md
    assert "2026-09-11 12:00:00" in md
    # 数值插值（4 位小数）
    assert "0.9000" in md and "0.6400" in md
    assert "3.08%" in md            # 违约率百分比
    assert "90.12%" in md           # 通过率


def test_full_ctx_injects_llm_sections():
    md = render_model_report(_full_ctx())
    assert "本模型面向 t+1 年违约风险排序。" in md
    assert "未见明显过拟合" in md
    assert DEFAULT_PLACEHOLDER not in md


def test_full_ctx_includes_cutoff_and_grades():
    md = render_model_report(_full_ctx())
    assert "552.30" in md                 # cut-off
    assert "2.50×" in md                  # lift
    assert "</details>" in md             # 全网格折叠块
    assert "Critical" not in md
    assert "占用分布" in md or "测试集分档分布" in md   # 分档表
    assert "ROA_3y_avg" in md             # 附录分箱


def test_full_ctx_includes_retry_and_guardrail():
    md = render_model_report(_full_ctx())
    assert "轮" in md
    assert "regularization" in md
    assert "已触发人工复核流程" in md
    assert "样本集中度" in md


# ============================================================================
# 十一、模型复核结论（Critic）
# ============================================================================
def _critic_ctx(report) -> dict:
    ctx = _full_ctx()
    ctx["critic_report"] = report
    return ctx


def test_critic_chapter_exists():
    md = render_model_report(_full_ctx())
    assert "## 十一、模型复核结论（Critic）" in md
    assert "十二、风险与局限" in md and "十三、附录：分箱明细" in md


def test_critic_chapter_renders_counts_and_table():
    report = {
        "passed": False, "source": "rule+llm",
        "n_issues": 3, "n_high": 1, "n_mid": 1, "n_low": 1,
        "vif_fail": 2, "vif_warn": 1, "coef_flip": 1, "concentration_fail": 0,
        "issues": [
            {"check": "vif", "severity": "high",
             "item": "变量 ROA_3y_avg 存在多重共线性（VIF=12.30）", "suggestion": "剔除或降维"},
            {"check": "coef_sign", "severity": "mid",
             "item": "变量 ROE 系数符号（+1）与 WOE 风险方向（-1）相反", "suggestion": "人工复核方向"},
            {"check": "concentration", "severity": "low",
             "item": "列 Province 的取值 广东省 占比 41.2%", "suggestion": "关注外推可靠性"},
        ],
    }
    md = render_model_report(_critic_ctx(report))
    assert "rule+llm" in md
    assert "high=1" in md and "mid=1" in md and "low=1" in md
    assert "另有 1 项 VIF>5" in md                # fail + warn 分开计数
    assert "变量 ROA_3y_avg 存在多重共线性（VIF=12.30）" in md
    assert "处置建议" in md                        # issues 明细表头
    # severity 降序：high 排在最前
    assert md.index("high") < md.index("low")


def test_critic_chapter_missing_shows_fallback():
    ctx = _full_ctx()
    ctx["critic_report"] = None
    md = render_model_report(ctx)
    assert "Critic 复核结果缺失" in md


def test_critic_passed_without_issues():
    md = render_model_report(_critic_ctx(
        {"passed": True, "source": "rule", "n_issues": 0,
         "n_high": 0, "n_mid": 0, "n_low": 0,
         "vif_fail": 0, "coef_flip": 0, "concentration_fail": 0, "issues": []}
    ))
    assert "三项审计均未发现超限项" in md
    assert "通过" in md


def test_partial_ctx_missing_cutoff_and_grades():
    ctx = _full_ctx()
    ctx["cutoff"] = None
    ctx["grades"] = None
    md = render_model_report(ctx)
    assert "cut-off 结果缺失" in md
    assert "分档结果缺失" in md


def test_bins_none_renders_fallback():
    ctx = _full_ctx()
    ctx["bins"] = {}
    md = render_model_report(ctx)
    assert "无分箱数据" in md


# ============================================================================
# render_reject_letter
# ============================================================================
def _score_detail() -> pd.DataFrame:
    return pd.DataFrame([{
        "ROA_3y_avg": 0.01, "ROE": -0.05, "CurrentRatio": 0.8, "FirmAge": 12.0,
        "ROA_3y_avg_points": -40.0, "ROE_points": -30.0, "CurrentRatio_points": -15.0,
        "FirmAge_points": 8.0, "score": 523.0,
    }])


def test_reject_letter_picks_top_negative():
    txt = render_reject_letter(_score_detail(), top_k=2, cutoff=600.0)
    assert "523" in txt
    assert "ROA_3y_avg" in txt and "ROE" in txt
    assert "CurrentRatio" not in txt          # top_k=2 后应被截掉
    assert txt.index("ROA_3y_avg") < txt.index("ROE")   # 按拖累程度排序
    assert "拒绝" in txt                       # 523 < 600 → 拒绝


def test_reject_letter_row_selection_and_approve():
    detail = pd.concat([_score_detail(), pd.DataFrame([{
        "ROA_3y_avg": 0.12, "ROE": 0.2, "CurrentRatio": 2.5, "FirmAge": 20.0,
        "ROA_3y_avg_points": 20.0, "ROE_points": 15.0, "CurrentRatio_points": 10.0,
        "FirmAge_points": 5.0, "score": 650.0,
    }])], ignore_index=True)
    txt = render_reject_letter(detail, row=1, cutoff=600.0)
    assert "650" in txt
    assert "通过" in txt
    assert "主要加分因素" in txt


def test_reject_letter_field_desc_mapping():
    desc = {"ROA_3y_avg": "近三年平均总资产收益率"}
    txt = render_reject_letter(_score_detail(), field_desc=desc)
    assert "近三年平均总资产收益率" in txt
    assert "（当前值 0.010）" in txt           # 原始值回显


def test_reject_letter_empty_input():
    assert "无法生成理由书" in render_reject_letter(pd.DataFrame())


def test_reject_letter_missing_points_columns():
    txt = render_reject_letter(pd.DataFrame([{"score": 500.0}]))
    assert "无法分解各变量贡献" in txt


def test_reject_letter_no_negative_contribution():
    detail = pd.DataFrame([{"A_points": 5.0, "score": 605.0}])
    txt = render_reject_letter(detail)
    assert "未出现明显负向扣分变量" in txt


# ============================================================================
# reporter_node（LangGraph 节点）：降级与落盘
# ============================================================================
def _node_state(tmp_path, **override) -> dict:
    y = np.concatenate([np.zeros(900), np.ones(100)])
    score = np.concatenate([RNG.normal(700, 30, 900), RNG.normal(400, 30, 100)])
    state = {
        "csv_path": "cache/x.csv",
        "target": "is_default_next_year",
        "test_df": pd.DataFrame({"is_default_next_year": y}),
        "scored_test": pd.DataFrame({"score": score}),
        "config_overrides": {
            "report_dir": str(tmp_path),
            "generate_report": True,
            "allow_llm_report": False,   # 离线确定性，避免测试触发真实 LLM
        },
    }
    state.update(override)
    return state


def test_reporter_writes_report_to_disk(tmp_path):
    from agent.nodes import reporter_node

    out = reporter_node(_node_state(tmp_path), {"configurable": {"thread_id": "abc123"}})
    assert "[OK] reporter" in out["step_history"]
    assert out["report_md"]
    path = out["report_path"]
    assert path.endswith("model_report_abc123.md")
    import os

    assert os.path.exists(path)
    assert os.path.getsize(path) > 500


def test_reporter_computes_cutoff_and_grades(tmp_path):
    from agent.nodes import reporter_node

    out = reporter_node(_node_state(tmp_path), {"configurable": {"thread_id": "t2"}})
    assert out["cutoff_info"]["cutoff"] is not None
    assert 400.0 < out["cutoff_info"]["cutoff"] < 700.0
    assert list(out["score_grades"].columns) == ["score", "grade"]
    assert len(out["score_grades"]) == 1000


def test_reporter_survives_missing_scored_test(tmp_path):
    from agent.nodes import reporter_node

    state = _node_state(tmp_path)
    state["scored_test"] = None
    out = reporter_node(state, {"configurable": {"thread_id": "t3"}})
    assert out["report_md"]
    assert "cut-off 结果缺失" in out["report_md"]
    assert "cutoff_info" not in out


def test_reporter_no_write_when_disabled(tmp_path):
    from agent.nodes import reporter_node

    state = _node_state(tmp_path)
    state["config_overrides"]["generate_report"] = False
    out = reporter_node(state, {"configurable": {"thread_id": "t4"}})
    assert out["report_md"]
    assert "report_path" not in out


def test_reporter_llm_sections_parsed(tmp_path, monkeypatch):
    from agent import nodes
    from agent.nodes import reporter_node

    monkeypatch.setattr(
        nodes, "_llm_text",
        lambda *a, **k: "[OVERVIEW] 段落甲\n[PERFORMANCE] 段落乙\n[RISK] 段落丙",
    )
    state = _node_state(tmp_path)
    state["config_overrides"]["allow_llm_report"] = True
    out = reporter_node(state, {"configurable": {"thread_id": "t5"}})
    md = out["report_md"]
    assert "段落甲" in md and "段落乙" in md and "段落丙" in md
    assert DEFAULT_PLACEHOLDER not in md


def test_reporter_llm_failure_falls_back_to_placeholder(tmp_path, monkeypatch):
    from agent import nodes
    from agent.nodes import reporter_node

    monkeypatch.setattr(nodes, "_llm_text", lambda *a, **k: "")
    state = _node_state(tmp_path)
    state["config_overrides"]["allow_llm_report"] = True
    out = reporter_node(state, {"configurable": {"thread_id": "t6"}})
    assert DEFAULT_PLACEHOLDER in out["report_md"]


def test_reporter_uses_unknown_thread_id_without_config(tmp_path):
    from agent.nodes import reporter_node

    out = reporter_node(_node_state(tmp_path))
    assert out["report_path"].endswith("model_report_unknown.md")


def test_reporter_fills_psi_into_metrics(tmp_path):
    from agent.nodes import reporter_node

    state = _node_state(tmp_path, metrics={"test": {"ks": 0.5, "auc": 0.8}})
    state["guardrail"] = GuardrailResult(passed=True, metrics={"ks": 0.5, "auc": 0.8, "psi": 0.0434})
    out = reporter_node(state, {"configurable": {"thread_id": "t7"}})
    assert "0.0434" in out["report_md"]


# ============================================================================
# _parse_tags：LLM 段落解析（容忍噪声 + 数字泄漏拦截）
# ============================================================================
def test_parse_tags_plain():
    from agent.nodes import _parse_tags

    raw = "[OVERVIEW] 甲\n\n[PERFORMANCE] 乙\n\n[RISK] 丙"
    out = _parse_tags(raw)
    assert out == {"overview": "甲", "performance": "乙", "risk": "丙"}


def test_parse_tags_tolerates_markdown_noise():
    from agent.nodes import _parse_tags

    raw = "**[OVERVIEW]**\n甲\n\n## [PERFORMANCE]：\n乙\n\n**[RISK]**：\n丙"
    out = _parse_tags(raw)
    assert out["overview"] == "甲"
    assert out["performance"] == "乙"
    assert out["risk"] == "丙"


def test_parse_tags_handles_reordered_sections():
    from agent.nodes import _parse_tags

    raw = "[PERFORMANCE] 乙\n\n[OVERVIEW] 甲\n\n[RISK] 丙"
    out = _parse_tags(raw)
    assert out == {"overview": "甲", "performance": "乙", "risk": "丙"}


def test_parse_tags_drops_section_with_digits():
    from agent.nodes import _parse_tags

    raw = "[OVERVIEW] 甲\n\n[PERFORMANCE] AUC 达到 0.89，KS 0.77\n\n[RISK] 丙"
    out = _parse_tags(raw)
    assert out["overview"] == "甲"
    assert "performance" not in out      # 含数字 → 整段作废
    assert out["risk"] == "丙"


def test_parse_tags_returns_empty_without_markers():
    from agent.nodes import _parse_tags

    assert _parse_tags("完全没有标记的纯文本段落。") == {}


def test_md_table_formats_floats():
    from tools.report_tools import _md_table

    df = pd.DataFrame({"v": [1.23456789, 2.0], "g": ["A", "B"]})
    out = _md_table(df)
    assert "1.2346" in out and "2.0000" in out
    assert out.count("|") >= 6


def test_feature_importance_is_deterministic():
    from tools import coefficient_importance

    class _M:
        coef_ = [[0.2, -1.5, 0.9]]

    df = coefficient_importance(_M(), ["a", "b", "c"])
    assert list(df["feature"]) == ["b", "c", "a"]      # |coef| 降序
    assert list(df["rank"]) == [1, 2, 3]
    assert df["abs_coef"].is_monotonic_decreasing


def test_feature_importance_fallback_names():
    from tools import coefficient_importance

    class _M:
        coef_ = [[1.0, 2.0]]

    df = coefficient_importance(_M(), ["only_one"])
    # 名称长度不匹配 → 兜底 f0/f1，仍按 |coef| 降序排列
    assert list(df["feature"]) == ["f1", "f0"]
