"""tests.test_nl_config — 自然语言配置解析与校验（模块 4）

覆盖：
1. 高频中文表达 → 正确键（正常路径）
2. 越界值被夹取并 warning
3. 未知键被丢弃并 warning
4. 枚举值非法被丢弃
5. LLM 不可用时的降级路径（use_llm=False 强制走规则）
6. 空输入 / 无意义输入
"""
from __future__ import annotations

import pytest

from agent.nl_config import (
    ALL_KEYS,
    parse_nl_config,
    validate_patch,
)


# ============================================================================
# validate_patch：纯校验层
# ============================================================================
def test_valid_patch_passes_through():
    patch, warns = validate_patch({"max_bins": 6, "time_split": True, "min_ks": 0.4})
    assert patch == {"max_bins": 6, "time_split": True, "min_ks": 0.4}
    assert warns == []


def test_unknown_key_dropped():
    patch, warns = validate_patch({"unknown_key": 1, "max_bins": 6})
    assert patch == {"max_bins": 6}
    assert any("未知配置项" in w for w in warns)


def test_out_of_range_clamped():
    patch, warns = validate_patch({"min_ks": 0.99})
    assert patch["min_ks"] == 0.6  # 夹到上界
    assert any("夹取" in w for w in warns)


def test_below_range_clamped():
    patch, warns = validate_patch({"max_bins": 1})
    assert patch["max_bins"] == 3  # 夹到下界
    assert any("夹取" in w for w in warns)


def test_invalid_choice_dropped():
    patch, warns = validate_patch({"binning_method": "magic"})
    assert patch == {}
    assert any("不在合法选项" in w for w in warns)


def test_valid_choice_kept():
    patch, warns = validate_patch({"binning_method": "chimerge"})
    assert patch == {"binning_method": "chimerge"}
    assert warns == []


def test_bool_string_parsed():
    assert validate_patch({"time_split": "true"})[0] == {"time_split": True}
    assert validate_patch({"time_split": "false"})[0] == {"time_split": False}
    assert validate_patch({"use_lgbm": 1})[0] == {"use_lgbm": True}
    assert validate_patch({"use_lgbm": 0})[0] == {"use_lgbm": False}


def test_numeric_string_converted():
    patch, _ = validate_patch({"max_bins": "6"})
    assert patch["max_bins"] == 6
    assert isinstance(patch["max_bins"], int)


def test_non_dict_patch():
    patch, warns = validate_patch("not a dict")  # type: ignore[arg-type]
    assert patch == {}
    assert any("不是 dict" in w for w in warns)


# ============================================================================
# parse_nl_config：规则解析（LLM 降级路径）
# ============================================================================
def test_nl_year_bins_timesplit():
    """验收标准 1：能解析出三个正确键。"""
    patch, warns = parse_nl_config("用2018年以后数据、最多分6箱、开启时序验证", use_llm=False)
    assert patch.get("min_year") == 2018
    assert patch.get("max_bins") == 6
    assert patch.get("time_split") is True


def test_nl_out_of_range_rejected():
    """验收标准 2：越界项被拒绝（夹取）并给出 warning。"""
    patch, warns = parse_nl_config("把正则化调成 0.5 并把 KS 调到 0.99", use_llm=False)
    assert patch.get("regularization") == 0.5
    assert patch.get("min_ks") == 0.6  # 0.99 被夹到 0.6
    assert any("夹取" in w and "min_ks" in w for w in warns)


def test_nl_exclude_industry():
    patch, _ = parse_nl_config("排除房地产行业", use_llm=False)
    assert patch.get("exclude_industries") == ["K70"]


def test_nl_lgbm_on():
    patch, _ = parse_nl_config("开启 LGBM 对照模型", use_llm=False)
    assert patch.get("use_lgbm") is True


def test_nl_ks_auc_psi():
    patch, _ = parse_nl_config("KS 目标 0.4，AUC 0.75，PSI 0.2", use_llm=False)
    assert patch.get("min_ks") == 0.4
    assert patch.get("min_auc") == 0.75
    assert patch.get("max_psi") == 0.2


def test_nl_test_size_and_iv():
    patch, _ = parse_nl_config("测试集比例设为 0.25，IV 阈值 0.05", use_llm=False)
    assert patch.get("test_size") == 0.25
    assert patch.get("iv_threshold") == 0.05


def test_nl_empty_input():
    patch, warns = parse_nl_config("", use_llm=False)
    assert patch == {}
    assert warns == []


def test_nl_no_match():
    patch, warns = parse_nl_config("今天天气不错", use_llm=False)
    assert patch == {}
    assert any("未能从需求中解析" in w for w in warns)


def test_nl_all_keys_are_whitelisted():
    """解析出的所有键都必须在白名单内（防 LLM 幻觉键）。"""
    for text in [
        "用2020年以后数据、最多分8箱、KS 0.35、开启时序验证",
        "排除银行业，测试集 0.2，正则化 0.05",
    ]:
        patch, _ = parse_nl_config(text, use_llm=False)
        assert set(patch.keys()).issubset(ALL_KEYS)


def test_nl_llm_fallback_warns(monkeypatch):
    """LLM 不可用（无 key / 抛异常）时必须降级到规则解析，且给出 warning。"""
    import app.llm.client as llm_client

    def boom(*args, **kwargs):
        raise RuntimeError("no api key")

    # mock 底层 get_model，让 _llm_json 内部的 try/except 捕获并返回 None
    monkeypatch.setattr(llm_client, "get_model", boom)
    patch, warns = parse_nl_config("用2019年以后数据", use_llm=True)
    # 降级后规则解析仍然生效
    assert patch.get("min_year") == 2019
    assert any("LLM" in w for w in warns)
