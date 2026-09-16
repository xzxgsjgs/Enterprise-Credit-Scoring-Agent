"""tests.test_app_pages — Streamlit 页面冒烟测试（streamlit.testing.v1.AppTest）

覆盖：
1. Home 页可渲染
2. 训练页（Agent 模式）可渲染，且六模块对应的控件都存在
3. 切换到快速训练模式不报错，并给出模式差异提示
4. 目标列不存在时提前报错（不进入训练）
5. 顶部状态区能反映「尚未运行」

说明：只做**渲染层**冒烟——不点「启动 Agent 闭环」（那会真的跑几分钟图）。
图本身的行为已由 tests/test_graph.py / test_agent_loop.py 覆盖。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
TRAIN_PAGE = str(APP_DIR / "pages" / "1_模型训练.py")
HOME_PAGE = str(APP_DIR / "Home.py")

TIMEOUT = 120


@pytest.fixture()
def tiny_panel(tmp_path, monkeypatch) -> Path:
    """把默认数据集与评分卡落盘路径都指到临时目录，避免读 28MB 真实面板。"""
    rng = np.random.default_rng(20260916)
    n = 600
    df = pd.DataFrame({
        "Symbol": [f"{i:06d}" for i in range(n)],
        "year": rng.integers(2015, 2024, n),
        "ROA": rng.normal(0.05, 0.06, n),
        "ROE": rng.normal(0.08, 0.09, n),
        "CurrentRatio": rng.normal(1.8, 0.7, n),
        "DebtToAsset": rng.normal(0.45, 0.18, n),
        "ln_TotalAssets": rng.normal(22.0, 1.2, n),
        "IndustrySector": rng.choice(["制造业", "信息技术", "批发零售"], n),
        "is_default_next_year": (rng.random(n) < 0.06).astype(int),
    })
    csv = tmp_path / "tiny_panel.csv"
    df.to_csv(csv, index=False, encoding="utf-8-sig")

    import app.core.paths as paths

    monkeypatch.setattr(paths, "DEFAULT_DATASET_PATH", csv)
    monkeypatch.setattr(paths, "LATEST_SCORECARD_PATH", tmp_path / "card.pkl")
    return csv


def _fresh_page(path: str) -> AppTest:
    at = AppTest.from_file(path, default_timeout=TIMEOUT)
    at.run()
    return at


# ============================================================================
# Home
# ============================================================================
def test_home_page_renders():
    at = _fresh_page(HOME_PAGE)
    assert not at.exception, [e.value for e in at.exception]
    titles = " ".join(t.value for t in at.title)
    assert "信用评分卡" in titles


# ============================================================================
# 训练页：Agent 模式
# ============================================================================
def test_training_page_renders_in_agent_mode(tiny_panel):
    at = _fresh_page(TRAIN_PAGE)
    assert not at.exception, [e.value for e in at.exception]

    # 标题与模式选择
    assert any("模型训练与复核" in t.value for t in at.title)
    mode_radio = at.sidebar.radio[0]
    assert "Agent 闭环" in mode_radio.value

    # 模块 4：自然语言需求输入框
    assert any(k == "nl_text" for k in at.session_state.filtered_state)
    labels = " ".join(m.value for m in at.markdown)
    assert "自然语言建模需求" in labels

    # 数据概览展开区（含行数）
    summaries = " ".join(e.label for e in at.expander)
    assert "数据概览" in summaries


def test_training_page_has_all_agent_setting_sections(tiny_panel):
    at = _fresh_page(TRAIN_PAGE)
    sections = [e.label for e in at.sidebar.expander]
    # 侧边栏必须有 ①~⑩ 十个分区
    for mark in ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩"]:
        assert any(mark in s for s in sections), f"缺少分区 {mark}: {sections}"
    joined = " ".join(sections)
    assert "数据守门阈值" in joined and "Critic 阈值" in joined and "cut-off 与报告" in joined


def test_training_page_shows_not_run_status(tiny_panel):
    at = _fresh_page(TRAIN_PAGE)
    body = " ".join(m.value for m in at.markdown)
    assert "尚未运行" in body


def test_training_page_run_button_label(tiny_panel):
    at = _fresh_page(TRAIN_PAGE)
    buttons = [b.label for b in at.button]
    assert any("启动 Agent 闭环" in b for b in buttons)


# ============================================================================
# 训练页：快速训练模式
# ============================================================================
def test_training_page_switches_to_quick_mode(tiny_panel):
    at = _fresh_page(TRAIN_PAGE)
    at.sidebar.radio[0].set_value("⚡ 快速训练").run()
    assert not at.exception, [e.value for e in at.exception]

    buttons = [b.label for b in at.button]
    assert any("开始快速训练" in b for b in buttons)
    # 时序切分只在快速训练模式出现
    toggles = [t.label for t in at.sidebar.toggle]
    assert any("时序切分" in t for t in toggles)
    # Agent 模式下的自然语言输入框应消失
    assert not any(k == "nl_text" for k in at.session_state.filtered_state)


# ============================================================================
# 目标列校验
# ============================================================================
def _sidebar_input(at: AppTest, label: str):
    """按 label 找侧边栏输入框（比按下标稳妥，分区增减不会误伤）。"""
    for w in at.sidebar.text_input:
        if w.label == label:
            return w
    raise AssertionError(f"未找到侧边栏输入框: {label}（现有: {[w.label for w in at.sidebar.text_input]}）")


def test_training_page_stops_on_missing_target(tiny_panel):
    at = _fresh_page(TRAIN_PAGE)
    _sidebar_input(at, "目标列名").set_value("not_a_real_column").run()
    assert not at.exception, [e.value for e in at.exception]
    errs = " ".join(e.value for e in at.error)
    assert "不在数据中" in errs


# ============================================================================
# 组件层：时间线与面板在空数据下也必须能渲染
# ============================================================================
def test_timeline_helpers_are_pure():
    from app.ui import apply_event, init_timeline, timeline_summary
    from app.core.agent_runner import NodeEvent

    tl = init_timeline()
    assert len(tl) == 18
    assert timeline_summary(tl)["pending"] == 18

    apply_event(tl, NodeEvent(node="data_gate", label="数据守门", module="模块 6", status="ok"))
    apply_event(tl, NodeEvent(node="diagnose", label="诊断", module="模块 1/2",
                              status="retry", detail="retune → model_train"))
    s = timeline_summary(tl)
    assert s["ok"] == 1 and s["retry"] == 1 and s["pending"] == 16
    assert tl["diagnose"]["detail"].startswith("retune")


def test_apply_event_tolerates_dict_and_unknown_node():
    from app.ui import apply_event, init_timeline

    tl = init_timeline()
    apply_event(tl, {"node": "critic", "status": "warn", "detail": "x"})
    assert tl["critic"]["status"] == "warn"
    apply_event(tl, {"status": "ok"})           # 无 node → 原样返回
    assert len(tl) == 18
    apply_event(tl, {"node": "new_node", "status": "ok"})   # 未知节点 → 追加而非报错
    assert tl["new_node"]["status"] == "ok"
