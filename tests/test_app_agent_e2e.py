"""tests.test_app_agent_e2e — Streamlit 页面里真正跑通 Agent 闭环（慢，默认跳过）

这个文件覆盖的是「UI 接线」而非业务逻辑：
- Agent 模式：点「启动 Agent 闭环」→ 图 streaming 推进 → 结果页签/面板全部渲染
- 守门拦截：小样本被拦时只渲染守门面板，不渲染 7 个页签
- 人工复核：护栏转人审时进入中断态、人审面板出现；提交 approve 后能续跑到报告

运行方式（默认 `pytest` 不带 `-m slow` 会跳过）：
    pytest -m slow tests/test_app_agent_e2e.py

为什么默认跳过：单个用例要真跑一遍 18 节点图（约 30~60 秒），
放进默认套件会把全量测试从 4 分钟拉到 6 分钟以上。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.slow

APP_DIR = Path(__file__).resolve().parents[1] / "app"
TRAIN_PAGE = str(APP_DIR / "pages" / "1_模型训练.py")
TIMEOUT = 3600


def _make_panel(tmp_path: Path, n: int, seed: int = 20260916) -> Path:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Symbol": [f"{i:06d}" for i in range(n)],
        "year": rng.integers(2015, 2024, n),
        "ROA": rng.normal(0.05, 0.05, n),
        "ROE": rng.normal(0.08, 0.08, n),
        "CurrentRatio": rng.normal(1.9, 0.7, n),
        "DebtToAsset": rng.normal(0.45, 0.18, n),
        "ln_TotalAssets": rng.normal(22.0, 1.2, n),
        "GrossMargin": rng.normal(0.28, 0.12, n),
        "IndustrySector": rng.choice(["制造业", "信息技术", "批发零售"], n),
    })
    latent = (
        -1.2 * (df["ROA"] / 0.05)
        + 0.9 * (df["DebtToAsset"] / 0.18)
        - 0.5 * (df["CurrentRatio"] / 0.7)
    )
    df["is_default_next_year"] = (rng.random(n) < 1 / (1 + np.exp(-(latent - 1.5)))).astype(int)
    csv = tmp_path / "panel.csv"
    df.to_csv(csv, index=False, encoding="utf-8-sig")
    return csv


@pytest.fixture()
def patched_paths(tmp_path, monkeypatch):
    import app.core.paths as paths

    csv = _make_panel(tmp_path, n=2500)
    monkeypatch.setattr(paths, "DEFAULT_DATASET_PATH", csv)
    monkeypatch.setattr(paths, "LATEST_SCORECARD_PATH", tmp_path / "card.pkl")
    return {"csv": csv, "card": tmp_path / "card.pkl"}


def _open_page() -> AppTest:
    at = AppTest.from_file(TRAIN_PAGE, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _disable_llm(at: AppTest) -> AppTest:
    """关掉三个 LLM 开关，让 e2e 完全确定性（不依赖网络与 API key）。"""
    labels = {t.label: t for t in at.sidebar.toggle}
    for name in ("LLM 参与诊断决策", "LLM 参与 Critic 复核", "LLM 撰写报告分析段"):
        labels[name].set_value(False)
    return at.run()


def _click(at: AppTest, text: str) -> None:
    next(b for b in at.button if text in b.label).click().run()


def test_agent_mode_full_run_renders_all_modules(patched_paths):
    at = _disable_llm(_open_page())
    _click(at, "启动 Agent 闭环")

    assert not at.exception, [e.value for e in at.exception]
    payload = at.session_state["agent_result"]
    assert payload["status"] == "completed"
    assert not payload["error"]

    s = payload["summary"]
    # 模块 6：守门通过；模型产出；护栏有结论
    assert s["gate_passed"] is True
    assert s["auc"] is not None and s["ks"] is not None
    assert s["guardrail_passed"] is not None
    # 模块 5：cut-off 与报告
    assert s["cutoff"].get("cutoff") is not None
    assert len(s["report_md"]) > 3000
    # 模块 3：Critic 有结论（纯规则，因为 LLM 关了）
    assert s["critic"].get("passed") is not None
    assert s["critic"].get("source") == "rule"
    # 评分卡落盘给实时评分页
    assert s.get("scorecard_saved") is True
    assert patched_paths["card"].exists()

    # 七个结果页签 + 六模块面板都渲染出来
    assert len(at.tabs) == 7
    md = " ".join(m.value for m in at.markdown)
    for panel in ["模块 4", "模块 6", "模块 1 / 2", "模块 3", "模块 5"]:
        assert panel in md, f"缺少面板: {panel}"


def test_agent_mode_gate_block_hides_result_tabs(tmp_path, monkeypatch):
    """只有 800 行 → 被 min_rows=1000 拦住 → 只渲染守门面板，不渲染 7 个页签。"""
    import app.core.paths as paths

    csv = _make_panel(tmp_path, n=800)
    monkeypatch.setattr(paths, "DEFAULT_DATASET_PATH", csv)
    monkeypatch.setattr(paths, "LATEST_SCORECARD_PATH", tmp_path / "card.pkl")

    at = _disable_llm(_open_page())
    _click(at, "启动 Agent 闭环")

    assert not at.exception, [e.value for e in at.exception]
    payload = at.session_state["agent_result"]
    assert payload["summary"]["gate_passed"] is False
    assert len(at.tabs) == 0, "守门拦截时不应渲染结果页签"
    errs = " ".join(e.value for e in at.error)
    assert "数据守门未通过" in errs
    assert "下一步建议" in " ".join(m.value for m in at.markdown)


def test_agent_mode_interrupt_then_resume(patched_paths):
    """护栏阈值抬到不可能达到 → 直接转人工；提交 approve 后能续跑到报告。"""
    at = _open_page()
    labels = {t.label: t for t in at.sidebar.toggle}
    for name in ("LLM 参与诊断决策", "LLM 参与 Critic 复核", "LLM 撰写报告分析段"):
        labels[name].set_value(False)
    for ni in at.sidebar.number_input:
        if ni.label == "最小 KS":
            ni.set_value(0.95)
    max_retry = next(s for s in at.sidebar.slider if s.label.startswith("最大自愈轮数"))
    max_retry.set_value(0)
    at.run()

    _click(at, "启动 Agent 闭环")
    assert at.session_state["agent_result"]["status"] == "interrupted"
    warns = " ".join(w.value for w in at.warning)
    assert "人工复核" in warns

    # 人审面板出现 → 提交 approve
    radio = next(r for r in at.radio if r.label == "决策")
    radio.set_value("approve（采纳）")
    _click(at, "提交决策并继续")

    assert not at.exception, [e.value for e in at.exception]
    payload = at.session_state["agent_result"]
    assert payload["status"] == "completed"
    assert payload["summary"]["hitl_decision"] == "approve"
    assert payload["summary"]["report_md"]
