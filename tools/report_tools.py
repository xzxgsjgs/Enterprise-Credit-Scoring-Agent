"""报告渲染工具集（模块 3 Reporter + 模块 5 可解释输出）。

设计原则（硬约束 1/2/3）：
- **所有数值由 Python 插值**：报告里的 KS/AUC/Gini/PSI/cut-off/分档占比等，
  一律从 ctx 中已算好的 DataFrame/dict 取，函数本身不做任何建模计算。
- **LLM 只写「分析与结论」段落**：由调用方（reporter_node）先调 `render_model_report`
  拿到带占位符的 Markdown，再把占位段落交给 LLM 润色后替换。
- **无 LLM 也能用**：所有章节都有确定性兜底文案，绝不因缺 key 报错。

包含：
    - render_model_report(ctx) -> str        : 13 个固定章节的 Markdown 报告
    - render_reject_letter(score_detail, top_k) -> str : 拒贷理由书（负向贡献 top-k）
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_PLACEHOLDER = "_（LLM 分析段落生成失败，已保留占位文本）_"


# ============================================================================
# 通用小工具
# ============================================================================
def _get(ctx: dict[str, Any], key: str, default: Any = None) -> Any:
    v = ctx.get(key)
    return default if v is None else v


def _fmt_pct(v: Any, digits: int = 2) -> str:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "-"
        return f"{float(v) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "-"


def _fmt_num(v: Any, digits: int = 4) -> str:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "-"
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _md_table(df: pd.DataFrame | None, max_rows: int = 30, float_fmt: str = ".4f") -> str:
    """DataFrame → Markdown 表格（超行自动截断）。"""
    if df is None or (hasattr(df, "empty") and df.empty):
        return "_无数据_"
    try:
        view = df.head(max_rows)
        out = view.to_markdown(index=False, floatfmt=float_fmt)
    except Exception:  # noqa: BLE001 - tabulate 缺失时退化手写表格（并自行格式化浮点）
        cols = list(df.columns)
        head = "| " + " | ".join(str(c) for c in cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        lines = [head, sep]
        for _, r in df.head(max_rows).iterrows():
            cells = []
            for c in cols:
                v = r[c]
                if isinstance(v, float) and pd.notna(v):
                    v = format(v, float_fmt)          # 与 to_markdown 的 floatfmt 保持一致
                cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        out = "\n".join(lines)
    if len(df) > max_rows:
        out += f"\n\n_（共 {len(df)} 行，已显示前 {max_rows} 行）_"
    return out


def _bullets(items: list[str], fallback: str = "_无_") -> str:
    items = [str(i).strip() for i in items if str(i).strip()]
    return "\n".join(f"- {i}" for i in items) if items else fallback


# ============================================================================
# 《模型开发报告》
# ============================================================================
SECTIONS = [
    "一、项目概述", "二、数据说明", "三、标签定义", "四、特征工程",
    "五、模型与参数", "六、性能指标", "七、特征重要性", "八、评分卡刻度与分档",
    "九、cut-off 建议", "十、稳定性与护栏结论", "十一、模型复核结论（Critic）",
    "十二、风险与局限", "十三、附录：分箱明细",
]


def render_model_report(ctx: dict[str, Any]) -> str:
    """渲染《模型开发报告》Markdown。

    Args:
        ctx: 上下文字典，全部键均可缺失（缺的章节显示「无数据」）：
            thread_id, generated_at, csv_path, target, config_overrides,
            eda, data_gate, metrics, cv_per_fold, cv_mean, scorecard,
            feature_importance, guardrail, critic_report, retry_history,
            diagnosis, cutoff, grades, scored_test, bins, analysis(dict)

    Returns:
        Markdown 文本
    """
    cfg = _get(ctx, "config_overrides", {}) or {}
    metrics = _get(ctx, "metrics", {}) or {}
    test_metrics = metrics.get("test", {}) if isinstance(metrics, dict) else {}
    train_metrics = metrics.get("train", {}) if isinstance(metrics, dict) else {}
    eda = _get(ctx, "eda", {}) or {}
    gate = _get(ctx, "data_gate", {}) or {}
    guardrail = _get(ctx, "guardrail")
    analysis = _get(ctx, "analysis", {}) or {}

    lines: list[str] = []
    add = lines.append

    # ---------------- 封面 ----------------
    add("# 企业信用评分卡 —— 模型开发报告")
    add("")
    add(f"- **运行编号 thread_id**：`{_get(ctx, 'thread_id', 'N/A')}`")
    add(f"- **生成时间**：{_get(ctx, 'generated_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}")
    add(f"- **框架**：credit_agent v2（LangGraph 编排 + 确定性工具 + 可降级 LLM）")
    add("")
    add("> 说明：本报告中的全部数值均由 `tools/` 下的确定性函数计算；")
    add("> 标为「分析」的段落由 LLM 撰写，LLM 不可用时回退占位文本。")
    add("")

    # ---------------- 一、项目概述 ----------------
    add("## 一、项目概述")
    add("")
    add(f"本流程对企业**未来一年违约事件**建模，产出可解释的评分卡。")
    add(f"数据文件：`{_get(ctx, 'csv_path', 'N/A')}`；目标列：`{_get(ctx, 'target', 'N/A')}`。")
    add("")
    add(analysis.get("overview", DEFAULT_PLACEHOLDER))
    add("")

    # ---------------- 二、数据说明 ----------------
    add("## 二、数据说明")
    add("")
    stats = gate.get("stats") or eda.get("stats") or {}
    if stats:
        add(f"- 样本行数：**{stats.get('n_rows', '-')}**，列数：**{stats.get('n_cols', '-')}**")
        add(f"- 违约样本：**{stats.get('n_default', '-')}**，违约率：**{_fmt_pct(stats.get('default_rate'))}**")
        if "n_years" in stats:
            add(f"- 年份覆盖：**{stats.get('n_years')}** 年")
    else:
        add("_无数据_")
    add("")
    add("**数据质量守门结果**：")
    add("")
    if gate:
        add(f"- 是否通过：`{'是' if gate.get('passed') else '否'}`")
        add(f"- 已执行过滤：{_bullets(gate.get('applied_filters') or [], fallback='_无_')}")
        if gate.get("warnings"):
            add("- 警告：")
            add(_bullets(gate.get("warnings")))
    else:
        add("_守门结果缺失（可能未启用 data_gate）_")
    add("")

    # ---------------- 三、标签定义 ----------------
    add("## 三、标签定义")
    add("")
    add(f"- 目标列：`{_get(ctx, 'target', 'N/A')}`")
    add("- 取值含义：`1` = t+1 年发生违约（ST/*ST/暂停上市），`0` = 正常")
    add("- 构造方式：按公司 ID 分组，取 t+1 年的困境标签作为 t 年样本的标签（无时间穿越）")
    add("")

    # ---------------- 四、特征工程 ----------------
    add("## 四、特征工程")
    add("")
    add("- 缺失值处理：`" + str(cfg.get("missing_strategy", "median")) + "`")
    add("- 异常值处理：`" + str(cfg.get("outlier_method", "cap")) + f"`，sigma = "
        f"{cfg.get('outlier_sigma', 5.0)}")
    add(f"- IV 筛选阈值：`iv >= {cfg.get('iv_threshold', 0.02)}`、"
        f"缺失率 `<= {cfg.get('missing_threshold', 0.5)}`、"
        f"单一值占比 `<= {cfg.get('identical_threshold', 0.95)}`")
    add(f"- 分箱：`{cfg.get('binning_method', 'tree')}`，"
        f"最多 {cfg.get('max_bins', 8)} 箱，最小箱占比 {cfg.get('min_bin_size', 0.05)}")
    exclude_vars = cfg.get("exclude_vars") or []
    if exclude_vars:
        add(f"- 排除变量：{', '.join(f'`{v}`' for v in exclude_vars)}")
    add("")

    # ---------------- 五、模型与参数 ----------------
    add("## 五、模型与参数")
    add("")
    add("- 主模型：逻辑回归（WOE 编码），`class_weight='balanced'`")
    add(f"- 正则化强度 1/C：`{cfg.get('regularization', 0.01)}`，最大迭代：{cfg.get('max_iter', 1000)}")
    add(f"- 评分卡刻度：base_score={cfg.get('base_score', 600)}，"
        f"PDO={cfg.get('pdo', 20)}，base_odds={cfg.get('base_odds', 50)}")
    if cfg.get("use_lgbm"):
        add("- 对照模型：LightGBM（仅用于能力对照，不参与评分卡刻度）")
    history = _get(ctx, "retry_history") or []
    if history:
        add("")
        add(f"**Agent 自动重规划记录了 {len(history)} 轮**：")
        add("")
        add(_md_table(pd.DataFrame([
            {"轮次": h.get("round"), "动作": h.get("action"), "回跳": h.get("goto"),
             "参数": str(h.get("param_patch")), "来源": h.get("source"),
             "KS": _fmt_num(h.get("ks"), 4), "AUC": _fmt_num(h.get("auc"), 4)}
            for h in history
        ]), max_rows=10))
    add("")

    # ---------------- 六、性能指标 ----------------
    add("## 六、性能指标")
    add("")
    perf_rows = []
    for name, m in (("训练集", train_metrics), ("测试集", test_metrics)):
        if isinstance(m, dict) and m:
            perf_rows.append({
                "数据集": name, "KS": _fmt_num(m.get("ks")), "AUC": _fmt_num(m.get("auc")),
                "Gini": _fmt_num(m.get("gini")), "PSI": _fmt_num(m.get("psi")),
            })
    add(_md_table(pd.DataFrame(perf_rows) if perf_rows else None))
    add("")
    cv_per_fold = _get(ctx, "cv_per_fold")
    cv_mean = _get(ctx, "cv_mean") or {}
    if cv_per_fold is not None and not (hasattr(cv_per_fold, "empty") and cv_per_fold.empty):
        add("**Expanding Window CV（逐年）**：")
        add("")
        add(_md_table(cv_per_fold, max_rows=20))
        add("")
    if cv_mean:
        add(f"- CV 均值：AUC={_fmt_num(cv_mean.get('auc'))}，KS={_fmt_num(cv_mean.get('ks'))}，"
            f"Gini={_fmt_num(cv_mean.get('gini'))}，fold 数={cv_mean.get('n_folds', '-')}")
        add("")
    add(analysis.get("performance", DEFAULT_PLACEHOLDER))
    add("")

    # ---------------- 七、特征重要性 ----------------
    add("## 七、特征重要性")
    add("")
    fi = _get(ctx, "feature_importance")
    add(_md_table(fi, max_rows=20))
    add("")

    # ---------------- 八、评分卡刻度与分档 ----------------
    add("## 八、评分卡刻度与分档")
    add("")
    card = _get(ctx, "scorecard") or {}
    if card:
        add(f"- 入模变量数：**{len(card)}**")
        add(f"- 刻度：base_score={cfg.get('base_score', 600)}，PDO={cfg.get('pdo', 20)}"
            f"（分数每高 {cfg.get('pdo', 20)} 分，违约 odds 翻倍）")
    else:
        add("_评分卡缺失_")
    add("")
    grades = _get(ctx, "grades")
    if grades is not None and not (hasattr(grades, "empty") and grades.empty):
        dist = (
            grades.groupby("grade", observed=True)
            .agg(n=("score", "size"), min_score=("score", "min"), max_score=("score", "max"))
            .reset_index()
        )
        dist["占比"] = (dist["n"] / dist["n"].sum()).map(lambda x: f"{x * 100:.2f}%")
        add("**测试集分档分布**：")
        add("")
        add(_md_table(dist, max_rows=10))
    else:
        add("_分档结果缺失_")
    add("")

    # ---------------- 九、cut-off 建议 ----------------
    add("## 九、cut-off 建议")
    add("")
    cutoff = _get(ctx, "cutoff") or {}
    if cutoff and cutoff.get("cutoff") is not None:
        add(f"- **推荐 cut-off：{_fmt_num(cutoff.get('cutoff'), 2)}**（择优口径：`{cutoff.get('method')}`）")
        add(f"- 通过率：**{_fmt_pct(cutoff.get('approve_rate'))}**")
        add(f"- 通过后坏率：**{_fmt_pct(cutoff.get('bad_rate_after'))}**"
            f"（整体坏率 {_fmt_pct(cutoff.get('base_bad_rate'))}）")
        add(f"- 违约捕获率：**{_fmt_pct(cutoff.get('capture_rate'))}**")
        add(f"- KS @ cut-off：**{_fmt_num(cutoff.get('ks_at_cutoff'))}**")
        lift = cutoff.get("lift")
        add(f"- 提升度 Lift：**{_fmt_num(lift, 2)}×**"
            + ("（通过后坏率极低，Lift 数值失真）" if lift in (float("inf"), float("nan")) else ""))
        tbl = cutoff.get("table")
        if tbl is not None and len(tbl):
            add("")
            add("<details><summary>展开全网格明细</summary>")
            add("")
            add(_md_table(tbl, max_rows=25))
            add("")
            add("</details>")
    else:
        add("_cut-off 结果缺失（可能样本无违约或在择优点退化）_")
    add("")

    # ---------------- 十、稳定性与护栏结论 ----------------
    add("## 十、稳定性与护栏结论")
    add("")
    if guardrail is not None:
        passed = getattr(guardrail, "passed", None)
        gr_metrics = getattr(guardrail, "metrics", {}) or {}
        lines.append(f"- 护栏是否通过：**{'是' if passed else '否'}**")
        add(f"- 结论文案：{getattr(guardrail, 'rationale', '-')}")
        add(f"- KS={_fmt_num(gr_metrics.get('ks'))}（阈值 {cfg.get('min_ks', 0.3)}），"
            f"AUC={_fmt_num(gr_metrics.get('auc'))}（阈值 {cfg.get('min_auc', 0.7)}），"
            f"PSI={_fmt_num(gr_metrics.get('psi'))}（阈值 {cfg.get('max_psi', 0.1)}）")
        gw = getattr(guardrail, "warnings", []) or []
        if gw:
            add("- 护栏告警：")
            add(_bullets(gw))
        if getattr(guardrail, "requires_human_review", False):
            add("- ⚠️ 已触发人工复核流程")
    else:
        add("_护栏结果缺失_")
    add("")

    # ---------------- 十一、模型复核结论（Critic） ----------------
    add("## 十一、模型复核结论（Critic）")
    add("")
    add("由 `tools/audit_tools.py` 的三项确定性审计产出，LLM 仅做定性复核"
        "（不得改写数值、不得新增规则外的条目）。")
    add("")
    critic = _get(ctx, "critic_report") or {}
    if critic:
        add(f"- 复核结论：**{'通过' if critic.get('passed') else '存在问题，需人工确认'}**"
            f"（来源 `{critic.get('source', 'rule')}`）")
        add(f"- 问题计数：共 **{critic.get('n_issues', 0)}** 项，"
            f"高危 high={critic.get('n_high', 0)} / 中 mid={critic.get('n_mid', 0)} / 低 low={critic.get('n_low', 0)}")
        add("")
        add("| 审计项 | 超限数 | 阈值/口径 |")
        add("| --- | --- | --- |")
        add(f"| 多重共线性 VIF | {critic.get('vif_fail', 0)}"
            f"（另有 {critic.get('vif_warn', 0)} 项 VIF>5） | VIF > "
            f"{cfg.get('vif_threshold', 10.0)} 判高危，> 5 判中 |")
        add(f"| 系数方向反转 | {critic.get('coef_flip', 0)} | 系数符号 vs WOE 风险方向相反 |")
        add(f"| 取值过度集中 | {critic.get('concentration_fail', 0)} | 单一取值占比 > "
            f"{_fmt_pct(cfg.get('concentration_threshold', 0.5), 0)} |")
        add("")
        issues = critic.get("issues") or []
        if issues:
            iss_df = pd.DataFrame([
                {
                    "严重度": str(i.get("severity", "low")),
                    "审计项": i.get("check", "-"),
                    "问题描述": i.get("item", "-"),
                    "处置建议": i.get("suggestion", "-"),
                }
                for i in issues
            ])
            add(_md_table(iss_df, max_rows=int(cfg.get("critic_max_rows", 20)), float_fmt=".3f"))
        else:
            add("三项审计均未发现超限项。")
    else:
        add("_Critic 复核结果缺失（可能未启用 critic 节点）_")
    add("")

    # ---------------- 十二、风险与局限 ----------------
    add("## 十二、风险与局限")
    add("")
    add(_bullets([
        "标签口径为「ST/*ST/暂停上市」，与真实债务违约存在偏差，反映的是监管口径的财务困境。",
        "WOE + 逻辑回归假设各变量对 odds 的影响可加，交互效应未被显式建模。",
        "PSI 基于训练/测试分布比较，样本外的时间稳定性需持续监控。",
        "自动重规划只调整超参数，不新增外部数据，无法突破数据本身的信息上限。",
    ]))
    add("")
    add(analysis.get("risk", DEFAULT_PLACEHOLDER))
    add("")

    # ---------------- 十三、附录 ----------------
    add("## 十三、附录：分箱明细")
    add("")
    bins = _get(ctx, "bins") or {}
    if bins:
        parts = []
        for var, bdf in bins.items():
            try:
                body = _md_table(bdf, max_rows=15)
            except Exception:  # noqa: BLE001
                continue
            parts.append(f"### {var}\n\n{body}\n")
        add("\n".join(parts) if parts else "_分箱明细渲染失败_")
    else:
        add("_无分箱数据_")

    add("")
    add("---")
    add("")
    add(f"_报告由 `tools/report_tools.py::render_model_report` 生成，Sections: {len(SECTIONS)}_")

    return "\n".join(lines)


# ============================================================================
# 《拒贷理由书》
# ============================================================================
def render_reject_letter(
    score_detail: pd.DataFrame,
    top_k: int = 5,
    row: int = 0,
    cutoff: float | None = None,
    field_desc: dict[str, str] | None = None,
) -> str:
    """生成单笔样本的拒贷/扣分理由书（纯模板，不依赖 LLM）。

    Args:
        score_detail: `scorecard_ply(..., only_total_score=False)` 的输出，
                      含各变量 `<var>_points` 列与总分 `score` 列
        top_k: 取负向贡献最大的 k 个变量
        row: 取第几行样本（默认第一行）
        cutoff: 若给出，用于在结论里给出通过/拒绝建议
        field_desc: 变量名 → 中文业务名映射，缺省直接用变量名

    Returns:
        中文理由书文本
    """
    if score_detail is None or len(score_detail) == 0:
        return "评分明细为空，无法生成理由书。"

    rec = score_detail.iloc[row] if isinstance(score_detail, pd.DataFrame) else score_detail
    field_desc = field_desc or {}

    var_cols = [c for c in score_detail.columns if c.endswith("_points") and c != "score"]
    if not var_cols:
        return "评分明细中缺少 `<变量>_points` 列，无法分解各变量贡献。"

    contrib = {c.replace("_points", ""): float(rec[c]) for c in var_cols}
    # 负分最多 = 拖累最大
    negatives = sorted(
        [(v, p) for v, p in contrib.items() if p < 0], key=lambda x: x[1]
    )[:top_k]
    positives = sorted(
        [(v, p) for v, p in contrib.items() if p > 0], key=lambda x: -x[1]
    )[:3]

    total = float(rec["score"]) if "score" in score_detail.columns else sum(contrib.values())

    lines = ["## 评分结果与主要扣分因素", ""]
    lines.append(f"- **总评分：{total:.0f} 分**")
    if cutoff is not None:
        lines.append(f"- **决策建议：{'通过' if total >= cutoff else '拒绝'}**（cut-off = {cutoff:.0f}）")
    lines.append("")

    if negatives:
        lines.append("**主要扣分因素（按拖累程度排序）：**")
        lines.append("")
        for i, (var, pts) in enumerate(negatives, 1):
            name = field_desc.get(var, var)
            raw = rec[var] if var in score_detail.columns else None
            raw_txt = ""
            try:
                if raw is not None and pd.notna(raw):
                    raw_txt = f"（当前值 {float(raw):.3f}）"
            except (TypeError, ValueError):
                raw_txt = f"（当前值 {raw}）"
            lines.append(f"{i}. **{name}**{raw_txt}：扣 **{abs(pts):.0f}** 分")
    else:
        lines.append("未出现明显负向扣分变量，总分偏低主要由基准分与各变量小幅扣分叠加导致。")

    if positives:
        lines.append("")
        lines.append("**主要加分因素：**")
        lines.append("")
        for var, pts in positives:
            name = field_desc.get(var, var)
            lines.append(f"- {name}：加 **{pts:.0f}** 分")

    lines.append("")
    lines.append("> 说明：以上贡献值来自评分卡各变量分值与基准分的加总，为本模型的确定性分解结果；")
    lines.append("> 仅用于披露主要影响因素，不构成最终信贷决策依据。")
    return "\n".join(lines)
