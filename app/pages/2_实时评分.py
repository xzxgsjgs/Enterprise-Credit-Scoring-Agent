"""app.pages.2_实时评分 / 加载评分卡并对新样本打分。

UX 设计：
- 自动加载「模型训练」页最新保存的评分卡；左侧上传框变为可选覆盖。
- 「单笔表单」改为「待评企业指标」，字段使用中文标签 + 输入提示。
- 「basepoints」等评分卡内部参数不出现在输入表单中。
- 分类变量使用 selectbox 而不是 number_input。
- 数值字段提供中位数 / 训练集分布作为默认值提示。
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.core.paths import LATEST_SCORECARD_PATH
from app.llm import SCORE_EXPLANATION, get_model
from tools import scorecard_ply

st.set_page_config(page_title="实时评分", page_icon="⚡", layout="wide")

st.title("⚡ 实时评分")

# ---- 字段元数据（中文标签 + 单位 + 英文缩写公式） ----
# 公式风格：英文缩写 + 下标 t / t-1。便于快速识别计算口径。
# 缩写字典：
#   TA = Total Assets, Rev = Revenue, NP = Net Profit, COGS = Cost of Goods Sold
#   CA = Current Assets, CL = Current Liabilities, Inv = Inventory, AR = Accounts Receivable
#   TL = Total Liabilities, Fin_Liab = Financial Liability, Op_Liab = Operating Liability
#   CFO = Operating Cash Flow, CapEx = Capital Expenditure, Equity = 净资产
#   ind_year = 行业当年均值, pp = percentage points
FIELD_META: dict[str, tuple[str, str, str | None]] = {
    # —— 资产负债表（绝对量）——
    "TotalAssets": ("总资产", "元", None),
    "OperatingRevenue": ("营业收入", "元", None),
    "FinancialLiability": ("金融性负债", "元", "ST_debt + LT_debt + Bonds_payable + Current_portion_LT_debt"),
    "OperatingLiability": ("经营性负债", "元", "AP + Notes_payable + Advance_from_customers + Payroll + Tax_payable + Other_payables"),

    # —— 增长率 ——
    "TotalAsset_Growth": ("总资产增长率", "%", "(TA_t - TA_t-1) / TA_t-1 × 100"),
    "Revenue_Growth": ("营业收入增长率", "%", "(Rev_t - Rev_t-1) / Rev_t-1 × 100"),
    "NetProfit_Growth": ("净利润增长率", "%", "(NP_t - NP_t-1) / NP_t-1 × 100"),

    # —— 盈利能力 ——
    "ROA": ("资产回报率 ROA", "%", "NP / Avg(TA_t, TA_t-1) × 100"),
    "ROE": ("净资产收益率 ROE", "%", "NP_parent / Avg(Equity_t, Equity_t-1) × 100"),
    "GrossMargin": ("销售毛利率", "%", "(Rev - COGS) / Rev × 100"),
    "NetProfitMargin": ("销售净利率", "%", "NP / Rev × 100"),
    "ROA_x_Turnover": ("ROA × 总资产周转率", "—", "ROA × TotalAsset_Turnover  杜邦分解"),

    # —— 偿债能力 ——
    "CurrentRatio": ("流动比率", "倍", "CA / CL"),
    "QuickRatio": ("速动比率", "倍", "(CA - Inv) / CL"),
    "CashRatio": ("现金比率", "倍", "(Cash + ST_investment) / CL"),
    "DebtToAsset": ("资产负债率", "%", "TL / TA × 100"),
    "FinancialDebtRatio": ("金融性负债占比", "%", "Fin_Liab / TL × 100"),
    "OperatingDebtRatio": ("经营性负债占比", "%", "Op_Liab / TL × 100"),

    # —— 营运能力 ——
    "Inventory_Turnover": ("存货周转率", "次/年", "COGS / Avg(Inv_t, Inv_t-1)"),
    "AR_Turnover": ("应收账款周转率", "次/年", "Rev / Avg(AR_t, AR_t-1)"),
    "TotalAsset_Turnover": ("总资产周转率", "次/年", "Rev / Avg(TA_t, TA_t-1)"),

    # —— 现金流 ——
    "CashFlow1": ("经营活动现金流净额", "元", "CFO  经营流入 - 经营流出"),
    "FCF": ("自由现金流", "元", "CFO - CapEx"),

    # —— 对数变换 ——
    "ln_TotalAssets": ("总资产对数", "ln(元)", "ln(TA)"),
    "ln_OperatingRevenue": ("营业收入对数", "ln(元)", "ln(Rev)"),

    # —— 公司基础信息 ——
    "FirmAge": ("公司年龄", "年", "Year_t - IPO_year"),
    "ContrshrProportion": ("控股股东持股比例", "%", "Top1_shares / Total_shares × 100"),
    "Ucsp": ("实际控制人类型", "代码", "1=国资 / 2=自然人 / 3=集体 / 4=民营 / 5=外资 / ..."),
    "IndustryCode": ("行业代码", "代码", "申万行业分类代码"),
    "IndustrySector": ("行业大类", "类别", "申万一级行业名称"),
    "Province": ("所在省份", "省份名", None),

    # —— 行业偏离（公司值 − 同行业当年均值）——
    "ROA_industry_dev": ("ROA 行业偏离", "pp", "ROA - mean(ROA)_ind_year"),
    "ROE_industry_dev": ("ROE 行业偏离", "pp", "ROE - mean(ROE)_ind_year"),
    "GrossMargin_industry_dev": ("毛利率行业偏离", "pp", "GrossMargin - mean(GrossMargin)_ind_year"),
    "NetProfitMargin_industry_dev": ("净利率行业偏离", "pp", "NetProfitMargin - mean(NetProfitMargin)_ind_year"),
    "CurrentRatio_industry_dev": ("流动比率行业偏离", "倍", "CurrentRatio - mean(CurrentRatio)_ind_year"),
    "DebtToAsset_industry_dev": ("资产负债率行业偏离", "pp", "DebtToAsset - mean(DebtToAsset)_ind_year"),
    "TotalAsset_Turnover_industry_dev": ("总资产周转率行业偏离", "次/年", "TAT - mean(TAT)_ind_year"),

    # —— 3 年滚动统计（t-2 / t-1 / t）——
    "DebtToAsset_3y_avg": ("近 3 年资产负债率均值", "%", "mean(DebtToAsset_t-2, _, t)"),
    "DebtToAsset_3y_std": ("近 3 年资产负债率波动", "%", "std(DebtToAsset_t-2, _, t)"),
    "ROA_3y_avg": ("近 3 年 ROA 均值", "%", "mean(ROA_t-2, _, t)"),
    "ROA_3y_std": ("近 3 年 ROA 波动", "%", "std(ROA_t-2, _, t)"),
    "OperatingRevenue_3y_avg": ("近 3 年营业收入均值", "元", "mean(Rev_t-2, _, t)"),
    "OperatingRevenue_3y_std": ("近 3 年营业收入波动", "元", "std(Rev_t-2, _, t)"),
    "ln_TotalAssets_3y_avg": ("近 3 年总资产规模均值", "ln(元)", "mean(lnTA_t-2, _, t)"),
    "ln_TotalAssets_3y_std": ("近 3 年总资产规模波动", "ln(元)", "std(lnTA_t-2, _, t)"),
    "CurrentRatio_3y_avg": ("近 3 年流动比率均值", "倍", "mean(CurrentRatio_t-2, _, t)"),
    "CurrentRatio_3y_std": ("近 3 年流动比率波动", "倍", "std(CurrentRatio_t-2, _, t)"),

    # —— 其他 ——
    "year": ("年份", "年", "数据所属年份"),
}

# 类别变量（用 selectbox 而不是 number_input）。key=变量名，value=可选值列表。
CATEGORICAL_FIELDS: dict[str, list[Any]] = {
    "Ucsp": [1, 2, 3, 4, 5, 6, 7, 8, 9],
}

# 评分卡内部参数，不应让用户填
INTERNAL_KEYS = {"basepoints"}


def _meta_of(var: str) -> tuple[str, str, str | None]:
    """返回 (中文标签, 单位, 计算公式) 三元组；缺失时给默认值。"""
    if var in FIELD_META:
        return FIELD_META[var]
    return (var, "", None)


def _label_of(var: str) -> str:
    return _meta_of(var)[0]


# ---- 加载评分卡 ----
scorecard_file = st.sidebar.file_uploader(
    "上传评分卡 (可选 · 已自动加载最新训练结果)",
    type=["pkl"],
    help="默认从训练页最新产物加载；如需使用旧评分卡，手动上传覆盖",
)
cutoff = st.sidebar.number_input(
    "通过阈值 cutoff", 300, 1000, 600, 10,
    help="分数大于等于该值时，决策建议「通过」",
)
mode = st.radio("输入模式", ["单笔表单", "批量 CSV"], horizontal=True)

scorecard_bytes = None
if scorecard_file is not None:
    scorecard_bytes = scorecard_file.getvalue()
    st.sidebar.caption("已使用手动上传的评分卡")
elif LATEST_SCORECARD_PATH.exists():
    scorecard_bytes = LATEST_SCORECARD_PATH.read_bytes()
    st.sidebar.caption(f"已自动加载最新评分卡: {LATEST_SCORECARD_PATH.name}")

if scorecard_bytes is None:
    st.info("请在「模型训练」页面训练后自动加载，或手动上传 scorecard.pkl。")
    st.stop()

try:
    scorecard = pickle.loads(scorecard_bytes)
    variables = [v for v in scorecard.keys() if v not in INTERNAL_KEYS]
    st.sidebar.caption(f"评分卡变量数: {len(variables)}（已隐藏内部基准分）")
except Exception as e:
    st.error(f"评分卡加载失败: {e}")
    st.stop()


# ---- 单笔表单 ----
if mode == "单笔表单":
    st.subheader("填写待评企业指标")
    st.caption("按训练时使用的特征填入。分类变量以下拉选择，数值变量填整数或浮点数。")

    with st.form("scoring_form"):
        inputs: dict[str, Any] = {}
        cols = st.columns(3)
        for i, var in enumerate(variables):
            with cols[i % 3]:
                label_zh, unit, formula = _meta_of(var)
                # 中文标签后拼接单位（如「总资产（元）」），空单位则只保留标签
                label_display = f"{label_zh}（{unit}）" if unit else label_zh
                # 帮助文本：计算公式 + 原始字段名
                help_parts = [f"原始字段名: {var}"]
                if formula:
                    help_parts.append(f"计算公式: {formula}")
                help_text = "\n\n".join(help_parts)

                if var in CATEGORICAL_FIELDS:
                    options = CATEGORICAL_FIELDS[var]
                    inputs[var] = st.selectbox(
                        label_display,
                        options=options,
                        index=None,
                        placeholder="选择 ...",
                        key=f"input_{var}",
                        help=help_text,
                    )
                else:
                    inputs[var] = st.number_input(
                        label_display,
                        value=None,
                        placeholder=f"输入数值（{unit}）" if unit else "输入数值",
                        key=f"input_{var}",
                        help=help_text,
                    )
        submitted = st.form_submit_button("开始打分", type="primary", width="stretch")

    if submitted:
        # 校验必填
        missing = [v for v, val in inputs.items() if val is None or val == ""]
        if missing:
            st.error(f"以下变量未填写: {[ _label_of(v) for v in missing ]}")
            st.stop()

        sample_df = pd.DataFrame([inputs])
        try:
            scored = scorecard_ply(sample_df, scorecard, only_total_score=False)
        except Exception as e:
            st.error(f"评分失败：{e}（可能原因：变量值不在训练分箱范围内）")
            st.stop()

        total_score = int(scored["score"].iloc[0])
        passed = total_score >= cutoff

        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("最终评分", total_score)
        col2.metric("决策", "通过" if passed else "拒绝")
        col3.metric("与阈值差", f"{total_score - cutoff:+d}")

        var_score_cols = [c for c in scored.columns if c.endswith("_points") and c != "score"]
        if var_score_cols:
            var_scores = {c.replace("_points", ""): int(scored[c].iloc[0]) for c in var_score_cols}
            # 用中文标签显示
            disp = pd.DataFrame(
                [
                    {
                        "指标": _label_of(k),
                        "变量名": k,
                        "得分": v,
                    }
                    for k, v in var_scores.items()
                ]
            ).sort_values("得分", ascending=False)
            st.write("各变量得分（按贡献排序）", disp)

            if st.button("LLM 解释评分", key="explain_score"):
                try:
                    llm = get_model()
                    prompt = SCORE_EXPLANATION.format(
                        variable_scores_json=json.dumps(var_scores, ensure_ascii=False, indent=2),
                        total_score=total_score,
                        cutoff=cutoff,
                    )
                    with st.spinner("LLM 思考中..."):
                        response = llm.invoke(prompt)
                    st.markdown(response.content)
                except Exception as e:
                    st.error(f"LLM 调用失败: {e}")

# ---- 批量 CSV ----
else:
    st.subheader("批量打分")
    st.caption("上传一张包含与训练数据相同字段的表格，逐行输出信用分数。")

    with st.expander("必填字段说明（含计算公式）", expanded=True):
        st.caption("CSV 列名需与训练数据一致；变量下标 t / t-1 表示本年/上年。")
        # 构造字段说明表
        meta_rows = []
        for v in variables:
            label_zh, unit, formula = _meta_of(v)
            meta_rows.append({
                "变量名": f"`{v}`",
                "中文标签": label_zh,
                "单位": unit if unit else "—",
                "计算公式": formula if formula else "（直接读取，无计算）",
            })
        meta_df = pd.DataFrame(meta_rows)
        st.dataframe(
            meta_df,
            width="stretch",
            hide_index=True,
            column_config={
                "变量名": st.column_config.TextColumn("变量名", width="medium"),
                "中文标签": st.column_config.TextColumn("中文标签", width="medium"),
                "单位": st.column_config.TextColumn("单位", width="small"),
                "计算公式": st.column_config.TextColumn("计算公式", width="large"),
            },
        )

    batch_file = st.file_uploader("上传待打分 CSV / Excel", type=["csv", "xlsx", "xls"])
    if batch_file is not None:
        suffix = Path(batch_file.name).suffix.lower()
        batch_df = pd.read_csv(batch_file) if suffix == ".csv" else pd.read_excel(batch_file)
        st.write(f"待打样本: {len(batch_df)} 行 × {batch_df.shape[1]} 列", batch_df.head(20))

        if st.button("批量打分", type="primary", width="stretch"):
            try:
                scored = scorecard_ply(batch_df, scorecard, only_total_score=True)
                result_df = batch_df.copy()
                result_df["score"] = scored["score"].values
                result_df["decision"] = result_df["score"].apply(
                    lambda s: "通过" if s >= cutoff else "拒绝"
                )

                st.write("打分结果", result_df)
                st.download_button(
                    label="下载结果 (CSV)",
                    data=result_df.to_csv(index=False).encode("utf-8"),
                    file_name="scoring_result.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"批量打分失败: {e}")