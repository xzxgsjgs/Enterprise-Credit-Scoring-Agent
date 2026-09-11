"""prepare_csmar_panel.py —— 将 CSMAR 三个 CSV 合并为企业信用评分训练面板。

输入（解压后）：
    D:/vibe coding/data store/extracted/
        - FI_T1(Merge Query).csv          财务指标
        - BDT_FinConstSA.csv              ST/退市/暂停上市标签
        - STK_LISTEDCOINFOANL(Merge Query).csv  公司基本信息/控制变量

输出：
    D:/vibe coding/data store/csmar_enterprise_panel.csv

关键处理（v2）：
1. 财务数据只保留合并报表(A)、年报(12-31)、2015-2025 年。
2. ST 标签构造违约事件：STPT=1 或 IsSuspend=1 视为 distressed。
3. is_default_next_year：用 t 年特征预测 t+1 年是否 distressed。
4. 特征工程：对数总资产/营收、金融负债率、经营负债率、企业年龄。
5. 合并键统一为 6 位字符串股票代码 + 年份。
6. 【v2 新增】剔除金融业 J6*（GB/T 4754 大类 J：货币金融/资本市场/保险/其他金融）。
7. 【v2 新增】行业相对特征：company_metric − industry_median（group by year+IndustrySector）。
8. 【v2 新增】3 年滚动时序特征：avg / std，捕捉趋势与波动。
9. 【v2 新增】核心数值特征按 1%/99% 分位 winsorize，抑制极端值。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# 路径配置
# -----------------------------------------------------------------------------
DATA_DIR = Path("D:/vibe coding/data store")
EXTRACTED_DIR = DATA_DIR / "extracted"
OUTPUT_PATH = DATA_DIR / "csmar_enterprise_panel.csv"

FIN_PATH = EXTRACTED_DIR / "FI_T1(Merge Query).csv"
ST_PATH = EXTRACTED_DIR / "BDT_FinConstSA.csv"
CTRL_PATH = EXTRACTED_DIR / "STK_LISTEDCOINFOANL(Merge Query).csv"

# -----------------------------------------------------------------------------
# 1. 读取原始表
# -----------------------------------------------------------------------------
print("读取财务指标表...")
fin = pd.read_csv(FIN_PATH, encoding="utf-8")
print(f"  原始 shape: {fin.shape}")

print("读取 ST 标签表...")
st = pd.read_csv(ST_PATH, encoding="utf-8")
print(f"  原始 shape: {st.shape}")

print("读取控制变量表...")
ctrl = pd.read_csv(CTRL_PATH, encoding="utf-8")
print(f"  原始 shape: {ctrl.shape}")

# -----------------------------------------------------------------------------
# 2. 清洗财务指标表
# -----------------------------------------------------------------------------
fin_cols = {
    "FI_T1.Stkcd": "Symbol",
    "FI_T1.ShortName": "ShortName",
    "FI_T1.Accper": "EndDate",
    "FI_T1.F010101A": "CurrentRatio",
    "FI_T1.F010201A": "QuickRatio",
    "FI_T1.F010401A": "CashRatio",
    "FI_T1.F011201A": "DebtToAsset",
    "FI_T5.F050201B": "ROA",
    "FI_T5.F050501B": "ROE",
    "FI_T5.F053301B": "GrossMargin",
    "FI_T5.F051501B": "NetProfitMargin",
    "FI_T4.F040201B": "AR_Turnover",
    "FI_T4.F040501B": "Inventory_Turnover",
    "FI_T4.F041701B": "TotalAsset_Turnover",
    "FI_T8.F080601A": "TotalAsset_Growth",
    "FI_T8.F081001B": "NetProfit_Growth",
    "FI_T8.F081601B": "Revenue_Growth",
    "FI_T6.F061301B": "CashFlow1",
    "FI_T6.F062401B": "FCF",
    "BDT_FinIndex.TotalAssets": "TotalAssets",
    "BDT_FinIndex.OperatingRevenue": "OperatingRevenue",
    "BDT_FinIndex.FinancialLiability": "FinancialLiability",
    "BDT_FinIndex.OperatingLiability": "OperatingLiability",
}
fin = fin.rename(columns=fin_cols)

# 统一股票代码为 6 位字符串
fin["Symbol"] = fin["Symbol"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)

# 只保留年报、合并报表、指定时间范围
fin = fin[fin["EndDate"].str.endswith("-12-31")]
fin = fin[fin["FI_T1.Typrep"] == "A"]
fin = fin[(fin["EndDate"] >= "2015-12-31") & (fin["EndDate"] <= "2025-12-31")].copy()
fin["year"] = fin["EndDate"].str[:4].astype(int)

print(f"财务指标表清洗后 shape: {fin.shape}")
print(f"财务指标表年份范围: {fin['year'].min()} - {fin['year'].max()}")

# -----------------------------------------------------------------------------
# 3. 清洗 ST / 退市标签表
# -----------------------------------------------------------------------------
st["Symbol"] = st["Symbol"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
st["year"] = pd.to_datetime(st["Enddate"]).dt.year
st["is_distressed"] = ((st["STPT"] == 1) | (st["IsSuspend"] == 1)).astype(int)
print(f"ST 标签表年份范围: {st['year'].min()} - {st['year'].max()}")
print(f" distressed 样本数: {st['is_distressed'].sum()} / {len(st)}")

# -----------------------------------------------------------------------------
# 4. 清洗控制变量表
# -----------------------------------------------------------------------------
ctrl_cols = {
    "STK_LISTEDCOINFOANL.Symbol": "Symbol",
    "STK_LISTEDCOINFOANL.ShortName": "CtrlShortName",
    "STK_LISTEDCOINFOANL.EndDate": "EndDate",
    "STK_LISTEDCOINFOANL.IndustryCode": "IndustryCode",
    "PRI_Basic.Ucsp": "Ucsp",
    "BDT_ManaGovAbil.ContrshrProportion": "ContrshrProportion",
    "csmar_listedcoinfo.PROVINCECD": "Province",
    "csmar_listedcoinfo.Listdt": "ListDate",
}
ctrl = ctrl.rename(columns=ctrl_cols)
ctrl["Symbol"] = ctrl["Symbol"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
ctrl["year"] = pd.to_datetime(ctrl["EndDate"]).dt.year

# 行业代码取大类（前两位字母）
ctrl["IndustrySector"] = ctrl["IndustryCode"].astype(str).str[:2]

# 上市日期转年份
ctrl["ListYear"] = pd.to_datetime(ctrl["ListDate"], errors="coerce").dt.year

print(f"控制变量表年份范围: {ctrl['year'].min()} - {ctrl['year'].max()}")

# -----------------------------------------------------------------------------
# 5. 合并三表
# -----------------------------------------------------------------------------
print("合并三表...")
panel = fin.merge(
    st[["Symbol", "year", "is_distressed"]],
    on=["Symbol", "year"],
    how="left",
)

panel = panel.merge(
    ctrl[[
        "Symbol", "year", "IndustryCode", "IndustrySector",
        "Ucsp", "ContrshrProportion", "Province", "ListYear",
    ]],
    on=["Symbol", "year"],
    how="left",
)

# -----------------------------------------------------------------------------
# 6. 构造 is_default_next_year
# -----------------------------------------------------------------------------
print("构造 is_default_next_year...")
panel = panel.sort_values(["Symbol", "year"]).reset_index(drop=True)
panel["is_default_next_year"] = panel.groupby("Symbol")["is_distressed"].shift(-1)

# 只有知道下一年标签的样本才有效
panel = panel.dropna(subset=["is_default_next_year"]).copy()
panel["is_default_next_year"] = panel["is_default_next_year"].astype(int)

# 由于 ST 标签只到 2024 年，2024 年的特征对应 2025 年标签未知，因此
# 可用训练特征年最大为 2023（预测 2024 年违约）。这里保留用户可见说明。
print(f"构造标签后 shape: {panel.shape}")
print(f"  特征年份范围: {panel['year'].min()} - {panel['year'].max()}")
print(f"  is_default_next_year=1 占比: {panel['is_default_next_year'].mean():.2%}")

# -----------------------------------------------------------------------------
# 6.5 【v2】剔除金融业 J6*（GB/T 4754 大类 J：货币金融/资本市场/保险/其他金融）
# -----------------------------------------------------------------------------
# CSMAR IndustryCode 取前两位字母后，J6 代表"金融业"大类（含 J66-J69 全部子类）。
# 这些行业的资产负债结构与实体企业差异巨大，财务指标不可比，剔除后再建模。
EXCLUDED_SECTORS = ["J6"]
before = len(panel)
panel = panel[~panel["IndustrySector"].astype(str).str.startswith("J6")].copy()
print(f"剔除金融业 J6*: {before} -> {len(panel)}（剔除 {before - len(panel)} 行）")

# -----------------------------------------------------------------------------
# 7. 特征工程
# -----------------------------------------------------------------------------
print("特征工程...")

# 对数规模（避免 0 / 负值）
panel["ln_TotalAssets"] = np.log(panel["TotalAssets"].where(panel["TotalAssets"] > 0))
panel["ln_OperatingRevenue"] = np.log(panel["OperatingRevenue"].where(panel["OperatingRevenue"] > 0))

# 负债结构
panel["FinancialDebtRatio"] = panel["FinancialLiability"] / panel["TotalAssets"]
panel["OperatingDebtRatio"] = panel["OperatingLiability"] / panel["TotalAssets"]

# 企业年龄
panel["FirmAge"] = panel["year"] - panel["ListYear"]

# 盈利能力交互
panel["ROA_x_Turnover"] = panel["ROA"] * panel["TotalAsset_Turnover"]

# 实际控制人类型：CSMAR 中可能为 "1,2" 等多重控制，保留字符串分类
panel["Ucsp"] = panel["Ucsp"].fillna("8").astype(str).str.strip()

# -----------------------------------------------------------------------------
# 7.5 【v2】行业相对特征（company − industry median，同 year）
# -----------------------------------------------------------------------------
# 行业内中位数更能反映"相对竞争力/相对负担"，跨行业绝对值不可比。
IND_RELATIVE_FEATS = [
    "DebtToAsset", "CurrentRatio", "ROA", "ROE",
    "GrossMargin", "NetProfitMargin", "TotalAsset_Turnover",
]
print("构造行业相对特征...")
for col in IND_RELATIVE_FEATS:
    grp_med = panel.groupby(["year", "IndustrySector"])[col].transform("median")
    panel[f"{col}_industry_dev"] = panel[col] - grp_med

# -----------------------------------------------------------------------------
# 7.6 【v2】3 年滚动时序特征（avg / std，捕捉趋势与波动）
# -----------------------------------------------------------------------------
# shift(1) 避免使用当前年份信息穿越；窗口 [t-3, t-1] 共 3 个历史年度。
ROLLING_FEATS = ["DebtToAsset", "ROA", "OperatingRevenue", "ln_TotalAssets", "CurrentRatio"]
ROLLING_WINDOWS = [3]
print("构造滚动时序特征...")
panel = panel.sort_values(["Symbol", "year"]).reset_index(drop=True)
g = panel.groupby("Symbol")
for col in ROLLING_FEATS:
    for w in ROLLING_WINDOWS:
        s = g[col].shift(1)  # 严格使用历史期
        panel[f"{col}_{w}y_avg"] = s.rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)
        panel[f"{col}_{w}y_std"] = s.rolling(window=w, min_periods=2).std().reset_index(level=0, drop=True)

# -----------------------------------------------------------------------------
# 7.7 【v2】核心数值特征 1%/99% Winsorize（按全样本分位，不分年，避免小年样本噪音）
# -----------------------------------------------------------------------------
WINSORIZE_FEATS = [
    "DebtToAsset", "CurrentRatio", "QuickRatio", "CashRatio",
    "ROA", "ROE", "GrossMargin", "NetProfitMargin",
    "AR_Turnover", "Inventory_Turnover", "TotalAsset_Turnover",
    "TotalAsset_Growth", "NetProfit_Growth", "Revenue_Growth",
    "FinancialDebtRatio", "OperatingDebtRatio",
    "ContrshrProportion",
]
print("Winsorize 1%/99% 分位...")
for col in WINSORIZE_FEATS:
    if col not in panel.columns:
        continue
    lo = panel[col].quantile(0.01)
    hi = panel[col].quantile(0.99)
    panel[col] = panel[col].clip(lower=lo, upper=hi)

# -----------------------------------------------------------------------------
# 8. 列选择与输出
# -----------------------------------------------------------------------------
feature_cols = [
    # 标识
    "Symbol", "ShortName", "year", "EndDate",
    # 目标
    "is_default_next_year",
    # 偿债能力
    "CurrentRatio", "QuickRatio", "CashRatio", "DebtToAsset",
    # 盈利能力
    "ROA", "ROE", "GrossMargin", "NetProfitMargin",
    # 经营能力
    "AR_Turnover", "Inventory_Turnover", "TotalAsset_Turnover",
    # 发展能力（原始 CSMAR 字段是单季度环比，建议谨慎使用；后续可替换为自行计算的年度同比）
    "TotalAsset_Growth", "NetProfit_Growth", "Revenue_Growth",
    # 现金流
    "CashFlow1", "FCF",
    # 规模与负债结构
    "TotalAssets", "OperatingRevenue",
    "FinancialLiability", "OperatingLiability",
    "ln_TotalAssets", "ln_OperatingRevenue",
    "FinancialDebtRatio", "OperatingDebtRatio",
    # 控制变量
    "IndustryCode", "IndustrySector", "Ucsp",
    "ContrshrProportion", "Province", "FirmAge",
    # 交互
    "ROA_x_Turnover",
]
# 【v2】扩展：行业相对特征
for col in IND_RELATIVE_FEATS:
    feature_cols.append(f"{col}_industry_dev")
# 【v2】扩展：滚动时序特征
for col in ROLLING_FEATS:
    for w in ROLLING_WINDOWS:
        feature_cols.append(f"{col}_{w}y_avg")
        feature_cols.append(f"{col}_{w}y_std")
# 去重保序
seen = set()
feature_cols = [c for c in feature_cols if not (c in seen or seen.add(c))]

out_df = panel[feature_cols].copy()

# 基本统计
print("\n数据集基本统计:")
print(f"  样本数: {len(out_df)}")
print(f"  公司数: {out_df['Symbol'].nunique()}")
print(f"  年份范围: {out_df['year'].min()} - {out_df['year'].max()}")
print(f"  违约率 (is_default_next_year=1): {out_df['is_default_next_year'].mean():.2%}")
print(f"  缺失值最多的 5 列:")
print((out_df.isnull().sum() / len(out_df)).sort_values(ascending=False).head())

out_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(f"\n已保存: {OUTPUT_PATH}")
