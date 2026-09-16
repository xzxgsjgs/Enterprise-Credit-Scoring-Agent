"""scripts.generate_test_data —— 从原始面板抽样 + 扰动，生成测试用小样本。

目的：
- 让你在 Streamlit UI 里手动上传一个 ~300-500 行的 CSV 快速跑通训练流程
- 测试样本保留原面板的列结构、时间序列覆盖、违约率分布
- 输出到外部缓存目录 credit_agent_cache/test_panel.csv，不污染仓库

用法：
    python scripts/generate_test_data.py              # 默认 50 公司 × 7 年 ≈ 350 行
    python scripts/generate_test_data.py --n 800      # 自定义行数
    python scripts/generate_test_data.py --seed 7    # 自定义随机种子
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 项目根加入 path，便于 import paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.paths import CACHE_DIR  # noqa: E402

SOURCE_CSV = Path(r"D:\vibe coding\data store\csmar_enterprise_panel.csv")
OUTPUT_CSV = CACHE_DIR / "test_panel.csv"

# 时序切分所需的年份范围（与原面板一致）
DEFAULT_YEARS = list(range(2018, 2025))  # 2018-2024


def sample_companies(df: pd.DataFrame, n_companies: int, seed: int) -> list[str]:
    """从面板里随机抽取 n_companies 个 Symbol。"""
    symbols = df["Symbol"].dropna().unique().tolist()
    rng = np.random.default_rng(seed)
    if n_companies > len(symbols):
        n_companies = len(symbols)
    return rng.choice(symbols, size=n_companies, replace=False).tolist()


def build_sample(
    df: pd.DataFrame,
    symbols: list[str],
    years: list[int],
    seed: int,
    min_default: int = 30,
) -> pd.DataFrame:
    """对每个 Symbol 抽取指定年份范围，再加微小高斯扰动（让分数能学到东西）。

    关键约束：
    - target 列 (is_default_next_year) 必须是 0/1 整数，绝不扰动
    - 强制保证违约样本数 ≥ min_default，否则追加历史违约 Symbol
    """
    rng = np.random.default_rng(seed + 1)

    # 1. 切出子集
    sub = df[df["Symbol"].isin(symbols) & df["year"].isin(years)].copy()
    if sub.empty:
        raise ValueError(f"未匹配到任何样本。请检查面板是否覆盖这些 Symbol/year: {symbols[:3]} ... {years}")

    # 2. 保证违约样本足够：追加历史上违约过的 Symbol
    n_default = int(sub["is_default_next_year"].sum())
    if n_default < min_default:
        default_symbols = (
            df.loc[df["is_default_next_year"] == 1, "Symbol"].dropna().unique().tolist()
        )
        need = min_default - n_default
        # 用独立 rng 选，避免污染主 rng
        pick_rng = np.random.default_rng(seed + 2)
        extra = pick_rng.choice(
            [s for s in default_symbols if s not in symbols],
            size=min(need, len(default_symbols)),
            replace=False,
        ).tolist()
        if extra:
            extra_sub = df[df["Symbol"].isin(extra) & df["year"].isin(years)].copy()
            sub = pd.concat([sub, extra_sub], ignore_index=True)
            print(f"[test-data] 违约样本不足，追加 {len(extra)} 家违约 Symbol；当前违约数 = {int(sub['is_default_next_year'].sum())}")

    # 3. 按 Symbol × year 排序，再 reset
    sub = sub.sort_values(["Symbol", "year"]).reset_index(drop=True)

    # 4. 对数值列加入微小扰动（sigma=1%）。明确排除 target + year。
    PROTECTED = {"year", "is_default_next_year"}
    num_cols = [c for c in sub.columns if sub[c].dtype.kind in "iuf" and c not in PROTECTED]
    for c in num_cols:
        col = sub[c].astype(float)
        std = col.std()
        if std and not np.isnan(std):
            noise = rng.normal(0, 0.01 * std, size=len(sub))
            sub[c] = col + noise

    # 5. 强制 target 列保持 0/1 整数（防止之前 float 化）
    sub["is_default_next_year"] = sub["is_default_next_year"].round().clip(0, 1).astype(int)

    # 6. 按 Symbol 略微扰动行业/省份，避免某行业占比过高
    cat_cols = ["IndustryCode", "IndustrySector", "Province", "Ucsp"]
    for c in cat_cols:
        if c in sub.columns:
            mask = rng.random(len(sub)) < 0.05  # 5% 概率扰动
            choices = sub[c].dropna().unique().tolist()
            if choices:
                sub.loc[mask, c] = rng.choice(choices, size=int(mask.sum()))

    return sub


def print_summary(df: pd.DataFrame, output_csv: Path | None = None) -> None:
    n = len(df)
    n_companies = df["Symbol"].nunique()
    year_range = (int(df["year"].min()), int(df["year"].max()))
    default_rate = float(df["is_default_next_year"].mean())
    print(f"[test-data] {n} 行 × {df.shape[1]} 列")
    print(f"[test-data] {n_companies} 家公司；年份 {year_range[0]}-{year_range[1]}")
    print(f"[test-data] 违约率 {default_rate:.2%}")
    print(f"[test-data] 输出 -> {output_csv or OUTPUT_CSV}")


def main() -> None:
    p = argparse.ArgumentParser(description="生成测试用面板样本")
    p.add_argument("--n", type=int, default=50, help="抽多少家公司（默认 50）")
    p.add_argument("--years", type=str, default=",".join(str(y) for y in DEFAULT_YEARS),
                   help="年份列表，逗号分隔")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--source", type=Path, default=SOURCE_CSV, help="源面板 CSV")
    p.add_argument("--output", type=Path, default=OUTPUT_CSV, help="输出 CSV")
    p.add_argument("--min-default", type=int, default=30, help="最少违约样本数（不够会自动追加违约 Symbol）")
    args = p.parse_args()

    if not args.source.exists():
        print(f"[error] 源面板不存在: {args.source}", file=sys.stderr)
        sys.exit(1)

    years = [int(y) for y in args.years.split(",") if y.strip()]
    print(f"[test-data] 加载源面板: {args.source}")
    df = pd.read_csv(args.source, encoding="utf-8-sig")
    print(f"[test-data] 源面板: {len(df)} 行 × {df.shape[1]} 列")

    symbols = sample_companies(df, args.n, args.seed)
    sample = build_sample(df, symbols, years, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.output, index=False, encoding="utf-8-sig")

    print_summary(sample, args.output)


if __name__ == "__main__":
    main()