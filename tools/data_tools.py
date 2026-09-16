"""数据加载、预处理工具集

涵盖:
    - load_data: CSV/Excel 数据加载与基础校验
    - explore_data: EDA 探索性数据分析
    - split_dataset: 训练/测试集划分
    - handle_missing: 缺失值处理
    - handle_outliers: 异常值盖帽处理
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_data(
    file_path: str | Path,
    target: str | None = None,
    encoding: str = "utf-8-sig",
    sep: str | None = None,
) -> pd.DataFrame:
    """加载 CSV / Excel 数据文件并执行基础校验。

    Args:
        file_path: 数据文件路径，支持 .csv / .xlsx / .xls
        target: 目标列名（标签列），若指定则必须存在于数据中
        encoding: 文件编码，默认 utf-8-sig（自动处理 BOM）
        sep: CSV 分隔符，默认自动推断

    Returns:
        加载后的 DataFrame

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 目标列缺失或数据为空

    Example:
        >>> df = load_data("german_credit.csv", target="creditability")
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        # pandas 3.0+ 移除了 sep=None 自动推断，改用 engine="python" + sep=None
        if sep is None:
            df = pd.read_csv(path, encoding=encoding, sep=None, engine="python")
        else:
            df = pd.read_csv(path, encoding=encoding, sep=sep)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"不支持的文件格式: {suffix}")

    if df.empty:
        raise ValueError("加载的数据为空，请检查文件内容")

    if target is not None and target not in df.columns:
        raise ValueError(f"目标列 '{target}' 不在数据列中: {list(df.columns)}")

    logger.info("数据加载成功: shape=%s, target=%s", df.shape, target)
    return df


def explore_data(
    df: pd.DataFrame,
    target: str,
) -> dict[str, Any]:
    """对数据进行探索性分析，返回结构化摘要。

    Args:
        df: 输入数据
        target: 目标列名

    Returns:
        包含以下键的字典:
            - n_rows, n_cols
            - target_distribution: dict (label -> count)
            - missing_ratio: dict (col -> ratio)
            - dtype_summary: dict (col -> dtype)
            - numeric_stats: DataFrame
            - categorical_stats: DataFrame
    """
    if target not in df.columns:
        raise ValueError(f"目标列 '{target}' 不在数据列中")

    target_dist = df[target].value_counts().to_dict()
    missing_ratio = (df.isnull().sum() / len(df)).round(4).to_dict()
    dtype_summary = df.dtypes.astype(str).to_dict()

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_stats = df[num_cols].describe().T if num_cols else pd.DataFrame()
    categorical_stats = (
        df[cat_cols].describe(include="all").T if cat_cols else pd.DataFrame()
    )

    summary = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "target_distribution": {str(k): int(v) for k, v in target_dist.items()},
        "missing_ratio": {k: float(v) for k, v in missing_ratio.items()},
        "dtype_summary": dtype_summary,
        "numeric_stats": numeric_stats,
        "categorical_stats": categorical_stats,
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
    }
    logger.info("EDA 完成: %d 行 × %d 列", df.shape[0], df.shape[1])
    return summary


def split_dataset(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.3,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """将数据集划分为训练集与测试集。

    Args:
        df: 输入数据
        target: 目标列名
        test_size: 测试集占比
        random_state: 随机种子
        stratify: 是否分层抽样

    Returns:
        (train_df, test_df) — 划分后的两个 DataFrame
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size 必须在 (0, 1) 之间，得到 {test_size}")

    strat = df[target] if stratify else None
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=strat
    )
    logger.info(
        "数据集划分: train=%d, test=%d", len(train_df), len(test_df)
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def handle_missing(
    df: pd.DataFrame,
    strategy: Literal["mean", "median", "mode", "constant", "drop"] = "median",
    fill_value: float | str | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """对数值/类别变量进行缺失值处理。

    Args:
        df: 输入数据
        strategy: 策略，median/mean 用于数值列，mode/constant/drop 通用
        fill_value: 当 strategy='constant' 时使用的填充值
        columns: 指定需要处理的列，默认对全数据生效

    Returns:
        处理后的 DataFrame (原 DataFrame 不被修改)
    """
    out = df.copy()
    target_cols = columns if columns is not None else out.columns.tolist()

    num_cols = [
        c for c in target_cols
        if c in out.columns and pd.api.types.is_numeric_dtype(out[c])
    ]

    if strategy == "mean":
        for c in num_cols:
            out[c] = out[c].fillna(out[c].mean())
    elif strategy == "median":
        for c in num_cols:
            out[c] = out[c].fillna(out[c].median())
    elif strategy == "mode":
        for c in target_cols:
            if c in out.columns and out[c].isnull().any():
                out[c] = out[c].fillna(out[c].mode().iloc[0])
    elif strategy == "constant":
        if fill_value is None:
            raise ValueError("strategy='constant' 必须提供 fill_value")
        for c in target_cols:
            if c in out.columns:
                out[c] = out[c].fillna(fill_value)
    elif strategy == "drop":
        out = out.dropna(subset=target_cols).reset_index(drop=True)
    else:
        raise ValueError(f"未知的缺失值处理策略: {strategy}")

    logger.info("缺失值处理完成: strategy=%s, 列数=%d", strategy, len(target_cols))
    return out


def handle_outliers(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: Literal["cap", "remove"] = "cap",
    n_sigma: float = 5.0,
) -> pd.DataFrame:
    """对数值列异常值进行盖帽或剔除处理。

    算法: 使用 mu +- n_sigma 作为上下界（n_sigma 默认 5）。
    处理逻辑在训练集 fit 后可保存上下界，应用于测试集。

    Args:
        df: 输入数据
        columns: 需要处理的数值列，默认全数据
        method: 'cap' 盖帽(替换为边界值), 'remove' 剔除
        n_sigma: 标准差倍数

    Returns:
        处理后的 DataFrame 与边界值 dict
    """
    out = df.copy()

    target_cols = columns or out.select_dtypes(include=[np.number]).columns.tolist()

    for c in target_cols:
        if c not in out.columns or not pd.api.types.is_numeric_dtype(out[c]):
            continue
        col = out[c].dropna()
        mu, sd = col.mean(), col.std()
        lower = mu - n_sigma * sd
        upper = mu + n_sigma * sd

        if method == "cap":
            out[c] = out[c].clip(lower=lower, upper=upper)
        elif method == "remove":
            mask = (out[c] >= lower) & (out[c] <= upper) | out[c].isna()
            out = out.loc[mask].reset_index(drop=True)
        else:
            raise ValueError(f"未知方法: {method}")

    logger.info("异常值处理完成: method=%s, sigma=%s, n=%d", method, n_sigma, len(target_cols))
    return out