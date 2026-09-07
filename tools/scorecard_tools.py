"""评分卡生成与应用工具集。

核心: 包装 scorecardpy.scorecard 与 scorecard_ply，并显式锚定基准分 600、PDO 20。
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

try:
    import scorecardpy as sc
except ImportError as e:
    raise ImportError("未安装 scorecardpy") from e

logger = logging.getLogger(__name__)


def build_scorecard(
    bins: dict[str, pd.DataFrame],
    model: Any,
    xcolumns: list[str] | None = None,
    base_score: int = 600,
    pdo: int = 20,
    base_odds: float = 50,
    double_odds: float = 1,
    digits: int = 0,
) -> dict[str, pd.DataFrame]:
    """生成评分卡。

    默认德勤标准: base_score=600, pdo=20, base_odds=50。

    说明:
        scorecardpy 0.1.9.x 接口:
            scorecard(bins, model, xcolumns, points0, odds0, pdo, ...)
        其中 odds0 = base_odds (odds = p/(1-p))
        PDO (Points to Double the Odds) 即 pdo。

    Args:
        bins: woebin() 返回的分箱字典
        model: 训练好的 LogisticRegression
        xcolumns: 进入模型的特征列名(必须与 model 训练时的列对齐)
        base_score: 基准分
        pdo: Points to Double Odds
        base_odds: 基准 odds (p/(1-p))
        double_odds: 为兼容参数名保留 (scorecardpy 0.1.9.x 不使用)
        digits: 分数小数位数

    Returns:
        dict[var_name, 单变量评分卡 DataFrame]
    """
    logger.info(
        "评分卡构建: base=%d, pdo=%d, odds=%g",
        base_score, pdo, base_odds,
    )
    if xcolumns is None:
        raise ValueError("xcolumns 必须显式传入 (与模型训练时的 WOE 列对齐)")
    card = sc.scorecard(
        bins,
        model,
        xcolumns=xcolumns,
        points0=base_score,
        odds0=base_odds,
        pdo=pdo,
        digits=digits,
    )
    logger.info("评分卡构建完成: %d 个变量", len(card))
    return card


def scorecard_ply(
    df: pd.DataFrame,
    card: dict[str, pd.DataFrame],
    only_total_score: bool = False,
    replace_blank_na: bool = True,
) -> pd.DataFrame:
    """对单笔或多笔数据应用评分卡打分。

    Args:
        df: 待打分原始数据
        card: build_scorecard() 生成的评分卡
        only_total_score: True 仅返回总评分列, False 包含各变量分
        replace_blank_na: 是否将空白视为 NA

    Returns:
        包含 score 列(及各变量 score_x)的 DataFrame

    Example:
        >>> score_df = scorecard_ply(applicant_df, card)
        >>> print(score_df["score"])
    """
    out = sc.scorecard_ply(
        df,
        card,
        only_total_score=only_total_score,
        print_step=0,
        replace_blank_na=replace_blank_na,
    )
    logger.info("评分卡应用: %d 行, total_score 均值=%.2f", len(out), out["score"].mean())
    return out