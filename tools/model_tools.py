"""模型训练、预测与评估工具集。"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

logger = logging.getLogger(__name__)


def model_train(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    C: float = 1.0,
    penalty: str = "l2",
    max_iter: int = 1000,
    class_weight: str | dict | None = "balanced",
    random_state: int = 42,
) -> LogisticRegression:
    """训练逻辑回归模型。

    Args:
        X_train: 训练特征 (WOE 转换后)
        y_train: 训练标签 (0/1)
        C: 正则化强度倒数
        penalty: 'l1' / 'l2' / 'elasticnet'
        max_iter: 最大迭代次数
        class_weight: 类别权重
        random_state: 随机种子

    Returns:
        训练好的 LogisticRegression 模型
    """
    solver = "liblinear" if penalty in ("l1", "l2") else "saga"
    model = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    logger.info(
        "模型训练完成: n_features=%d, C=%.2f, n_iter=%d",
        X_train.shape[1], C, model.n_iter_[0],
    )
    return model


def coefficient_importance(
    model: Any,
    feature_names: list[str] | pd.Index | None = None,
) -> pd.DataFrame:
    """逻辑回归 + WOE 场景下的「简化 SHAP」：用 |coef| 排序特征重要性。

    原理（确定性，无随机性、不依赖 shap 库）：
        WOE 编码后每个特征的取值在 [-k, k] 量纲可比范围内，
        对 log-odds 的边际贡献为 `coef * woe`，因此 |coef| 就代表
        「该变量 WOE 变动 1 单位时违约 log-odds 的变化幅度」，可作为相对权重排序。

    Args:
        model: 已训练的逻辑回归（需有 `coef_`）
        feature_names: 特征名；长度不匹配时用 f0..fn 兜底

    Returns:
        DataFrame: feature / coef / abs_coef / rank（abs_coef 降序，rank 从 1 起）
    """
    coef = np.asarray(model.coef_).reshape(-1)
    names = list(feature_names) if feature_names is not None else []
    if len(names) != len(coef):
        names = [f"f{i}" for i in range(len(coef))]
    out = pd.DataFrame({
        "feature": names,
        "coef": coef,
        "abs_coef": np.abs(coef),
    })
    out = out.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out


def model_predict(
    model: LogisticRegression,
    X: pd.DataFrame | np.ndarray,
    as_prob: bool = True,
) -> np.ndarray:
    """使用训练好的模型预测。

    Args:
        model: 训练好的模型
        X: 待预测特征
        as_prob: True 返回 bad 概率, False 返回类别

    Returns:
        ndarray
    """
    if as_prob:
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def stepwise_selection(
    X: pd.DataFrame,
    y: pd.Series,
    direction: str = "backward",
    criterion: str = "aic",
    p_enter: float = 0.05,
    p_remove: float = 0.1,
) -> list[str]:
    """基于 statsmodels 的逐步回归变量筛选。

    仅当 X 已被 WOE 化且样本量较大时使用，可在 stage 中视情况关闭。

    Args:
        X: 特征 DataFrame
        y: 标签 Series
        direction: 'forward' / 'backward' / 'both'
        criterion: 'aic' / 'bic'
        p_enter: 进入门槛
        p_remove: 剔除门槛

    Returns:
        保留的变量名列表
    """
    try:
        import statsmodels.api as sm
    except ImportError as e:
        raise ImportError("请安装 statsmodels: pip install statsmodels") from e

    logger.info("逐步回归开始: direction=%s, criterion=%s", direction, criterion)
    if direction == "backward":
        included = list(X.columns)
        while True:
            X_curr = sm.add_constant(X[included])
            pvals = sm.Logit(y, X_curr).fit(disp=0).pvalues.iloc[1:]
            worst = pvals.idxmax()
            if pvals[worst] > p_remove:
                included.remove(worst)
            else:
                break
        selected = included

    elif direction == "forward":
        remaining = list(X.columns)
        selected = []
        while remaining:
            best_p, best_v = 1.0, None
            for v in remaining:
                X_try = sm.add_constant(X[selected + [v]])
                p = sm.Logit(y, X_try).fit(disp=0).pvalues.iloc[-1]
                if p < best_p:
                    best_p, best_v = p, v
            if best_p < p_enter and best_v:
                selected.append(best_v)
                remaining.remove(best_v)
            else:
                break
    else:
        raise ValueError(f"暂不支持的 direction: {direction}")

    logger.info("逐步回归完成: 保留 %d 个变量", len(selected))
    return selected


def evaluate_performance(
    y_true: np.ndarray | pd.Series,
    y_pred_proba: np.ndarray,
) -> dict[str, float]:
    """计算 KS、AUC、Gini 指标。

    KS = max(TPR - FPR), Gini = 2*AUC - 1。

    Args:
        y_true: 真实标签 (0/1)
        y_pred_proba: 预测为 bad/1 的概率

    Returns:
        dict 包含 'ks', 'auc', 'gini'
    """
    auc = float(roc_auc_score(y_true, y_pred_proba))
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    ks = float(np.max(tpr - fpr))
    gini = float(2 * auc - 1)

    metrics = {"auc": auc, "ks": ks, "gini": gini}
    logger.info("模型评估: %s", metrics)
    return metrics