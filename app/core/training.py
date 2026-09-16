"""app.core.training —— 不依赖 LangGraph HITL 的一键训练流程。

直接调用 tools 函数，适合 Streamlit UI 的无阻塞体验。
数值计算完全由确定性 Python 工具完成，与 Phase 2 LangGraph 节点逻辑保持一致。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from tools import (
    build_scorecard,
    compute_psi,
    evaluate_performance,
    explore_data,
    guardrail_check,
    handle_missing,
    handle_outliers,
    load_data,
    model_predict,
    model_train,
    scorecard_ply,
    split_dataset,
    var_filter,
    woebin,
    woebin_ply,
)


class TrainingResult(dict):
    """训练结果容器，支持点号访问与 dict 访问。"""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


def _numeric_only(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """保留数值列（去除 target 与未成功 WOE 化的字符串列）。"""
    cols = [c for c in df.columns if c != target and df[c].dtype.kind in "iuf"]
    return df[cols].astype(float)


def _maybe_convert_target(train: pd.DataFrame, test: pd.DataFrame, target: str) -> None:
    """若 target 为 'good'/'bad' 字符串，则原地转换为 0/1。"""
    if train[target].dtype == object or str(train[target].dtype) == "str":
        unique_vals = train[target].dropna().unique().tolist()
        if set(map(str, unique_vals)) == {"good", "bad"}:
            train[target] = (train[target] == "good").astype(int)
            test[target] = (test[target] == "good").astype(int)


def _gini(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Gini = 2*AUC - 1（不需要 sklearn，更轻）。"""
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, y_proba)
    return float(2 * auc - 1)


def _ks_stat(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """KS 统计量：|TPR - FPR| 的最大值。"""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(np.abs(tpr - fpr)))


def _prepare_xy(
    df: pd.DataFrame,
    target: str,
    cfg: dict[str, Any],
    bins: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """对单个时间窗的数据执行与主流程一致的预处理+分箱+WOE。

    与 run_training_pipeline 的步骤 3-5 共享逻辑：剔除 ID → 数值缺失填充 →
    异常值截尾 → 分类 NaN 填 'Missing' → var_filter → woebin → woebin_ply。

    Args:
        bins: 若提供（来自上一窗），复用同一份分箱；否则基于本窗数据分箱。
    Returns:
        (X_woe, y, bins_used)
    """
    y = df[target].astype(int).copy()
    feat = df.drop(columns=[target])
    id_cols = {"Symbol", "ShortName", "EndDate"}
    feat = feat.drop(columns=[c for c in id_cols if c in feat.columns])

    feat = handle_missing(feat, strategy=cfg.get("missing_strategy", "median"))
    feat = handle_outliers(feat, method=cfg.get("outlier_method", "cap"), n_sigma=cfg.get("outlier_sigma", 5.0))

    for c in feat.columns:
        if feat[c].dtype == object or str(feat[c].dtype) == "str":
            feat[c] = feat[c].fillna("Missing").astype(str)

    feat[target] = y.values
    train_filtered, _ = var_filter(
        feat,
        target=target,
        iv_threshold=cfg.get("iv_threshold", 0.02),
        missing_threshold=cfg.get("missing_threshold", 0.5),
        identical_threshold=cfg.get("identical_threshold", 0.95),
    )
    kept = [c for c in train_filtered.columns if c != target]
    y = train_filtered[target]
    train_filtered = train_filtered[kept + [target]]

    if bins is None:
        bins = woebin(
            train_filtered,
            target=target,
            max_bins=cfg.get("max_bins", 8),
            min_bin_size=cfg.get("min_bin_size", 0.05),
            method=cfg.get("binning_method", "tree"),
            parallel=False,
        )
    x_woe = _numeric_only(woebin_ply(train_filtered, bins), target)
    return x_woe, y, bins


def _run_expanding_window_cv(
    df: pd.DataFrame,
    target: str,
    year_col: str,
    cfg: dict[str, Any],
    min_train_years: int = 3,
    resume_from_csv: str | Path | None = None,
    progress_callback=None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Expanding Window 交叉验证：train_year<=t-1，test_year==t。

    Args:
        df: 完整面板（必须包含 year 列）
        target: 目标列名
        year_col: 年份列名（默认 'year'）
        min_train_years: 最少用前 N 年训练才启动验证
        resume_from_csv: 断点续跑。若提供且文件存在，读取其中 auc 非空的 fold 直接复用，
            跳过重算（用于长耗时任务被打断后继续）。

    Returns:
        (per_fold_df, mean_metrics)
        - per_fold_df: 列 = [year, n_train, n_test, n_test_default, default_rate, auc, ks, gini]
        - mean_metrics: 仅 test 期的均值，用于主结果展示
    """
    years = sorted(df[year_col].dropna().unique().astype(int).tolist())
    if len(years) < min_train_years + 1:
        raise ValueError(f"可用年份 {years} 不足以执行 Expanding Window CV（需 >= {min_train_years + 1}）")

    shared_bins: dict | None = None
    shared_cols: list[str] | None = None

    rows: list[dict] = []
    done_years: set[int] = set()
    if resume_from_csv is not None and Path(resume_from_csv).exists():
        try:
            prev = pd.read_csv(resume_from_csv)
            for _, r in prev.iterrows():
                if "auc" in prev.columns and pd.notna(r.get("auc")):
                    rows.append(r.to_dict())
                    done_years.add(int(r["year"]))
            if done_years:
                print(f"[resume] 从 {resume_from_csv} 复用已完成 fold: {sorted(done_years)}", flush=True)
        except Exception as _e:
            print(f"[resume] 读取失败，忽略: {_e}", flush=True)

    for t in years[min_train_years:]:
        if t in done_years:
            print(f"[resume] 跳过已完成 fold year={t}", flush=True)
            continue
        train_df = df[df[year_col] <= t - 1]
        test_df = df[df[year_col] == t]
        if len(test_df) == 0 or len(train_df) == 0:
            continue
        if progress_callback is not None:
            progress_callback(
                stage="cv_fold",
                detail=f"CV fold year={t} · train={len(train_df)} / test={len(test_df)} · var_filter + woebin",
            )

        # 优先复用首窗 bins（节省 ~3 分钟/fold）；若失败则回退到 per-fold bins
        # （常见原因：测试集出现训练集未见的分类值，woebin_ply 解析失败）
        use_shared = shared_bins is not None
        try:
            if use_shared:
                x_tr, y_tr, _ = _prepare_xy(train_df, target, cfg, bins=shared_bins)
                x_te, y_te, _ = _prepare_xy(test_df, target, cfg, bins=shared_bins)
                x_te = x_te.reindex(columns=shared_cols, fill_value=0.0)
                x_tr = x_tr.reindex(columns=shared_cols, fill_value=0.0)
            else:
                x_tr, y_tr, bins_used = _prepare_xy(train_df, target, cfg, bins=None)
                x_te, y_te, _ = _prepare_xy(test_df, target, cfg, bins=bins_used)
                shared_bins = bins_used
                shared_cols = list(x_tr.columns)
                # 仅在使用 per-fold 时对齐列
                x_te = x_te.reindex(columns=shared_cols, fill_value=0.0)
        except Exception:
            # 回退：per-fold 重新计算 bins（牺牲速度保正确性）
            try:
                x_tr, y_tr, bins_used = _prepare_xy(train_df, target, cfg, bins=None)
                x_te, y_te, _ = _prepare_xy(test_df, target, cfg, bins=bins_used)
                if shared_cols is None:
                    shared_cols = list(x_tr.columns)
                    shared_bins = bins_used
                x_te = x_te.reindex(columns=shared_cols, fill_value=0.0)
                x_tr = x_tr.reindex(columns=shared_cols, fill_value=0.0)
            except Exception as e2:
                import traceback as _tb
                rows.append({
                    "year": t, "n_train": len(train_df), "n_test": len(test_df),
                    "n_test_default": int(test_df[target].sum()) if target in test_df.columns else 0,
                    "default_rate": float(test_df[target].mean()) if target in test_df.columns else 0.0,
                    "auc": np.nan, "ks": np.nan, "gini": np.nan,
                    "error": f"FALLBACK FAIL: {type(e2).__name__}: {e2}\n{_tb.format_exc(limit=2)}",
                })
                continue

        if y_tr.nunique() < 2 or y_te.nunique() < 2:
            rows.append({
                "year": t, "n_train": len(train_df), "n_test": len(test_df),
                "n_test_default": int(y_te.sum()),
                "default_rate": float(y_te.mean()),
                "auc": np.nan, "ks": np.nan, "gini": np.nan,
                "error": "单类别，跳过",
            })
            continue

        C = 1.0 / cfg.get("regularization", 0.01)
        model = LogisticRegression(C=C, max_iter=cfg.get("max_iter", 1000), solver="lbfgs")
        model.fit(x_tr.values, y_tr.values)
        proba = model.predict_proba(x_te.values)[:, 1]
        auc = float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(y_te.values, proba))
        ks = _ks_stat(y_te.values, proba)
        gini = 2 * auc - 1
        rows.append({
            "year": t, "n_train": len(train_df), "n_test": len(test_df),
            "n_test_default": int(y_te.sum()),
            "default_rate": float(y_te.mean()),
            "auc": auc, "ks": ks, "gini": gini,
            "error": "",
        })

    per_fold = pd.DataFrame(rows)
    valid = per_fold.dropna(subset=["auc"])
    if len(valid) == 0:
        mean_metrics = {"auc": 0.0, "ks": 0.0, "gini": 0.0, "n_folds": 0}
    else:
        mean_metrics = {
            "auc": float(valid["auc"].mean()),
            "ks": float(valid["ks"].mean()),
            "gini": float(valid["gini"].mean()),
            "n_folds": int(len(valid)),
        }
    return per_fold, mean_metrics


def _coefficient_importance(model: Any, feature_names: list[str]) -> pd.DataFrame:
    """逻辑回归 WOE 模型下的"简化 SHAP"：用 |coef| 排序特征重要性。

    对 LR + WOE 而言，每个 WOE 特征对 log-odds 的边际贡献就是 coef * x_woe，
    因此 |coef| 等价于"每变 1 单位 WOE，违约 log-odds 变化幅度"的相对权重。
    返回 DataFrame: feature, coef, abs_coef, rank（abs_coef 降序）。
    """
    coef = np.asarray(model.coef_).reshape(-1)
    out = pd.DataFrame({
        "feature": feature_names,
        "coef": coef,
        "abs_coef": np.abs(coef),
    })
    out = out.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out


def _train_lgbm_branch(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    cfg: dict[str, Any],
) -> tuple[Any, np.ndarray, np.ndarray, dict[str, float], dict[str, float], pd.DataFrame | None]:
    """LGBM 对照模型 + SHAP 解释（不依赖 WOE，可直接消费数值/类别特征）。

    Returns:
        (model, train_proba, test_proba, train_metrics, test_metrics, shap_importance_df)
        shap_importance_df: 含 feature / mean_abs_shap / rank，按 mean_abs_shap 降序
    """
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise RuntimeError("未安装 lightgbm，请先 pip install lightgbm") from e

    cat_cols = [c for c in train.columns if c != target and (train[c].dtype == object or str(train[c].dtype) == "str")]
    num_cols = [c for c in train.columns if c != target and c not in cat_cols]

    X_tr = train.drop(columns=[target]).copy()
    X_te = test.drop(columns=[target]).copy()
    # 类别列转 category dtype，LGBM 原生支持
    for c in cat_cols:
        X_tr[c] = X_tr[c].astype("category")
        # 让测试集类别与训练集对齐
        X_te[c] = pd.Categorical(X_te[c], categories=X_tr[c].cat.categories)

    y_tr = train[target].astype(int).values
    y_te = test[target].astype(int).values

    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": cfg.get("lgbm_lr", 0.05),
        "num_leaves": cfg.get("lgbm_num_leaves", 31),
        "min_data_in_leaf": cfg.get("lgbm_min_data_in_leaf", 50),
        "feature_fraction": cfg.get("lgbm_feature_fraction", 0.8),
        "bagging_fraction": cfg.get("lgbm_bagging_fraction", 0.8),
        "bagging_freq": 5,
        "scale_pos_weight": pos_weight,
        "verbose": -1,
    }
    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
    dvalid = lgb.Dataset(X_te, label=y_te, categorical_feature=cat_cols, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=cfg.get("lgbm_num_boost_round", 500),
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(cfg.get("lgbm_early_stopping", 30), verbose=False)],
    )
    train_proba = model.predict(X_tr, num_iteration=model.best_iteration)
    test_proba = model.predict(X_te, num_iteration=model.best_iteration)

    train_metrics = evaluate_performance(y_tr, train_proba)
    test_metrics = evaluate_performance(y_te, test_proba)

    # SHAP：tree explainer，速度快
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        # 用 1000 行子样本计算 SHAP，避免 30k 行太慢
        sample_n = min(1000, len(X_te))
        shap_vals = explainer.shap_values(X_te.iloc[:sample_n])
        # shap_vals 可能是 list[neg, pos] 或直接 2D 矩阵；统一取正类
        if isinstance(shap_vals, list):
            sv = shap_vals[1]
        else:
            sv = shap_vals
        mean_abs = np.abs(sv).mean(axis=0)
        shap_imp = pd.DataFrame({
            "feature": X_te.columns.tolist(),
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        shap_imp["rank"] = shap_imp.index + 1
    except Exception as e:
        shap_imp = None
        import warnings
        warnings.warn(f"SHAP 计算失败: {e}", RuntimeWarning)

    return model, train_proba, test_proba, train_metrics, test_metrics, shap_imp


def run_training_pipeline(
    csv_path: str | Path,
    target: str,
    config: dict[str, Any] | None = None,
    progress_callback=None,
) -> TrainingResult:
    """跑完整训练流程并返回结果。

    Args:
        csv_path: 训练数据路径（CSV / Excel）
        target: 目标列名
        config: 覆盖默认配置的字典，键与 default_config.yaml 对应
        progress_callback: 可选回调，签名 (stage: str, detail: str) -> None。
            stage 取值: "load_data", "eda", "cv_start", "cv_fold", "cv_end",
                        "split", "preprocess", "var_filter", "woebin",
                        "train_lr", "evaluate", "scorecard", "lgbm_start",
                        "lgbm_end", "done"。调用方负责更新 UI 进度条与文字。

    Returns:
        TrainingResult 包含 eda / metrics / scorecard / scored_test / guardrail / model / bins
    """
    cfg = dict(config or {})

    def _emit(stage: str, detail: str = "") -> None:
        if progress_callback is not None:
            progress_callback(stage=stage, detail=detail)

    # 1. 加载与探索
    _emit("load_data", f"读取 {Path(csv_path).name} ...")
    df = load_data(
        csv_path,
        target=target,
        encoding=cfg.get("encoding", "utf-8"),
        sep=cfg.get("sep"),
    )
    _emit("eda", f"加载完成 · {len(df)} 行 × {df.shape[1]} 列 · 目标 {target}")
    eda = explore_data(df, target)

    # 1.5 【v2】可选：Expanding Window 时间序列 CV（仅当 time_split=True 且存在 year_col）
    cv_per_fold = None
    cv_mean = None
    if cfg.get("time_split", False):
        year_col = cfg.get("year_col", "year")
        if year_col in df.columns:
            _emit("cv_start", f"时序切分开启 · year_col={year_col}")
            try:
                cv_per_fold, cv_mean = _run_expanding_window_cv(
                    df, target=target, year_col=year_col, cfg=cfg,
                    min_train_years=cfg.get("min_train_years", 3),
                    progress_callback=progress_callback,
                )
            except Exception as e:
                cv_mean = {"error": f"{type(e).__name__}: {e}"}
            _emit("cv_end", f"CV 完成 · mean_auc={cv_mean.get('auc', 0):.3f}" if "error" not in cv_mean else f"CV 失败: {cv_mean.get('error', '')}")
        else:
            _emit("cv_start", f"跳过 · year 列 '{year_col}' 不在数据中")

    # 2. 划分
    _emit("split", "随机划分训练/测试集 ...")
    train, test = split_dataset(
        df,
        target=target,
        test_size=cfg.get("test_size", 0.3),
        random_state=cfg.get("random_state", 42),
    )
    _maybe_convert_target(train, test, target)

    # 3. 预处理（目标列不参与缺失值填充与异常值截尾，避免 target 被污染）
    missing_strategy = cfg.get("missing_strategy", "median")
    outlier_method = cfg.get("outlier_method", "cap")
    outlier_sigma = cfg.get("outlier_sigma", 5.0)

    y_train = train[target].copy()
    y_test = test[target].copy()
    train_features = train.drop(columns=[target])
    test_features = test.drop(columns=[target])

    train_features = handle_missing(train_features, strategy=missing_strategy)
    test_features = handle_missing(test_features, strategy=missing_strategy)
    train_features = handle_outliers(train_features, method=outlier_method, n_sigma=outlier_sigma)
    test_features = handle_outliers(test_features, method=outlier_method, n_sigma=outlier_sigma)

    train = train_features.copy()
    train[target] = y_train.values
    test = test_features.copy()
    test[target] = y_test.values

    # 3.5 删除纯标识列与时序索引列（ID、名称、日期、year 不参与建模）
    id_cols = {"Symbol", "\ufeffSymbol", "ShortName", "EndDate", "year"}
    train = train.drop(columns=[c for c in id_cols if c in train.columns])
    test = test.drop(columns=[c for c in id_cols if c in test.columns])

    # 3.6 对分类变量用 "Missing" 填充缺失，避免 scorecardpy 在 pandas 3.0 下解析 NaN 崩溃
    for c in train.columns:
        if c == target:
            continue
        if train[c].dtype == object or str(train[c].dtype) == "str":
            train[c] = train[c].fillna("Missing").astype(str)
            test[c] = test[c].fillna("Missing").astype(str)

    # 4. 变量筛选
    _emit("var_filter", f"变量筛选 · IV 阈值 {cfg.get('iv_threshold', 0.02)}")
    train_filtered, _ = var_filter(
        train,
        target=target,
        iv_threshold=cfg.get("iv_threshold", 0.02),
        missing_threshold=cfg.get("missing_threshold", 0.5),
        identical_threshold=cfg.get("identical_threshold", 0.95),
    )
    kept = [c for c in train_filtered.columns if c != target]
    test = test[[target] + kept].copy()
    train = train_filtered

    # 5. WOE 分箱与转换
    _emit("woebin", f"WOE 分箱 · 最大分箱数 {cfg.get('max_bins', 8)}")
    bins = woebin(
        train,
        target=target,
        max_bins=cfg.get("max_bins", 8),
        min_bin_size=cfg.get("min_bin_size", 0.05),
        method=cfg.get("binning_method", "tree"),
        parallel=False,
    )
    train_woe = _numeric_only(woebin_ply(train, bins), target)
    test_woe = _numeric_only(woebin_ply(test, bins), target)

    # 6. 训练
    _emit("train_lr", "逻辑回归训练 ...")
    C = 1.0 / cfg.get("regularization", 0.01)
    model = model_train(
        train_woe,
        train[target],
        C=C,
        max_iter=cfg.get("max_iter", 1000),
    )

    # 7. 预测与评估
    _emit("evaluate", "预测与评估 ...")
    train_proba = model_predict(model, train_woe, as_prob=True)
    test_proba = model_predict(model, test_woe, as_prob=True)
    train_metrics = evaluate_performance(train[target].values, train_proba)
    test_metrics = evaluate_performance(test[target].values, test_proba)
    metrics: dict[str, Any] = {"train": train_metrics, "test": test_metrics, **test_metrics}

    # 8. 评分卡与打分
    _emit("scorecard", f"生成评分卡 · 基准分 {cfg.get('base_score', 600)} / PDO {cfg.get('pdo', 20)}")
    xcolumns = list(train_woe.columns)
    card = build_scorecard(
        bins=bins,
        model=model,
        xcolumns=xcolumns,
        base_score=cfg.get("base_score", 600),
        pdo=cfg.get("pdo", 20),
        base_odds=cfg.get("base_odds", 50),
        double_odds=cfg.get("double_odds", 1),
    )
    scored_test = scorecard_ply(test, card, only_total_score=True)

    # 9. PSI 与护栏
    psi_val = compute_psi(np.asarray(train_proba), np.asarray(test_proba))
    metrics["psi"] = psi_val
    guardrail = guardrail_check(
        metrics=metrics,
        reference_dist=np.asarray(train_proba),
        current_dist=np.asarray(test_proba),
        min_ks=cfg.get("min_ks", 0.3),
        min_auc=cfg.get("min_auc", 0.7),
        max_psi=cfg.get("max_psi", 0.1),
        require_human_review=False,  # UI 层自己展示，不阻塞
    )

    # 10. 【v2】LGBM 对照模型 + SHAP（可选，cfg["use_lgbm"]=True 启用）
    lgbm_model = None
    lgbm_metrics: dict[str, Any] = {}
    lgbm_shap_importance = None
    if cfg.get("use_lgbm", False):
        _emit("lgbm_start", "LGBM 对照模型 + SHAP 训练 ...")
        try:
            (lgbm_model,
             _lgbm_train_proba, _lgbm_test_proba,
             lgbm_train_metrics, lgbm_test_metrics,
             lgbm_shap_importance) = _train_lgbm_branch(train, test, target, cfg)
            lgbm_metrics = {"train": lgbm_train_metrics, "test": lgbm_test_metrics, **lgbm_test_metrics}
            _emit("lgbm_end", f"LGBM AUC={lgbm_metrics.get('auc', 0):.3f}")
        except Exception as e:
            lgbm_metrics = {"error": f"{type(e).__name__}: {e}"}
            _emit("lgbm_end", f"LGBM 失败: {e}")

    _emit("done", "训练完成")

    return TrainingResult(
        eda=eda,
        metrics=metrics,
        scorecard=card,
        scored_test=scored_test,
        guardrail=guardrail,
        model=model,
        bins=bins,
        train_woe=train_woe,
        test_woe=test_woe,
        train_proba=train_proba,
        test_proba=test_proba,
        feature_importance=_coefficient_importance(model, xcolumns),
        cv_per_fold=cv_per_fold,
        cv_mean=cv_mean,
        lgbm_model=lgbm_model,
        lgbm_metrics=lgbm_metrics,
        lgbm_shap_importance=lgbm_shap_importance,
    )
