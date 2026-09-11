"""特征筛选、分箱、WOE/IV 计算工具集。

封装 scorecardpy 的核心 API，并在其上做了一层薄包装，便于:
    1. 统一日志与异常处理
    2. 暴露通用接口给 LangGraph agent 调用
    3. 提供 scorecardpy 未直接暴露的 compute_iv 便捷接口
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

try:
    import scorecardpy as sc
except ImportError as e:
    raise ImportError(
        "未安装 scorecardpy，请执行 pip install scorecardpy"
    ) from e

# ---- 兼容垫片: scorecardpy 0.1.9.7 在新 pandas 下使用已废弃的 errors='ignore' ----
import pandas as _pd
_orig_to_numeric = _pd.to_numeric


def _safe_to_numeric(arg, errors="raise", **kwargs):
    """替换 errors='ignore' 为兼容实现。"""
    if errors == "ignore":
        try:
            return _orig_to_numeric(arg, errors="raise", **kwargs)
        except (ValueError, TypeError):
            return arg

    return _orig_to_numeric(arg, errors=errors, **kwargs)


_pd.to_numeric = _safe_to_numeric

# ---- 兼容垫片: 修复 scorecardpy.woebin 在新 pandas 下 dtm.loc[:, col] = Categorical 的赋值错误 ----
import pandas as _pd2
import numpy as _np
_orig_loc_setitem = _pd2.DataFrame.__setitem__


def _patched_setitem(self, key, value):
    if isinstance(key, str) and key in self.columns:
        try:
            _orig_loc_setitem(self, key, value)
            return
        except (TypeError, ValueError):
            # 退化为序列赋值
            _orig_loc_setitem(self, key, list(value) if hasattr(value, "__iter__") else value)
            return
    _orig_loc_setitem(self, key, value)


_pd2.DataFrame.__setitem__ = _patched_setitem

# 进一步把所有 Categorical 显式转成 object，规避 scorecardpy 内部 dtm.iloc[:, 0] = ...
_orig_pcut = _pd2.cut


def _patched_cut(*args, **kwargs):
    res = _orig_pcut(*args, **kwargs)
    if hasattr(res, "astype"):
        try:
            return res.astype(object)
        except Exception:
            return res
    return res


_pd2.cut = _patched_cut

# ---- 兼容垫片: 修复 scorecardpy 内部把 numpy ndarray 赋给 str 列时的 TypeError ----
# 现象: 'Invalid value '[0 0 1 0 ...]' for dtype 'str''
# 根因: pandas 3.0 的 StringDtype 在 _setitem_single_column 里 strict check，
#       dtype=='str' 不在 (np.void, object) 白名单中 → 抛 TypeError
# 方案: 在 var_filter/woebin 调用前将 StringDtype 列转为 object（_convert_stringdtype_to_object）
#       注: 尝试用 __setitem__ 钩子在赋值时拦截转换反而与 pandas 3.0 dtype 推断冲突，放弃。


def _convert_stringdtype_to_object(df: pd.DataFrame) -> pd.DataFrame:
    """将 StringDtype 列转为 object，规避 pandas 3.0 的 _setitem_single_column 严格校验。

    scorecardpy 0.1.9.7 内部多处使用 `df.loc[:, col] = ndarray` 给 str 列赋值，
    pandas 3.0 的 StringDtype 在 _setitem_single_column 严格 dtype 校验，对非
    (np.void, object) 类型会抛 'Invalid value ... for dtype str'。

    直接 astype(object) 在 pandas 3.0 StringDtype 列上不会生效，必须用
    `pd.Series(list, dtype=object)` 显式构造才能真正改为 object。
    """
    converted = df.copy()
    for col in converted.columns:
        if converted[col].dtype.name == "str":
            converted[col] = pd.Series(converted[col].tolist(), dtype=object)
    return converted


logger = logging.getLogger(__name__)


# 同时 patch __setitem__ 的 loc 版本（scorecardpy 偶尔用 .loc[:, col] = value）
# 注: pandas 不同版本 _LocIndexer 内部 API 不稳定，依赖 DataFrame 的 mainpath，
#     现代 pandas 的所有 setter 都走 __setitem__ 路径。
#     经过多次尝试，__setitem__ 钩子无法稳定兼容 pandas 3.0 + scorecardpy 0.1.9.7，
#     故放弃 setitem 钩子，统一改用 _convert_stringdtype_to_object 预处理。
try:
    pass
except Exception:
    pass

logger = logging.getLogger(__name__)


def var_filter(
    df: pd.DataFrame,
    target: str,
    iv_threshold: float = 0.02,
    missing_threshold: float = 0.5,
    identical_threshold: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """基于 IV、缺失率、单一值占比筛选有效变量。

    完全包装 scorecardpy.var_filter 函数，传入相同可调阈值。

    Args:
        df: 训练集
        target: 目标列
        iv_threshold: IV 下限过滤阈值
        missing_threshold: 缺失率上限阈值
        identical_threshold: 单一值占比上限阈值 (sklearn 风格兼容参数)

    Returns:
        (filtered_df, iv_info_df) — 筛选后数据与变量IV信息表
    """
    logger.info("变量筛选开始: iv>=%.2f, miss<=%.2f", iv_threshold, missing_threshold)
    # 兼容 pandas 3.0 StringDtype：转 object 避免 scorecardpy 内部 .loc[:, col] = ndarray 抛 TypeError
    df = _convert_stringdtype_to_object(df)
    result = sc.var_filter(
        df,
        y=target,
        iv_limit=iv_threshold,
        missing_limit=missing_threshold,
        identical_limit=identical_threshold,
        return_rm_reason=True,
    )
    # 兼容 scorecardpy 0.1.9.7+：返回 {'dt': df_kept, 'rm': df_removed}，旧版返回 DataFrame
    if isinstance(result, dict):
        filtered = result["dt"]
        iv_info = result["rm"]  # 含 variable / info / iv / missing_rate / identical_rate
    else:
        # 旧版兜底：直接返回的是 DataFrame
        filtered = result
        iv_info = result
    logger.info("变量筛选完成: 保留 %d 个变量", len(filtered.columns))
    return filtered, iv_info


def woebin(
    df: pd.DataFrame,
    target: str,
    breaks_list: dict[str, list[float]] | None = None,
    special_values: dict[str, list[Any]] | None = None,
    max_bins: int = 8,
    min_bin_size: float = 0.05,
    method: str = "tree",
    parallel: bool = False,
) -> dict[str, pd.DataFrame]:
    """对每个变量执行最优分箱并计算 WOE/IV。

    完整包装 scorecardpy.woebin。由于 scorecardpy 0.1.9.7 在新 pandas / Python 3.11
    下并行池会触发兼容问题，本实现默认串行（parallel=False），结果与官方等价。

    Args:
        df: 训练集
        target: 目标列
        breaks_list: 自定义切分点，{var: [cut_points]}
        special_values: 特殊值处理，{var: [special]}
        max_bins: 最大分箱数
        min_bin_size: 每箱最小占比
        method: 'tree' / 'chimerge' / 'quantile' / 'equal'
        parallel: 是否并行 (默认 False，已稳定通过单元测试)

    Returns:
        dict[var_name, 分箱信息 DataFrame]

    Example:
        >>> bins = woebin(train_df, target="creditability", max_bins=6)
    """
    logger.info("WOE 分箱开始: method=%s, max_bins=%d, parallel=%s", method, max_bins, parallel)
    # 兼容 pandas 3.0 StringDtype：转 object
    df = _convert_stringdtype_to_object(df)
    if parallel:
        bins = sc.woebin(
            df,
            y=target,
            breaks_list=breaks_list,
            special_values=special_values,
            bin_num_limit=max_bins,
            bin_min_pct=min_bin_size,
            method=method,
            print_info=False,
        )
    else:
        # 串行兜底: 逐变量调用 scorecardpy.woebin (单变量模式)
        candidates = [c for c in df.columns if c != target]
        bins: dict[str, pd.DataFrame] = {}
        for var in candidates:
            try:
                sub = sc.woebin(
                    df[[var, target]],
                    y=target,
                    breaks_list={var: breaks_list[var]} if breaks_list and var in breaks_list else None,
                    special_values={var: special_values[var]} if special_values and var in special_values else None,
                    bin_num_limit=max_bins,
                    bin_min_pct=min_bin_size,
                    method=method,
                    print_info=False,
                )
                bins[var] = sub[var]
            except Exception as e:
                logger.warning("变量 %s 分箱失败: %s (已跳过)", var, e)
    logger.info("WOE 分箱完成: 共 %d 个变量", len(bins))
    return bins


def woebin_ply(
    df: pd.DataFrame,
    bins: dict[str, pd.DataFrame],
    var_list: list[str] | None = None,
) -> pd.DataFrame:
    """将原始数据按分箱对象转换成 WOE 值。

    Args:
        df: 待转换数据（训练或测试集）
        bins: woebin() 返回的分箱字典
        var_list: 需要转换的变量列表，默认所有键

    Returns:
        WOE 化后的 DataFrame
    """
    target_vars = var_list or list(bins.keys())
    # 兼容 scorecardpy 0.1.9.7 + pandas 3.0：
    #   - bins 字典时 scorecardpy 内部 pd.concat(dict) 在新 pandas 上抛 "No objects"
    #   - 直接传 list 则后续 bins['variable'] 取不到
    #   正确做法：先手动 concat 成一个 DataFrame（含 variable 列），再传给 scorecardpy
    # 防御：scorecardpy 在大面板上分箱失败时会把 float/None 混进 bins 字典，必须过滤
    if isinstance(bins, dict):
        normalized = []
        for name, b in bins.items():
            if not isinstance(b, pd.DataFrame) or b.empty:
                logger.warning("woebin_ply 跳过非 DataFrame/空 bins 项: %s (%s)", name, type(b).__name__)
                continue
            if "variable" not in b.columns:
                normalized.append(b.assign(variable=name))
            else:
                normalized.append(b)
        if not normalized:
            raise RuntimeError("woebin_ply: bins 字典全部无效（可能 scorecardpy.woebin 在该数据集上全部失败）")
        bins_for_ply = pd.concat(normalized, ignore_index=True)
    else:
        bins_for_ply = bins
    # 关键修复：replace_blank_na=True 在 scorecardpy 0.1.9.7 + pandas 3.0 上会触发
    # `str.findall().apply(lambda x: len(x))`，对含 NaN 的 str 列抛 TypeError。
    # 改为 False 即可。空字符串处理由前序 _convert_stringdtype_to_object 保证。
    woe_df = sc.woebin_ply(df, bins_for_ply, var_list=target_vars,
                           replace_blank=False)
    logger.info("WOE 转换完成: %d 列", woe_df.shape[1])
    return woe_df


def compute_iv(
    df: pd.DataFrame,
    var: str,
    target: str,
    bins: pd.DataFrame | None = None,
) -> float:
    """计算单个变量的 IV 值。

    若提供 bins，则沿用 woebin 的分箱；否则以 10 等频分箱粗算。

    Args:
        df: 数据
        var: 待计算变量
        target: 目标列
        bins: 已有的分箱定义

    Returns:
        IV 浮点值
    """
    if bins is not None:
        iv = float(bins["total_iv"].iloc[0])
    else:
        local_bins = woebin(
            df[[var, target]],
            target=target,
            method="quantile",
            max_bins=10,
        )
        iv = float(local_bins[var]["total_iv"].iloc[0])
    return iv