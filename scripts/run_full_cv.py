"""run_full_cv.py —— 完整 10 年 Expanding Window CV，结果落盘到 CSV。

输入：
    D:/vibe coding/data store/csmar_enterprise_panel.csv  (v2: 42,998 × 53)

输出：
    D:/vibe coding/data store/cv_full_results.csv       每 fold 一行
    D:/vibe coding/data store/cv_full_summary.txt       均值/合计摘要

预计耗时：~25 分钟（7k+ 行 × 50 列 scorecardpy.woebin 单次 ~3 分钟 + 6 fold var_filter）
"""
from __future__ import annotations
import sys, time, traceback
from pathlib import Path

sys.path.insert(0, r'D:/vibe coding/2026-09-08-16-37-15/credit_agent')

import pandas as pd
import numpy as np
from app.core.training import _run_expanding_window_cv

PANEL_PATH = r'D:/vibe coding/data store/csmar_enterprise_panel.csv'
RESULT_CSV = r'D:/vibe coding/data store/cv_full_results.csv'
SUMMARY_TXT = r'D:/vibe coding/data store/cv_full_summary.txt'

cfg = {
    'missing_strategy': 'median',
    'outlier_method': 'cap',
    'outlier_sigma': 5.0,
    'iv_threshold': 0.02,
    'missing_threshold': 0.5,
    'identical_threshold': 0.95,
    'max_bins': 6,
    'min_bin_size': 0.05,
    'binning_method': 'tree',
    'regularization': 0.1,
    'max_iter': 500,
}

t0 = time.time()
df = pd.read_csv(PANEL_PATH)
print(f"[{time.time()-t0:.1f}s] 面板已加载: {df.shape}", flush=True)
print(f"  年份范围: {df['year'].min()}-{df['year'].max()}", flush=True)
print(f"  违约率: {df['is_default_next_year'].mean():.2%}", flush=True)

per_fold, mean = _run_expanding_window_cv(
    df,
    target='is_default_next_year',
    year_col='year',
    cfg=cfg,
    min_train_years=3,
    resume_from_csv=RESULT_CSV,  # 断点续跑：复用已成功的 fold
)
per_fold = per_fold.sort_values('year').reset_index(drop=True)
elapsed = time.time() - t0
print(f"\n[{elapsed/60:.1f} min] CV 完成", flush=True)

per_fold.to_csv(RESULT_CSV, index=False, encoding='utf-8-sig')
print(f"\n每 fold 表格已保存: {RESULT_CSV}", flush=True)

# 写摘要
valid = per_fold.dropna(subset=['auc'])
lines = [
    "=" * 70,
    f"Expanding Window CV 完整结果 (10 年面板, min_train_years=3)",
    f"耗时: {elapsed/60:.1f} 分钟",
    f"有效 fold 数: {len(valid)} / {len(per_fold)}",
    "=" * 70,
    "",
    "每 fold 指标:",
    per_fold[['year', 'n_train', 'n_test', 'default_rate', 'auc', 'ks', 'gini', 'error']].to_string(index=False),
    "",
    "-" * 70,
    f"CV 均值 AUC: {mean.get('auc', 0):.4f}",
    f"CV 均值 KS:  {mean.get('ks', 0):.4f}",
    f"CV 均值 Gini:{mean.get('gini', 0):.4f}",
    "-" * 70,
]
summary = "\n".join(lines)
with open(SUMMARY_TXT, 'w', encoding='utf-8') as f:
    f.write(summary)
print(summary, flush=True)
print(f"\n摘要已保存: {SUMMARY_TXT}", flush=True)