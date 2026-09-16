"""app.core.paths —— 本地缓存与产物路径。

所有训练产物、缓存、过程文件、训练集历史统一放到项目仓库外部，保持 git 工作区干净。
"""
from __future__ import annotations

from pathlib import Path

# workspace root = credit_agent 的父目录
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = WORKSPACE_ROOT / "credit_agent_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 最新训练好的评分卡，供实时评分页自动加载
LATEST_SCORECARD_PATH = CACHE_DIR / "latest_scorecard.pkl"

# 训练集历史目录
DATASETS_DIR = CACHE_DIR / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
# 训练集索引（JSON），记录每个训练集的元信息
DATASETS_INDEX = CACHE_DIR / "datasets_index.json"

# 默认数据集路径（项目内的 CSMAR 完整面板）
DEFAULT_DATASET_PATH = Path(r"D:\vibe coding\data store\csmar_enterprise_panel.csv")
