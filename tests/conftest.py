"""Pytest 配置：将项目根加入 sys.path，让 tests 可直接 `from agent...` 导入。"""
from __future__ import annotations

import sys
from pathlib import Path

# 项目根 = tests/ 的父目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
