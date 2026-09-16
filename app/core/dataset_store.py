"""app.core.dataset_store —— 用户上传训练集的持久化与历史管理。

设计：
- 默认数据集（DEFAULT_DATASET_PATH）始终作为「默认训练集」选项，不复制到 DATASETS_DIR。
- 用户上传的新训练集保存到 DATASETS_DIR，并在 DATASETS_INDEX 注册元信息。
- 元信息：id（filename stem + timestamp）、label（用户命名）、path、rows、cols、uploaded_at、size_bytes。
- 启动时自动加载索引；删除/重命名操作同步更新索引。
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.paths import DATASETS_DIR, DATASETS_INDEX


def _slugify(name: str) -> str:
    """把用户输入的 label 转换为合法文件名（去特殊字符、保留中文/数字）。"""
    name = name.strip()
    if not name:
        return ""
    # 仅保留中文、英文字母、数字、下划线、连字符、空格
    safe = re.sub(r"[^\w\u4e00-\u9fff\- ]", "", name)
    return safe.replace(" ", "_")


def _read_index() -> list[dict[str, Any]]:
    if not DATASETS_INDEX.exists():
        return []
    try:
        return json.loads(DATASETS_INDEX.read_text(encoding="utf-8"))
    except Exception:
        return []


def _write_index(items: list[dict[str, Any]]) -> None:
    DATASETS_INDEX.write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def list_uploaded_datasets() -> list[dict[str, Any]]:
    """返回所有已注册的用户上传训练集（按 uploaded_at 倒序）。"""
    items = _read_index()
    # 兜底：若文件已被外部删除，索引仍能反映真实情况（不主动清理，保留历史）
    return sorted(items, key=lambda x: x.get("uploaded_at", ""), reverse=True)


def save_uploaded_dataset(
    uploaded_file,
    label: str,
    src_suffix: str = ".csv",
) -> dict[str, Any]:
    """保存一个用户上传的训练集到 DATASETS_DIR 并注册到索引。

    Args:
        uploaded_file: Streamlit UploadedFile 或类似文件对象（有 .read() / .getvalue() / .name）
        label: 用户命名的训练集别名
        src_suffix: 文件后缀（.csv / .xlsx / .xls）

    Returns:
        新注册的元信息 dict
    """
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    slug = _slugify(label) or "dataset"
    fname = f"{slug}_{ts}{src_suffix.lower()}"
    target = DATASETS_DIR / fname

    # 读取字节内容（兼容 UploadedFile / 临时路径 / 普通路径）
    if hasattr(uploaded_file, "getvalue"):
        data = uploaded_file.getvalue()
    elif hasattr(uploaded_file, "read"):
        data = uploaded_file.read()
    elif isinstance(uploaded_file, (str, Path)):
        data = Path(uploaded_file).read_bytes()
    else:
        raise TypeError(f"无法读取上传文件: {type(uploaded_file)}")
    target.write_bytes(data)

    # 计算元信息（行/列）
    rows, cols = _peek_shape(target)

    meta = {
        "id": fname,
        "label": label,
        "path": str(target),
        "rows": rows,
        "cols": cols,
        "size_bytes": target.stat().st_size,
        "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "uploaded",
    }
    items = _read_index()
    items.append(meta)
    _write_index(items)
    return meta


def delete_dataset(dataset_id: str) -> bool:
    """从索引删除一条记录，并尝试删除磁盘文件。"""
    items = _read_index()
    new_items = []
    removed = False
    for it in items:
        if it["id"] == dataset_id:
            try:
                Path(it["path"]).unlink(missing_ok=True)
            except Exception:
                pass
            removed = True
        else:
            new_items.append(it)
    if removed:
        _write_index(new_items)
    return removed


def _peek_shape(path: Path) -> tuple[int, int]:
    """只读前几行估算行列数，避免大文件全量读入。"""
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, nrows=5, encoding="utf-8-sig")
        else:
            df = pd.read_excel(path, nrows=5)
        # 用 chunk 读取总行数更稳
        if path.suffix.lower() == ".csv":
            with path.open("rb") as f:
                total = sum(1 for _ in f) - 1  # 减表头
        else:
            total = len(pd.read_excel(path))  # Excel 没办法 chunk，只能全读
        return max(total, 0), df.shape[1]
    except Exception:
        return 0, 0