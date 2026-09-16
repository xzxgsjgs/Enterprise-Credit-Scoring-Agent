"""scripts.train_full_panel — 用完整 CSMAR 面板训练评分卡并输出到缓存目录。

方便快速验证：直接跑本脚本即可在 credit_agent_cache 生成干净的 latest_scorecard.pkl。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.paths import CACHE_DIR  # noqa: E402
from app.core.training import run_training_pipeline  # noqa: E402

PANEL_CSV = Path(r"D:\vibe coding\data store\csmar_enterprise_panel.csv")
OUTPUT_SCORECARD = CACHE_DIR / "latest_scorecard.pkl"


def main() -> None:
    if not PANEL_CSV.exists():
        print(f"[error] 面板不存在: {PANEL_CSV}")
        sys.exit(1)

    config = {
        "test_size": 0.3,
        "iv_threshold": 0.02,
        "max_bins": 8,
        "regularization": 0.01,
        "base_score": 600,
        "pdo": 20,
        "base_odds": 50,
        "min_ks": 0.3,
        "min_auc": 0.7,
        "max_psi": 0.1,
        "time_split": False,  # 普通训练，快
        "year_col": "year",
        "min_train_years": 2,
        "use_lgbm": False,
    }

    print(f"[train] 开始训练: {PANEL_CSV}")
    result = run_training_pipeline(
        PANEL_CSV,
        target="is_default_next_year",
        config=config,
        progress_callback=lambda stage, detail: print(f"[{stage}] {detail}"),
    )
    print(f"[train] 完成 · test AUC={result.metrics.get('auc', 0):.3f}")
    print(f"[train] 评分卡变量数={len(result.scorecard)}")
    print(f"[train] 变量列表: {list(result.scorecard.keys())[:5]} ...")

    import pickle
    OUTPUT_SCORECARD.write_bytes(pickle.dumps(result.scorecard))
    print(f"[train] 评分卡已保存: {OUTPUT_SCORECARD}")


if __name__ == "__main__":
    main()