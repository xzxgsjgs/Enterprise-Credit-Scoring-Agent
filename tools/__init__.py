"""credit_agent.tools — 信用评分卡建模工具集

所有数值计算与建模由本包内确定的 Python 函数完成，供 LangGraph/LLM 智能体调用。
严禁 LLM 直接生成数值结果。
"""

from .data_tools import (
    load_data,
    explore_data,
    split_dataset,
    handle_missing,
    handle_outliers,
)
from .feature_tools import (
    var_filter,
    woebin,
    woebin_ply,
    compute_iv,
)
from .model_tools import (
    model_train,
    model_predict,
    stepwise_selection,
    evaluate_performance,
)
from .scorecard_tools import (
    build_scorecard,
    scorecard_ply,
)
from .validation_tools import (
    compute_psi,
    compute_csi,
    guardrail_check,
    GuardrailResult,
)

__all__ = [
    "load_data", "explore_data", "split_dataset", "handle_missing", "handle_outliers",
    "var_filter", "woebin", "woebin_ply", "compute_iv",
    "model_train", "model_predict", "stepwise_selection", "evaluate_performance",
    "build_scorecard", "scorecard_ply",
    "compute_psi", "compute_csi", "guardrail_check", "GuardrailResult",
]

__version__ = "0.1.0"