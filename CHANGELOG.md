# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-09-11

### Added
- **阶段一**：17 个确定性工具函数（5 模块）
  - `data_tools.py`: load_data / explore_data / split_dataset / handle_missing / handle_outliers
  - `feature_tools.py`: var_filter / woebin / woebin_ply / compute_iv
  - `model_tools.py`: model_train / model_predict / stepwise_selection / evaluate_performance
  - `scorecard_tools.py`: build_scorecard / scorecard_ply
  - `validation_tools.py`: compute_psi / compute_csi / guardrail_check
- **阶段二**：LangGraph 编排（14 节点 + HITL + SqliteSaver + PickleFallbackSerializer）
- **阶段三**：Streamlit 双页应用（Home + 模型训练 + 实时评分）
- **阶段四（v2 增强）**：
  - CSMAR v2 训练面板（剔除金融业 J6* + 7 行业相对特征 + 10 滚动 3y 时序特征 + 1%/99% Winsorize）
  - 完整 10 年 Expanding Window CV（`scripts/run_full_cv.py`，含断点续跑）
  - LGBM+SHAP 对照分支（`use_lgbm=True`）
  - 简化 SHAP（LR+WOE 的 |coef| 排序）
- **测试**：10/10 通过（5 tools + 5 graph）
- **文档**：README / LICENSE / CHANGELOG / CONTINUATION_PROMPT / RESUME_PROMPT

### Verified
- 标签构造无泄露（600721 案例 label_year = feature_year+1 命中 100%）
- 完整 10 年 CV：mean AUC=0.9347 / KS=0.7471 / Gini=0.8693（7/7 fold）
- LGBM 4 年子集对照：test AUC=0.948 / KS=0.772

### Known Limitations
- DEMO 用途，非生产级模型
- 仅中国 A 股上市公司
- 无 Docker / CI / 认证
- pickle 反序列化存在 RCE 风险（仅限可信环境）

[1.0.0]: https://github.com/xzxgsjgs/Enterprise-Credit-Scoring-Agent/releases/tag/v1.0.0
