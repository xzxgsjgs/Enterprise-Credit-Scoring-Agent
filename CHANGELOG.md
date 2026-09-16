# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Agent V2：从「线性流水线」升级为「自主决策闭环」（批次 A~D 全部完成，18 节点 / 185 tests）

- **批次 A**：数据质量前置守门（`data_gate_node` + `check_data_quality`）、
  不可恢复节点短路到 END（`critical_error` + `route_after_critical`）、
  自然语言配置入口（`nl_config.py`，白名单 + 范围校验）
- **批次 B**：诊断-重规划回路（`diagnose_node` + `Command(goto=)` 回跳 5 个上游节点），
  护栏失败分级自愈（PSI → CSI → 剔变量；KS 过低 → 换 LGBM 兜底），
  `retry_history` 指纹比对防 LLM 重复出同一个 patch
- **批次 C**：cut-off 择优（`optimize_cutoff` 三口径网格）+ 分数分档（`grade_scores`）、
  《模型开发报告》自动落盘 + 《拒贷理由书》（LLM 只写定性段落，禁止写数字 + `\d` 泄漏拦截）
- **批次 D**：Critic 模型复核——`tools/audit_tools.py` 三项确定性审计
  （VIF 共线性 / 系数符号 vs WOE 方向 / 样本取值集中度）+ LLM 可降级定性复核，
  产出报告第十一章「模型复核结论（Critic）」

### Changed
- 《模型开发报告》由 12 章扩到 13 章（护栏章之后插入 Critic 章，其后两章顺延）
- `check_sample_concentration` 新增 `exclude` 参数，默认排除 target / id / year 列

### Fixed
- `handle_outliers(cap)` 误处理 target 列（把标签 1 压成 0.88，导致出现第三个取值）
- LLM 复核恒定失效：qwen-turbo 把 `item=...` 整行抄回来，严格相等匹配必然不命中 →
  新增 `_match_known_item` 三级降级匹配
- 报告第十一章不再出现「列 target 的取值 0 占比 96.9%」这类无结论价值的噪音

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
