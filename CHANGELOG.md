# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Streamlit 界面接入 V2 六模块（模型训练页重构为 Agent 工作台）

- **`app/core/agent_runner.py`（新）**：在 Streamlit 中驱动 18 节点 LangGraph 图
  - `run_agent()`：`app.stream()` 跑图，逐节点回调事件；支持 `Command(resume=...)` 继续 HITL
  - `build_agent_config()`：三层配置合并（基础参数 / Agent 白名单 / NL patch）
  - `summarize_state()`、`save_scorecard()`：状态摘要与评分卡落盘（供实时评分页）
- **`app/ui/agent_timeline.py`（新）**：18 节点实时执行时间线，
  状态含 待执行 / 运行中 / 成功 / 告警 / 失败 / 🔁 回跳，回跳节点显示执行次数
- **`app/ui/agent_panels.py`（新）**：六模块面板 —— 数据守门、自愈履历、
  Critic 复核、cut-off 与分档、13 章开发报告、人工复核，外加顶部三卡概览
- **`app/pages/1_模型训练.py` 重构**：
  - 侧边栏分区 ① 训练集 ② 执行模式 ③ 数据 ④ 特征分箱 ⑤ 评分卡 ⑥ 护栏
    ⑦ 数据守门阈值 ⑧ Critic 阈值 ⑨ cut-off 与报告 ⑩ 对照模型
  - 主区：自然语言需求输入 → 数据概览 → 执行 → 实时时间线 → 结果 7 页签 → 人工复核
  - **双模式共存**：🤖 Agent 闭环（守门/自愈/Critic/报告）与 ⚡ 快速训练（时序 CV/LGBM）
  - 三个终局分别渲染：守门拦截 / 关键节点失败 / 人审中断（不再混在一起）
- **`app/Home.py`**：改为导航页，说明两个页面与六模块在界面上的位置

### Changed
- 训练页与实时评分页的 `use_container_width` 全部迁移为 `width="stretch"/"content"`
  （Streamlit 1.63 起该参数已弃用，共 20 处）

### Fixed
- 自然语言解析出的基础建模参数（`max_bins` / `iv_threshold` 等）会被配置白名单静默丢弃 →
  拆分 `build_agent_config(base, agent_opts, nl_patch)`，NL patch 原样合并（已过 `validate_patch`）
- 数据守门拦截时仍显示「一次通过，未触发自愈回路」→ 增加 `reached_modeling` 分支

### Tests
- 新增 `tests/test_agent_runner.py`（29 例）、`tests/test_app_pages.py`（9 例，AppTest 页面冒烟）、
  `tests/test_app_agent_e2e.py`（3 例页面内真跑闭环，`-m slow` 才执行）
- 全量 **223 passed**（+3 slow 默认跳过）

### Added — Agent V2：从「线性流水线」升级为「自主决策闭环」（批次 A~D，18 节点 / 185 tests）

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
