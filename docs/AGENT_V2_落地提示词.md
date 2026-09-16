# credit_agent Agent 能力升级 —— WorkBuddy 落地提示词

> 用法：把下面「一、任务」到「十、交付要求」的全部内容整段复制给新的 WorkBuddy 会话（或在 credit_agent 目录下开新会话直接粘贴）。
> 本文档自包含，无需补充背景。

---

## 一、任务

将 `credit_agent` 从「线性批处理流水线 + 一个人工确认按钮」升级为**具备自主决策闭环的建模 Agent**。

当前问题（已核实）：
- `agent/graph.py` 全部为 `add_edge`，仅 1 处条件边 → 本质是线性 DAG，无循环、无回溯
- graph 内 **0 个 LLM 节点**；LLM 只存在于 Streamlit UI 的两个按钮
- `route_after_hitl` 恒返回 `end` → 人工决策不改变后续路径
- `@safe` 只累积 `errors`，无任何节点消费它做决策
- `explore_data` 产出的 `eda_report` 无下游消费者

目标形态：**感知（EDA/指标/护栏） → 诊断（LLM + 规则） → 行动（改参数/回跳重跑） → 观察（重算指标） → 再判断**，形成有界闭环。

---

## 二、硬约束（违反即返工）

1. **LLM 严禁生成任何数值**。KS / AUC / Gini / PSI / 分箱边界 / 回归系数 / 分数 / cut-off 一律由 `tools/` 下确定性函数计算。
2. **LLM 只允许输出三类内容**：① 动作选择（文字枚举值）；② 参数名与参数值（必须通过白名单 + 范围校验）；③ 自然语言解释与建议。
3. **所有新增数值逻辑必须放 `tools/`**，函数级 docstring 需含公式说明，且可脱离 LangGraph 单测。
4. **LLM 必须可降级**：无 `.env` / 无 API key / 调用超时 / JSON 解析失败 → 自动回退到确定性规则分支，绝不允许整图失败。
5. **向后兼容**：`app/core/training.py::run_training_pipeline` 的返回字段、`agent/run.py` 现有 CLI（start / resume / show）不得破坏。
6. 新增 config 项一律带默认值，缺省时行为与升级前一致。

---

## 三、项目现状（绝对路径 + 环境）

### 3.1 路径

| 用途 | 路径 |
|---|---|
| 项目根 | `D:\vibe coding\2026-09-08-16-37-15\credit_agent` |
| 工具层（17 函数） | `credit_agent/tools/{data,feature,model,scorecard,validation}_tools.py` |
| 编排层 | `credit_agent/agent/{state,nodes,graph,serde,run}.py` |
| 一键训练（UI 用，无 HITL） | `credit_agent/app/core/training.py` |
| LLM 客户端 / 提示词 | `credit_agent/app/llm/{client,prompts}.py` |
| 配置 | `credit_agent/config/default_config.yaml` |
| 训练面板 | `D:\vibe coding\data store\csmar_enterprise_panel.csv`（44,017 行 × 36 列，5,718 家公司，违约率 2.92%） |
| 面板生成脚本 | `credit_agent/scripts/prepare_csmar_panel.py` |
| 全量训练脚本 | `credit_agent/scripts/train_full_panel.py`、`scripts/run_full_cv.py` |

### 3.2 运行环境（Windows，路径含空格，命令必须加引号）

```bash
# Python（正确的 venv，勿用裸 python / base anaconda）
"C:/Users/27514/.workbuddy/binaries/python/envs/default/Scripts/python.exe"

# 跑测试
cd "D:/vibe coding/2026-09-08-16-37-15/credit_agent"
"C:/Users/27514/.workbuddy/binaries/python/envs/default/Scripts/python.exe" -m pytest tests/ -v

# 跑 CLI
"C:/Users/27514/.workbuddy/binaries/python/envs/default/Scripts/python.exe" -m agent.run start --csv "路径" --target is_default_next_year
```
依赖（scorecardpy 0.1.9.7 / langgraph / langchain-openai / streamlit / lightgbm / shap / openpyxl）已装在上述 venv。

### 3.3 必读文件与关键机制

| 文件 | 必读要点 |
|---|---|
| `agent/state.py` | `AgentState` TypedDict；普通字段=覆盖写，`errors/warnings/step_history` 为 `Annotated[list, operator.add]` 累积写 |
| `agent/nodes.py` | 14 节点；`@safe(name)` 装饰器捕获异常写 `errors`；`_ok/_err` 辅助；`route_after_guardrail` / `route_after_hitl` |
| `agent/graph.py` | `build_graph()` 未编译图；`compile_with_sqlite()` 绑 SqliteSaver；条件边写法 |
| `agent/serde.py` | `PickleFallbackSerializer`：先 JsonPlus，失败退 pickle（DataFrame / sklearn model 必需） |
| `agent/run.py` | CLI；`_load_config_overrides` 把 YAML 嵌套键**扁平化**成一层 dict，`project/name/version` 被剔除 |
| `config/default_config.yaml` | 全部可调参数；已有 v2 键：`time_split` / `year_col` / `min_train_years` / `use_lgbm` / `lgbm_*` |
| `app/llm/client.py` | `get_model(label=None)` 返回 OpenAI 兼容 chat 模型；`LLM_PRESETS` 必须是**单行 JSON array** |
| `app/llm/prompts.py` | 已有 `GUARDRAIL_INTERPRETATION` / `SCORE_EXPLANATION` 模板，可复用风格 |

---

## 四、交付范围（6 个模块）

### 模块 1：诊断-重规划回路（核心）

**目标**：护栏未通过时，由 LLM（降级时用规则）决定"改哪些参数、回到哪一步"，自动重跑，最多 `max_retry` 次。

**新增 state 字段**（`agent/state.py`）
```python
diagnosis: dict[str, Any]            # 最近一次诊断：{action, goto, param_patch, reason, source}
retry_count: int                     # 已重试次数（普通字段，覆盖写，不累积）
retry_history: Annotated[list[dict], operator.add]   # 每轮诊断记录，用于报告
```

**新增 config 段**（`config/default_config.yaml`）
```yaml
agent:
  max_retry: 3
  allow_llm_diagnosis: true
  allowed_params: [iv_threshold, missing_threshold, identical_threshold,
                   max_bins, min_bin_size, binning_method, missing_strategy,
                   outlier_method, outlier_sigma, regularization, test_size,
                   use_lgbm, exclude_vars, min_ks, min_auc, max_psi]
  allowed_goto: [preprocess, var_filter, woebin, model_train, build_scorecard]
  param_bounds:
    iv_threshold: [0.0, 0.2]
    missing_threshold: [0.1, 0.9]
    identical_threshold: [0.5, 1.0]
    max_bins: [3, 12]
    min_bin_size: [0.01, 0.2]
    outlier_sigma: [2.0, 8.0]
    regularization: [0.001, 1.0]
    test_size: [0.1, 0.5]
    min_ks: [0.05, 0.6]
    min_auc: [0.5, 0.95]
    max_psi: [0.05, 0.5]
  binning_method_choices: [tree, chimerge, quantile, equal]
  missing_strategy_choices: [mean, median, mode, constant, drop]
```

**新增节点** `diagnose_node`（`agent/nodes.py`，**不加 `@safe`**，因为要返回 `Command`）
```python
def diagnose_node(state: AgentState) -> Command:
    """读护栏结果 + 指标 + 特征重要性 + CV 明细 → 输出结构化诊断 → 回跳或转人工。

    LLM 只在 allow_llm_diagnosis=True 且调用成功时使用；
    失败/超限/解析异常 → _rule_based_diagnosis() 兜底。
    返回 Command(goto=<节点名>, update={diagnosis, retry_count, retry_history,
                                        config_overrides, warnings})
    """
```
- **LLM 输出 schema**（严格 JSON，用 `temperature=0`）：
```json
{"action": "retune|escalate|abort",
 "goto": "var_filter",
 "param_patch": {"max_bins": 6},
 "reason": "中文一句话说明判断依据"}
```
- **校验层** `_validate_diagnosis(raw, cfg) -> tuple[dict, list[str]]`：
  - `goto` 不在 `allowed_goto` → 整条丢弃，转 escalate
  - `param_patch` 的 key 不在 `allowed_params` → 丢弃该键并记 warning
  - value 超出 `param_bounds` → 夹到边界值并记 warning
  - `binning_method` / `missing_strategy` 必须在各自的 choices 内
- **停止条件**：`retry_count >= max_retry` 或 `action in {escalate, abort}` → `Command(goto="hitl_review")`
- **规则兜底** `_rule_based_diagnosis(state, cfg) -> dict`（确定性，见模块 2 规则表）
- **回跳时必须重置**：`config_overrides` 合并 patch；`retry_count` 用 `state.get("retry_count", 0) + 1`（**不能**用 `add` reducer）

**图改造**（`agent/graph.py`）
```python
builder.add_node("diagnose", diagnose_node)
builder.add_conditional_edges(
    "guardrail_check", route_after_guardrail,
    {"end": END, "diagnose": "diagnose", "hitl": "hitl_review"},
)
builder.add_conditional_edges(
    "diagnose", route_after_diagnose,
    {"preprocess": "preprocess", "var_filter": "var_filter", "woebin": "woebin",
     "model_train": "model_train", "build_scorecard": "build_scorecard",
     "hitl": "hitl_review"},
)
```
`route_after_diagnose(state)`：读 `state["diagnosis"]["goto"]`，缺失或非法 → `"hitl"`。

**`route_after_guardrail` 新逻辑**
```python
gr is None                                    → "hitl"
passed and not requires_human_review          → "end"
retry_count 未超限 且 有可行动作               → "diagnose"
否则                                          → "hitl"
```

**验收**：构造一次必然失败的场景（`min_ks=0.99` 或人为降 KS），验证图自动回跳 ≥1 次、`retry_history` 有记录、超限后停在 `hitl_review`。

---

### 模块 2：护栏失败的分级自愈

**目标**：把 approve/reject 二元判断扩展成分级处置，能自动的先自动。

**新增工具** `tools/validation_tools.py::diagnose_psi_sources(...)`
```python
def diagnose_psi_sources(
    bins: dict[str, pd.DataFrame],
    expected_df: pd.DataFrame,   # 训练集原始值
    actual_df: pd.DataFrame,     # 测试集原始值
    target: str,
    top_n: int = 5,
) -> pd.DataFrame:
    """逐变量计算 CSI（特征级 PSI），定位分布漂移来源。

    对变量 j：按 bins[j] 的箱边界分别统计 expected / actual 的箱占比，
    CSI_j = Σ_i (a_i − e_i) · ln(a_i / e_i)，空箱用 1e-4 平滑。
    返回列：variable / csi / top_moved_bin / expected_pct / actual_pct
    """
```

**规则表**（写进 `_rule_based_diagnosis`，文件内以常量表形式给出，便于审阅）

| 触发条件 | goto | param_patch | 说明 |
|---|---|---|---|
| `psi > 0.25` 且 `diagnose_psi_sources` 返回 CSI 最高变量 | `var_filter` | `{"exclude_vars": [<csi_top>]}` | 剔除漂移变量重跑 |
| `psi > 0.25` 但 CSI 无明显集中 | `woebin` | `{"max_bins": 6}` | 重新分箱，粗化箱数 |
| `ks < 0.2` | `woebin` | `{"binning_method": "chimerge"}` | 换分箱算法 |
| `ks < min_ks` 且 `auc >= min_auc` | `woebin` | `{"min_bin_size": 0.08}` | 合并小箱提升稳定性 |
| `auc < min_auc` 且 `use_lgbm` 未开 | `model_train` | `{"use_lgbm": True}` | 切 LGBM 对照 |
| CV 某 fold 单类别 | — | 跳过并记 `retry_history` | 无需重跑 |
| 兜底 | `hitl_review` | — | 转人工 |

**新增 config**：`exclude_vars: []`（默认空）。在 `preprocess_node` 与 `var_filter_node` 生效：从 `train_df`/`test_df` 中 drop 这些列，并写入 `filtered_vars` 的排除说明。

**验收**：构造人为漂移（对测试集某变量做线性变换），验证 CSI 能定位到该变量、`exclude_vars` 生效、PSI 下降。

---

### 模块 3：多智能体分工（Critic + Reporter）

#### 3.1 新增 `tools/audit_tools.py`（纯确定性）

| 函数 | 公式 / 逻辑 | 返回 |
|---|---|---|
| `compute_vif(X: pd.DataFrame) -> pd.DataFrame` | 对每个变量 j 回归其余变量得 R²_j，`VIF_j = 1 / (1 − R²_j)`；>10 报多重共线性 | `variable / vif / flag` |
| `check_coef_consistency(bins, model, xcolumns) -> pd.DataFrame` | 对每个变量，用 WOE 的箱均值与坏账率算相关性 `sign(corr(woe_i, bad_rate_i))`，与模型系数符号比较；不一致报警（业务方向反转） | `variable / coef / expected_sign / actual_sign / ok` |
| `check_sample_concentration(df, cols) -> pd.DataFrame` | 单一取值占比，如某 `IndustrySector` 占比 > 50% 报警 | `column / top_value / share / flag` |

#### 3.2 `critic_node`（规则先行，LLM 复核）

- 规则结果写入 `critic_findings: list[dict]`（`Annotated[list, operator.add]`）
- LLM 复核：把规则结果 + 特征重要性 top10 + 指标 传给 LLM，输出
```json
{"passed": true, "issues": [{"item": "...", "severity": "high|mid|low", "suggestion": "..."}]}
```
- 写入 `critic_report: dict`（含 `source: "rule" | "rule+llm"`）
- 位置：`build_scorecard` 之后、`scorecard_ply` 之前（或与 guardrail 并联后再汇合）
- **不阻塞主流程**：Critic 失败只记 warning，不影响 guardrail 路由

#### 3.3 `reporter_node`

- 输入：`eda / metrics / cv_per_fold / cv_mean / scorecard / feature_importance / guardrail / critic_report / retry_history / bins / scored_test`
- 数值全部由 Python 插值，LLM 仅撰写「分析与结论」段落（失败则留占位符）
- 新增 `tools/report_tools.py::render_model_report(ctx: dict) -> str`，输出 Markdown
- 落盘：`credit_agent/reports/model_report_{thread_id}.md`
- 报告结构（固定章节）：项目概述 / 数据说明 / 标签定义 / 特征工程 / 模型与参数 / 性能指标（含逐年 CV 表）/ 特征重要性 / 评分卡刻度与分档 / cut-off 建议 / 稳定性与护栏结论 / 风险与局限 / 附录（分箱明细）
- 新增 config：`generate_report: true`、`report_dir: "reports"`

**验收**：跑一次全流程，产出报告文件；断网/无 key 时报告仍生成（分析段落为占位文本）。

---

### 模块 4：自然语言配置入口

**新增** `agent/nl_config.py`
```python
def parse_nl_config(text: str) -> tuple[dict, list[str]]:
    """把中文需求解析为 config_overrides。
    返回 (patch, warnings)；patch 已通过白名单与范围校验，warnings 记录被丢弃的项。
    LLM 不可用时返回 ({}, ["LLM 不可用，已忽略自然语言配置"])。
    """

def validate_patch(patch: dict) -> tuple[dict, list[str]]:
    """键必须在 default_config.yaml 的已知键集合内；值域检查；类型转换。"""
```
- Prompt 中必须给出**完整的合法键清单及各键含义**，并要求只输出 JSON
- 支持的高频表达 → 键映射示例：
  - "用 2020 年以后的数据" → `{"min_year": 2020}`（新增 config 键，`load_data` 后按 `year_col` 过滤）
  - "排除房地产行业" → `{"exclude_industries": ["K70"]}`（新增 config 键，`preprocess` 前生效）
  - "KS 目标 0.4" → `{"min_ks": 0.4}`
  - "最多分 6 箱" → `{"max_bins": 6}`
  - "开启时序验证" → `{"time_split": true}`
- `agent/run.py` 新增参数：`--ask "自然语言需求"`，与 `--config` 合并（`--ask` 优先级更高）
- Streamlit `app/pages/1_模型训练.py` 增加一个 text_area + "解析需求" 按钮，展示解析出的 patch 供用户确认后再跑

**验收**：`--ask "用2018年以后数据、最多分6箱、开启时序验证"` 能解析出三个正确键；输入 "把正则化调成 0.5 并把 KS 调到 0.99" 时越界项被拒绝并给出 warning。

---

### 模块 5：cut-off 择优与可解释输出

**新增** `tools/cutoff_tools.py`
```python
def optimize_cutoff(y_true, score, method="ks", target=None) -> dict:
    """在分数区间上网格搜索最优 cut-off。

    method="ks"           : 最大化 KS = max_t |TPR(t) − FPR(t)|
    method="approve_rate" : 命中目标通过率 target（如 0.7）
    method="bad_rate"     : 命中目标通过后坏率 target
    返回 {cutoff, approve_rate, bad_rate_after, ks_at_cutoff,
          capture_rate(违约捕获率), lift, table(各档明细 DataFrame)}
    """

def grade_scores(score: pd.Series, cutoffs: list[float] | None = None,
                 labels=("A", "B", "C", "D")) -> pd.DataFrame:
    """按分位或固定分界给分数分档，返回 score/grade 两列。"""
```
- 在 `scorecard_ply(only_total_score=False)` 下拿到逐变量分值，用于解释
- **新增** `tools/report_tools.py::render_reject_letter(score_detail: pd.DataFrame, top_k=5) -> str`
  - 取对总分**负向贡献最大**的 k 个变量，翻译成业务语言
  - 模板示例："短期偿债能力（流动比率 0.82，低于样本中位数）扣 18 分；近三年杠杆波动较大扣 12 分"
- 接进 `reporter_node` 报告章节；Streamlit 实时评分页展示 cut-off 建议与分档结果
- **不接 LLM 也可用**（纯模板）；LLM 仅用于润色措辞

**验收**：对测试集输出 cut-off、通过率、坏率、捕获率；报告含分档表；单笔样本可生成理由书。

---

### 模块 6：数据质量前置守门 + 错误短路

**新增节点** `data_gate_node`，位置：`explore_data` 之后、`split` 之前。

**判定规则（确定性，阈值写进 config）**
```yaml
data_gate:
  min_rows: 1000
  min_default_rate: 0.005
  max_default_rate: 0.5
  min_years: 3                 # 仅当 time_split=True 时校验
  max_missing_ratio: 0.6       # 单列缺失率上限，超出提示剔除
  max_identical_ratio: 0.98    # 单一值占比上限
```
- 输出 `data_gate: {"passed": bool, "blocks": [...], "warnings": [...], "suggestion": str}`
- `suggestion` 由 LLM 撰写（降级时用模板拼接）
- 路由：`passed=False` → 直接 `END`（附原因），不再硬跑后续节点

**同时修「错误短路」缺陷**
- 新增 state 字段 `critical_error: bool`（普通字段）
- 新增路由辅助 `has_critical_error(state) -> bool`：`load_data` / `split` / `woebin` / `model_train` 这四个不可恢复节点失败 → 置 `critical_error=True`
- 在上述节点之后挂条件边：`critical_error` 为真 → 直接 `END`，避免后续 13 个节点连环报错

**验收**：喂一个只有 200 行 / 违约率 0.1% 的数据集，图在 `data_gate` 处终止并给出原因；喂不存在的 csv，图在 `load_data` 后立即 `END`（而非跑满 13 个错误）。

---

## 五、文件清单

**新增**
```
agent/nl_config.py            自然语言 → config 解析与校验
agent/routing.py              所有路由函数集中（route_after_*、has_critical_error）
tools/audit_tools.py          compute_vif / check_coef_consistency / check_sample_concentration
tools/cutoff_tools.py         optimize_cutoff / grade_scores
tools/report_tools.py         render_model_report / render_reject_letter
tests/test_agent_loop.py      诊断-重规划回路测试
tests/test_audit_tools.py     审计工具测试
tests/test_cutoff_tools.py    cut-off 测试
tests/test_data_gate.py       数据守门与短路测试
docs/ARCHITECTURE_V2.md       改造后架构说明（含 mermaid 拓扑图）
```

**修改**
```
agent/state.py                新增 diagnosis / retry_count / retry_history / critic_* /
                              data_gate / critical_error / exclude_vars 等字段
agent/nodes.py                新增 diagnose_node / critic_node / reporter_node / data_gate_node；
                              插入 data_gate 节点与短路条件边；exclude_vars 生效
agent/graph.py                新增节点注册、diagnose 条件边、短路条件边
agent/run.py                  新增 --ask 参数；resume 支持携带 retry 上下文
config/default_config.yaml    新增 agent / data_gate / report / exclude_vars / min_year /
                              exclude_industries 段
app/core/training.py          同步新增字段（保持 UI 路径可用，不引入 HITL 阻塞）
app/pages/1_模型训练.py        自然语言输入框 + 诊断/重试历史展示 + 报告下载
tools/validation_tools.py     新增 diagnose_psi_sources
tools/__init__.py             导出新增函数
```

---

## 六、实施顺序（4 个批次，每批跑通再进下一批）

| 批次 | 内容 | 理由 |
|---|---|---|
| A | 模块 6 + 模块 4 的校验层 + 错误短路 | 基础设施，无 LLM 依赖，风险最低 |
| B | 模块 1（诊断-重规划）+ 模块 2（分级自愈） | 核心闭环，让图第一次出现"环" |
| C | 模块 5（cut-off/理由书）+ 模块 3 的 Reporter | 产出可展示的业务交付物 |
| D | 模块 3 的 Critic | 依赖 B/C 的产物，最后做 |

每批完成后必须：① 跑全量 `pytest`；② 更新 `docs/ARCHITECTURE_V2.md`；③ 在回复中给出该批的实际运行输出。

---

## 七、已知坑（务必遵守，否则会踩重复的雷）

1. **scorecardpy 0.1.9.7 + pandas 3.0 兼容**：
   - `woebin_ply` 默认 `replace_blank=True`，遇到含 NaN 的 object/str 列会抛 `TypeError: object of type 'float' has no len()`（面板 ≥2 万行必现）→ 调用时传 `replace_blank=False`
   - `bins` 字典不能直接传给 `sc.woebin_ply`（内部 `pd.concat(dict)` 会失败）→ 先手动 concat 并补 `variable` 列（见 `feature_tools.py::woebin_ply` 现有实现，勿改动）
   - pandas 3.0 的 `StringDtype`（dtype.name == "str"）需用 `pd.Series(list, dtype=object)` 重建列（`_convert_stringdtype_to_object`），`astype(object)` 无效
2. **`Command` 的正确用法**：`from langgraph.types import Command`；节点返回 `Command(goto=..., update={...})` 时**不要**再套 `@safe` 装饰器（会吞掉 Command）
3. **SqliteSaver + 自定义 serde**：回跳会重复写 checkpoint，`retry_count` 必须用普通字段（覆盖写），不能挂 `add` reducer
4. **`LLM_PRESETS` 必须是单行 JSON array**，多行会导致解析失败（`app/llm/client.py` 已有友好报错）
5. **路径含空格**：所有 shell 命令加双引号
6. **Python 环境**：必须用 venv 的 python / streamlit，裸 `streamlit` 指向 base anaconda（缺依赖）
7. **长耗时**：全量面板 + `time_split=True` 时单 fold 的 `var_filter` 约数分钟；开发调试请用 2018–2020 子集（约 7k 行），勿动全量

---

## 八、评审关注点（设计时就要想清楚）

| 问题 | 期望的答案方向 |
|---|---|
| 为什么回跳不会死循环？ | `max_retry` 硬上限 + 每轮回跳必改至少一个参数 + 参数单调约束（避免反复在两点间震荡，可加"同一 param_patch 不重复应用"去重） |
| LLM 判断错的代价？ | 校验层兜底（白名单 + 边界夹取），最坏情况是浪费一轮重算，不会产生错误数值 |
| 回跳重跑的耗时如何控制？ | 只在必要时回跳（优先 `build_scorecard`/`model_train`，其次 `woebin`，最后 `preprocess`）；首窗 bins 复用 |
| HITL 在闭环中的角色？ | 自动重试耗尽后的最后一道闸门，payload 需带上 `retry_history` 让人工看到"Agent 试过什么" |

---

## 九、不要做的事

- 不要为了"用上 LLM"而在确定性环节插 LLM（如让 LLM 决定分箱边界、判断缺失填充方式）
- 不要引入 LangChain AgentExecutor / AutoGen / CrewAI 等额外框架，保持 LangGraph 单栈
- 不要改动 `app/core/training.py` 中已稳定的数值流程（只做增量字段）
- 不要删除现有 17 个工具函数的签名与行为
- 不要生成 README 之类未被要求的文档（`docs/ARCHITECTURE_V2.md` 除外）

---

## 十、交付要求

1. **代码**：按第五节清单落地，全部通过 `pytest tests/ -v`
2. **架构说明**：`docs/ARCHITECTURE_V2.md`，含改造后的 mermaid 拓扑图（要能看出环和条件边）、新增 state 字段表、新增 config 项表
3. **实跑证据**：至少一次完整的"护栏失败 → 自动诊断 → 回跳 → 重跑 → 通过/转人工"运行日志（贴原始输出）
4. **测试**：每批新增测试文件，覆盖正常路径 + LLM 不可用降级路径 + 参数越界拒绝路径
5. **最终回复**：给出改造前后对比表（节点数、条件边数、是否有环、LLM 参与点、新增交付物），以及每条新增能力的验证命令
