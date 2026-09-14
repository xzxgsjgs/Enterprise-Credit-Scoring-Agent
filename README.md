# 企业信用评分卡自动化建模 Agent (Enterprise Credit Scoring Agent)

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-10%2F10%20passing-brightgreen.svg)](tests/)
[![Status](https://img.shields.io/badge/Status-Demo%20v1.0-orange.svg)]()

> **⚠️ DEMO 用途，非生产级模型**。本项目为演示项目，演示混合架构（LLM 编排 + 确定性工具）。**不可直接用于真实信贷决策**。

## 项目简介

基于 [CSMAR](https://www.gtarsc.com/) 中国上市公司财务数据 + ST 退市标签，构建企业主体信用评分卡（Application Scorecard for Enterprises）。

**核心架构**：
- **LLM 仅做编排与解释**（任务拆解、工具调用、护栏结果解读）
- **所有数值计算由确定性 Python 工具完成**（scorecardpy + sklearn + lightgbm）
- 严禁 LLM 直接生成数值结果，杜绝数值幻觉

## 技术栈

| 类别 | 选型 |
|---|---|
| 运行时 | Python 3.13 |
| 评分卡 | scorecardpy 0.1.9.7 + scikit-learn 1.9 |
| 流程编排 | LangGraph + SqliteSaver |
| UI | Streamlit 1.63 |
| LLM | 多模型切换（Qwen / GLM / DeepSeek via OpenAI 兼容接口）|
| 对照模型 | LightGBM 4.7 + SHAP 0.52 |

## 目录结构

```
credit_agent/
├── tools/                # 阶段一：17 个确定性工具
│   ├── data_tools.py         # 加载/EDA/划分/缺失值/异常值
│   ├── feature_tools.py      # var_filter/woebin/woebin_ply/IV
│   ├── model_tools.py        # LR 训练/预测/评估
│   ├── scorecard_tools.py    # 评分卡构建/打分
│   └── validation_tools.py   # PSI/CSI/Guardrail
├── agent/                # 阶段二：LangGraph 编排
│   ├── state.py              # AgentState TypedDict
│   ├── nodes.py              # 14 节点 + safe 装饰器
│   ├── graph.py              # StateGraph + SqliteSaver
│   ├── serde.py              # PickleFallbackSerializer
│   └── run.py                # CLI start/resume/show
├── app/                  # 阶段三：Streamlit 双页应用
│   ├── Home.py               # 入口 + 侧边栏 LLM 选择
│   ├── pages/                # 1_模型训练.py / 2_实时评分.py
│   ├── core/training.py      # 一键训练 pipeline（含时序 CV / LGBM）
│   ├── llm/                  # 多模型客户端
│   └── ui/                   # 指标卡 / 护栏面板 / 评分分布
├── scripts/              # 数据准备 + 离线脚本
│   ├── prepare_csmar_panel.py    # 构建 v2 训练面板
│   └── run_full_cv.py            # 完整 10 年 Expanding Window CV
├── tests/                # 单元测试
│   ├── test_tools.py         # 5 个工具测试
│   └── test_graph.py         # 5 个图测试
├── config/               # 默认配置
├── requirements.txt
├── run_streamlit.bat     # Windows 一键启动
├── LICENSE               # MIT
└── README.md
```

## 快速开始

### 环境要求
- Python 3.13（推荐 [managed venv](https://docs.python.org/3/library/venv.html)）
- Windows / macOS / Linux

### 安装

```bash
# 1. 创建并激活虚拟环境
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate    # macOS/Linux

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 LLM（可选，用于 LLM 解释功能）
cp .env.example .env
# 编辑 .env 填入 DASHSCOPE_API_KEY
```

### 启动 Streamlit UI

```bash
streamlit run app/Home.py
```

或 Windows 一键：

```bat
run_streamlit.bat
```

浏览器打开 http://localhost:8501

### 数据准备（v2 训练面板）

需先从 CSMAR 下载 3 张原始表：
- `FI_T1(Merge Query).csv`（财务指标）
- `BDT_FinConstSA.csv`（ST 标签）
- `STK_LISTEDCOINFOANL(Merge Query).csv`（公司信息）

放到 `D:/vibe coding/data store/extracted/`（或修改 `scripts/prepare_csmar_panel.py` 里的 `DATA_DIR`），然后：

```bash
python scripts/prepare_csmar_panel.py
# 输出：D:/vibe coding/data store/csmar_enterprise_panel.csv (42,998 × 53)
```

### 完整 10 年时序 CV

```bash
python scripts/run_full_cv.py
# 输出：D:/vibe coding/data store/cv_full_results.csv
#       D:/vibe coding/data store/cv_full_summary.txt
# 预计耗时：~60 分钟
```

支持断点续跑（`resume_from_csv` 参数）。

## 模型性能

### 时序验证（10 年 Expanding Window CV，7 fold 全通过）

| 指标 | CV 均值 | 单次随机划分 |
|---|---|---|
| **AUC** | **0.9347** | ~0.93 |
| **KS** | **0.7471** | ~0.69 |
| **Gini** | **0.8693** | ~0.81 |

| year | n_train | n_test | AUC | KS | Gini |
|---|---|---|---|---|---|
| 2018 | 9,918 | 3,677 | 0.9211 | 0.7330 | 0.8422 |
| 2019 | 13,595 | 4,077 | 0.9417 | 0.7352 | 0.8833 |
| 2020 | 17,672 | 4,567 | 0.9530 | 0.7851 | 0.9060 |
| 2021 | 22,239 | 4,945 | 0.9527 | 0.7918 | 0.9054 |
| 2022 | 27,184 | 5,208 | 0.9541 | 0.8178 | 0.9081 |
| 2023 | 32,392 | 5,258 | 0.9061 | 0.6797 | 0.8123 |
| 2024 | 37,650 | 5,348 | 0.9141 | 0.6868 | 0.8281 |

### 业务合理性
- AUC 高的根因（非泄露）：ST 退市标签本身由 ROA/DebtToAsset 等指标触发
- 模型本质 = "用触发 ST 的指标预测 ST"，符合评分卡预期
- 已用 600721 等案例验证 `label_year = feature_year+1` 命中率 100%

## 关键文档

- `CONTINUATION_PROMPT.md` — 项目背景与新会话接入
- `RESUME_PROMPT.md` — 后台任务续接
- `.workbuddy/memory/MEMORY.md` — 长期项目笔记（含 scorecardpy 兼容坑等）

## 已知限制

### 功能层面
- 仅支持中国 A 股上市公司（CSMAR 数据源限定）
- 类不平衡（违约率 2.96%）仅靠 `class_weight=balanced` 处理，无 SMOTE
- 单数据源，无备用 / 外部数据集成

### 工程层面
- 无 Docker 镜像、无 CI/CD
- Streamlit 无认证（仅本地或内网使用）
- LangGraph 用 pickle 反序列化（已知 RCE 风险，仅限可信环境）
- 无监控告警、无 DVC、无模型卡 / 数据卡

### 生产缺口
- 未经过监管审批（如银保监会的模型验证要求）
- 无压力测试 / 边界条件测试
- 无客户数据隐私保护设计
- 无版本化模型仓库

## 路线图

- [ ] Docker 镜像 + docker-compose
- [ ] GitHub Actions CI（lint + test + 数据 schema 校验）
- [ ] Streamlit 认证层（OAuth / LDAP）
- [ ] DVC + MLflow 模型版本化
- [ ] 模型卡 / 数据卡文档
- [ ] SHAP 全局摘要可视化
- [ ] 接入 Wind/巨潮等备用数据源

## 许可

[MIT](LICENSE) © 2026 xzxgsjgs
