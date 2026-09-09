"""credit_agent.agent — LangGraph 工作流编排

将阶段一 17 个确定性 Python 工具编排为可观测、可中断、可恢复的 DAG。
架构原则：所有数值计算由 tools/ 完成，agent 层仅做编排与异常捕获。
LLM 不直接生成数值结果 — 仅在阶段三 Streamlit UI 层供人决策。
"""