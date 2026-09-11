"""credit_agent.app —— Streamlit 双页应用。

入口为 Home.py，pages/ 下为子页面。
所有模块通过 `app.<subpkg>` 命名空间导入，依赖本文件使 app 成为合法 Python 包。
"""

from __future__ import annotations

from pathlib import Path

# 加载项目根目录的 .env（如果存在）。Streamlit 启动时以 credit_agent 为工作目录，
# 因此 Path.cwd()/.env 即为目标文件；幂等调用，不会覆盖已存在的环境变量。
_dotenv_path = Path.cwd() / ".env"
if _dotenv_path.exists():
    try:
        from dotenv import load_dotenv

        load_dotenv(_dotenv_path)
    except ImportError:
        pass  # 用户未安装 python-dotenv 时降级；但 .env 不会被加载
