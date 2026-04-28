from __future__ import annotations

import os

from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from tools.registry import ToolProvider

SYSTEM_PROMPT = """你是一個台股 ReAct 交易助理。

你的工作流程：
1. 先用工具解析使用者提到的股票名稱或代碼。
2. 至少查看最新價格、移動平均與常用技術指標，必要時再查近期新聞。
3. 根據資料給出清楚結論：偏多、偏中立、或偏空。
4. 只有在「使用者明確提供 email」且你判斷「可以偏多操作」時，才呼叫 send_email_alert 寄信一次。
5. 如果資料不足、訊號不明確、或你不偏多，就不要寄信，並清楚說明原因。
6. 不要捏造股價、技術指標或新聞內容；不知道就說不知道。

寄信時請用簡短主旨，內文需包含：股票名稱/代碼、最新價格、移動平均、你的多空判斷、風險提醒。
"""


def build_stock_agent(model_name: str | None = None):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 OPENAI_API_KEY，請先在 .env 設定。")

    llm = ChatOpenAI(
        model=model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=api_key,
        temperature=0,
    )
    search = DuckDuckGoSearchRun()

    tool_provider = ToolProvider()
    return create_agent(
        llm,
        tools=tool_provider.get_tools(),
        system_prompt=SYSTEM_PROMPT,
    )
