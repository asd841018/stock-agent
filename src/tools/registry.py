from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from langchain_core.tools import BaseTool
from tools.email import get_email_tools
from tools.stock import get_stock_tools

class ToolProvider():
    def get_tools(self) -> Sequence[BaseTool]:
        """
        Returns a sequence of tools provided by this tool provider.
        """
        
        tools: list[BaseTool] = []
        for provider in (get_stock_tools, get_email_tools):
            tools.extend(provider())
        return tools