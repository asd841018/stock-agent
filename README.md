# stock-agent

A minimal ReAct-style Taiwan stock agent built with LangChain, OpenAI, DuckDuckGo search, Taiwan stock tools, and optional email alerting.

## Quick Start

1. Put your OpenAI key in `.env`:

```env
OPENAI_API_KEY=...
```

2. Optional email alert settings for Gmail SMTP:

```env
ENABLE_EMAIL_SENDING=true
SMTP_HOST=smtp.gmail.com
SMTP_PORT=465
SMTP_USERNAME=your_gmail@gmail.com
SMTP_PASSWORD=your_app_password
SMTP_FROM=your_gmail@gmail.com
```

3. Run the agent:

```bash
.venv/bin/python -m src.main \
  --message "請分析台積電今天的股價、MA10 與近期新聞，判斷是否偏多" \
  --email "your_target@gmail.com"
```

If the agent concludes the stock is bullish and an email address is provided, it will send an alert email.

## Current Tool Wiring

The current implementation works, but `src/agent_service.py` directly imports every tool and manually lists them in `create_agent(...)`.

That means every time a new tool is added, `agent_service.py` must be modified:

```python
return create_agent(
    llm,
    tools=[
        resolve_taiwan_stock,
        get_stock_price,
        get_moving_average,
        send_email_alert,
        search,
    ],
    system_prompt=SYSTEM_PROMPT,
)
```

This is simple at first, but it does not follow the Open/Closed Principle well:

- Open for extension: adding new stock, email, news, or portfolio tools should be easy.
- Closed for modification: stable agent wiring should not need to change every time a new tool is added.

## Recommended OCP Architecture

Move tool discovery and composition into a small registry layer.

`src/agent_service.py` should only ask for the available tools. It should not know the name of every tool function.

Recommended structure:

```text
src/
  agent_service.py
  main.py
  tools/
    __init__.py
    registry.py
    stock.py
    email.py
    search.py
```

The responsibilities become:

- `agent_service.py`: builds the LLM and agent.
- `tools/registry.py`: owns tool collection and ordering.
- `tools/stock.py`: owns stock-related tools.
- `tools/email.py`: owns email-related tools.
- `tools/search.py`: owns search-related tool creation.

## Registry Design

Create a registry module:

```python
# src/tools/registry.py
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from langchain_core.tools import BaseTool


class ToolProvider(Protocol):
    def get_tools(self) -> Sequence[BaseTool]:
        """Return LangChain tools provided by this module."""


def build_tools() -> list[BaseTool]:
    from src.tools.email import get_tools as get_email_tools
    from src.tools.search import get_tools as get_search_tools
    from src.tools.stock import get_tools as get_stock_tools

    tools: list[BaseTool] = []
    for provider in (
        get_stock_tools,
        get_email_tools,
        get_search_tools,
    ):
        tools.extend(provider())
    return tools
```

Then update each tool module to expose a `get_tools()` function.

Example for stock tools:

```python
# src/tools/stock.py

def get_tools():
    return [
        resolve_taiwan_stock,
        get_stock_price,
        get_moving_average,
        get_kd_indicator,
        get_rsi_indicator,
        get_bollinger_bands,
        get_technical_indicators,
        get_stock_snapshot,
    ]
```

Example for email tools:

```python
# src/tools/email.py

def get_tools():
    return [
        send_email_alert,
    ]
```

Example for search tools:

```python
# src/tools/search.py
from __future__ import annotations

from langchain_community.tools import DuckDuckGoSearchRun


def get_tools():
    return [
        DuckDuckGoSearchRun(),
    ]
```

Finally, `agent_service.py` becomes closed to tool changes:

```python
from src.tools.registry import build_tools


def build_stock_agent(model_name: str | None = None):
    ...
    return create_agent(
        llm,
        tools=build_tools(),
        system_prompt=SYSTEM_PROMPT,
    )
```

## How To Add A New Tool

With this design, adding a new tool usually requires only one of these paths.

### Add to an existing category

If the tool belongs to an existing module, define the tool in that module and add it to that module's `get_tools()` list.

For example, a new stock valuation tool can live in `src/tools/stock.py`:

```python
@tool
def get_pe_ratio(stock_query: str) -> str:
    """當需要本益比估值資訊時使用。輸入 stock_query，可為台股代碼或中文名稱。"""
    ...
```

Then expose it:

```python
def get_tools():
    return [
        resolve_taiwan_stock,
        get_stock_price,
        get_pe_ratio,
    ]
```

No change is needed in `agent_service.py`.

### Add a new category

If the tool belongs to a new domain, create a new module:

```text
src/tools/portfolio.py
```

Example:

```python
from __future__ import annotations

from langchain_core.tools import tool


@tool
def calculate_position_size(capital: float, risk_percent: float, stop_loss_percent: float) -> str:
    """當需要根據資金、風險比例與停損幅度計算部位大小時使用。"""
    if capital <= 0:
        raise ValueError("capital 必須大於 0。")
    if risk_percent <= 0 or stop_loss_percent <= 0:
        raise ValueError("risk_percent 與 stop_loss_percent 必須大於 0。")

    risk_amount = capital * risk_percent / 100
    position_size = risk_amount / (stop_loss_percent / 100)
    return f"建議部位金額約為 {position_size:.2f}，最大風險金額約為 {risk_amount:.2f}。"


def get_tools():
    return [
        calculate_position_size,
    ]
```

Then register only the provider in `src/tools/registry.py`:

```python
from src.tools.portfolio import get_tools as get_portfolio_tools

for provider in (
    get_stock_tools,
    get_email_tools,
    get_search_tools,
    get_portfolio_tools,
):
    tools.extend(provider())
```

This keeps the agent builder stable while still making the toolset extensible.

## Optional Plugin-Style Registry

If you want the registry itself to avoid import edits, use a plugin-style provider list.

Example:

```python
# src/tools/registry.py
from __future__ import annotations

from importlib import import_module


TOOL_PROVIDER_MODULES = [
    "src.tools.stock",
    "src.tools.email",
    "src.tools.search",
]


def build_tools():
    tools = []
    for module_name in TOOL_PROVIDER_MODULES:
        module = import_module(module_name)
        tools.extend(module.get_tools())
    return tools
```

This is more open for extension, but slightly more implicit. For this project, the explicit provider registry is usually the better first step because it is easy to read, easy to test, and still removes tool-level imports from `agent_service.py`.

## Testing Strategy

At minimum, keep these checks:

```bash
.venv/bin/python -m compileall src
```

If tests are added or changed:

```bash
.venv/bin/python -m pytest
```

Recommended tests for the OCP refactor:

- `tests/test_tool_registry.py` verifies `build_tools()` returns all expected tool names.
- Existing stock and email tests continue to test tool behavior directly.
- External services should remain mocked: OpenAI, DuckDuckGo, SMTP, and live stock-data calls.

Example registry test:

```python
from src.tools.registry import build_tools


def test_build_tools_includes_core_tools():
    names = {tool.name for tool in build_tools()}

    assert "resolve_taiwan_stock" in names
    assert "get_stock_price" in names
    assert "send_email_alert" in names
```

## Design Rules For New Tools

- Keep each `@tool` function small and deterministic.
- Write a clear docstring that tells the agent when to call the tool.
- Return explicit strings that are useful to the final answer.
- Validate inputs close to the tool boundary.
- Avoid hidden side effects unless the tool name and docstring make the side effect obvious.
- Keep network, email, or API calls easy to mock in tests.
- Do not put business-specific tool imports back into `agent_service.py`.

## Suggested Migration Steps

1. Add `src/tools/registry.py`.
2. Add `get_tools()` to `src/tools/stock.py`.
3. Add `get_tools()` to `src/tools/email.py`.
4. Move `DuckDuckGoSearchRun()` into `src/tools/search.py`.
5. Replace direct tool imports in `src/agent_service.py` with `build_tools()`.
6. Add `tests/test_tool_registry.py`.
7. Run compile and tests.

After this migration, most future tool changes should happen inside `src/tools/` only. The agent builder remains stable, and the project becomes open for extension while closed for repeated modification.
