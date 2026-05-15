from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

class WeatherInput(BaseModel):
    location: str = Field(..., description="The location to get the weather for")
    
class AddInput(BaseModel):
    a: int = Field(..., description="The first number to add")
    b: int = Field(..., description="The second number to add")

@tool("get_weather", args_schema=WeatherInput)
def get_weather(location: str) -> str:
    # 給tool的描述必須要有，否則會被OpenAI拒絕，因為他們需要知道這個tool是幹嘛的
    """Get the current weather for a location."""
    return f"{location} is sunny with a high of 25°C."

@tool("add", args_schema=AddInput)
def add(a: int, b: int) -> int:
    """Add two integers together."""
    return a + b


def build_tools():
    # for openai model can see what tools are available, we need to convert the langchain tool to openai tool
    return [convert_to_openai_tool(get_weather), convert_to_openai_tool(add)]

# 1. 用 tool.name 當 key 建 registry
TOOLS = {t.name: t for t in [get_weather, add]}