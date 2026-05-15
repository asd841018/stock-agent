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
    return [convert_to_openai_tool(get_weather), convert_to_openai_tool(add)]
