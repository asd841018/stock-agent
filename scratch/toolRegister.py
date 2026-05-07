from typing import List, Dict, Any

class ToolExecutor:
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
    
    def register_tool(self, name: str, description: str, func: callable):
        """
        Register a tool that the agent can use.
        """
        if name in self.tools:
            print(f"工具 '{name}' 已经注册，覆盖旧的定义。")
        self.tools[name] = {
            "description": description,
            "func": func,
        }
        print(f"工具 '{name}' 注册成功")
        
    def getTool(self, name: str) -> callable:
        """
        根據名稱獲取工具函數
        """
        return self.tools.get(name, {}).get("func")
        
    def getAvailableTools(self) -> str:
        """
        獲取所有可用工具的名稱和描述
        """
        return "\n".join([
            f"{name}: {info['description']}" 
            for name, info in self.tools.items()
            ])