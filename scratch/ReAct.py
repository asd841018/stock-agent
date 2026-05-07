import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any
from llm_client import HelloAgentsLLM

from toolRegister import ToolExecutor
from tools.searchTool import search

load_dotenv()

# ReAct 提示词模板
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `{{tool_name}}[{{tool_input}}]`:调用一个可用工具。
- `Finish[最终答案]`:当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用 Finish[最终答案] 来输出最终答案。

现在，请开始解决以下问题:
Question: {question}
History: {history}
"""





class ReActAgent:
    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []
        
            # (这些方法是 ReActAgent 类的一部分)
    def _parse_output(self, text: str):
        """解析LLM的输出，提取Thought和Action。
        """
        # Thought: 匹配到 Action: 或文本末尾
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        # Action: 匹配到文本末尾
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """解析Action字符串，提取工具名称和输入。
        """
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def run(self, question: str):
        """
        ReAct agent for answering questions
        """
        self.history = []
        current_step = 0
        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- Step {current_step} ---\n")
            
            # 1. 格式化提示詞
            tools_description = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_description,
                question=question,
                history=history_str,
            )
            
            # 2. 调用模型获取回应
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages)
            
            if not response_text:
                print("错误:LLM未能返回有效响应。")
                break
            
            # 3. 解析模型输出
            thought, action = self._parse_output(response_text)
            
            if thought:
                print(f"Thought: {thought}")
            
            if not action:
                print(f"no action found in LLM response, treating as final answer.")
                break
            
            if action.startswith("Finish"):
                # 如果是Finish指令，提取最终答案并结束
                final_answer = re.match(r"Finish\[(.*)\]", action).group(1)
                print(f"🎉 最终答案: {final_answer}")
                return final_answer
            
            toolname, toolinput = self._parse_action(action)
            if not toolname or not toolinput:
                print(f"无法解析Action")
                continue
            
            print(f"Action: 调用工具 {toolname}，输入: {toolinput}")
            
            tool_function = self.tool_executor.getTool(toolname)
            if not tool_function:
                observation = f"工具 '{toolname}' 不存在。"
            else:
                observation = tool_function(toolinput)
                
            print(f"👀 观察: {observation}")
            
            # 将本轮的Action和Observation添加到历史记录中
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")
        
        print("⚠️ 达到最大步骤数，未能得到最终答案。")
        return None
    
if __name__ == '__main__':
    llm = HelloAgentsLLM()
    tool_executor = ToolExecutor()
    search_desc = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_executor.register_tool("Search", search_desc, search)
    agent = ReActAgent(llm_client=llm, tool_executor=tool_executor)
    question = "华为最新的手机是哪一款？它的主要卖点是什么？"
    agent.run(question)