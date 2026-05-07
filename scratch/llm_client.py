import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()

class HelloAgentsLLM:
    
    def __init__(self, timeout: int = None):
        self.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))
        
        self.client = OpenAI(
            # This is the default and can be omitted
            api_key=openai_api_key,
        )
        
    def think(self, messages: List[Dict[str, Any]], temperature: float = 1.0) -> Dict[str, Any]:
        """
        Args:
            messages (List[Dict[str, Any]]): user and assistant messages to send to the model. Each message should have a "role" (e.g., "user", "assistant") and "content" (the text of the message).
            temperature (float, optional): The sampling temperature to use. Defaults to 0.0.

        Returns:
            Dict[str, Any]: The response from the model.
        """
        print(f"Calling model {self.model} with messages: {messages} and temperature: {temperature}")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            
            # 处理流式响应
            print("✅ 大语言模型回應成功:")
            collected_content = []
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()  # 在流式输出结束后換行
            return "".join(collected_content)

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None

# --- 客户端使用示例 ---
if __name__ == '__main__':
    try:
        llmClient = HelloAgentsLLM()
        
        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "寫一個快速排序演算法"}
        ]
        
        print("--- 调用LLM ---")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        print(e)