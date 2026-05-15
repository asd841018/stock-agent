import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from src.tools.example import build_tools, TOOLS

load_dotenv()

def run_once(
    prompt: str,
    model: str = "gpt-5-nano"
):
    client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
    )
    messages =  [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    while True:
        print(messages)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=build_tools(),
        )
        
        msg = response.choices[0].message
        
        if not msg.tool_calls:
            print(msg.content)
            break
        
        # 重要:要把 assistant 的訊息(含 tool_calls)加回 messages，才能讓模型知道 tool_calls 是對應到哪一個訊息的
        messages.append(msg)
        
        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)
            # ★ 重點:LangChain tool 要用 .invoke(dict),不是 func(**dict)
            # func = tool_name_dict()[name]
            # tool_response = func(**args)
            try:
                result = TOOLS[name].invoke(args)
            except Exception as e:
                result = f"Error: {e}"
            # Invalid parameter: messages with role 'tool' must be a response to a preceeding message with 'tool_calls'
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": str(result),
            })
    return response

if __name__ == "__main__":
    prompt = "What is the weather in New York? Also, what is 5 + 7?"
    response = run_once(prompt)
