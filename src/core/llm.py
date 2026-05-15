import os
from dotenv import load_dotenv
from openai import OpenAI
from src.tools.example import build_tools

load_dotenv()

def run_once(
    prompt: str,
    model: str = "gpt-5-nano"
):
    client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt,
            },
        ],
        tools=build_tools(),
    )
    return response

if __name__ == "__main__":
    prompt = "What is the weather in New York? Also, what is 5 + 7?"
    response = run_once(prompt)
    print(response)
