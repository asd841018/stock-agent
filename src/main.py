from __future__ import annotations

import argparse
import os

import dotenv

try:
    from src.agent_service import build_stock_agent
except ModuleNotFoundError:
    from agent_service import build_stock_agent


def build_user_message(message: str, recipient_email: str | None) -> str:
    email_clause = (
        f"如果你判斷可以偏多操作，請寄提醒信到 {recipient_email}。"
        if recipient_email
        else "如果沒有提供 email，就只需要回覆分析結果，不要寄信。"
    )
    return f"{message.strip()}\n\n{email_clause}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ReAct stock agent with optional email alerting.")
    parser.add_argument(
        "--message",
        default="請分析台積電今天的股價、技術指標與近期新聞，並判斷是否偏多。",
        help="給 agent 的自然語言問題。",
    )
    parser.add_argument(
        "--email",
        default=os.getenv("ALERT_EMAIL_TO"),
        help="如果 agent 判斷偏多時，要寄送提醒的收件者 email。",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL"),
        help="覆蓋預設的 OpenAI 模型名稱。",
    )
    return parser.parse_args()


def main() -> None:
    dotenv.load_dotenv()
    args = parse_args()
    agent = build_stock_agent(model_name=args.model)
    user_message = build_user_message(args.message, args.email)
    result = agent.invoke({"messages": [("user", user_message)]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
