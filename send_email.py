import os
import resend
from dotenv import load_dotenv

load_dotenv()
resend.api_key = os.getenv("RESEND_API_KEY")

params = {
    "from": "Acme <onboarding@resend.dev>",   # 測試階段用這個
    "to": ["asd841018@gmail.com"],             # 只能寄到你註冊Resend的信箱
    "subject": "Hello from Resend",
    "html": "<h1>測試成功</h1><p>這是我第一封 Resend 寄出的信</p>",
}

try:
    email = resend.Emails.send(params)
    print(f"寄出成功，ID: {email['id']}")
except Exception as e:
    print(f"失敗: {e}")