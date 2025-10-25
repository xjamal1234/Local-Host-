# server/app/core/secret_config.py
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    # اختياري: طباعة تحذير أو رمي استثناء حسب بيئتك
    # raise RuntimeError("OPENAI_API_KEY is not set")
    pass
