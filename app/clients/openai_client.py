import os
from openai import OpenAI

def get_openai_client() -> OpenAI:
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("Falta OPENAI_API_KEY en el entorno.")
    return OpenAI(api_key=key)
