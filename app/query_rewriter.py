# app/query_rewriter.py
import json
from typing import Optional

from openai import OpenAI


REWRITE_PROMPT = """
Eres un asistente que reescribe preguntas para búsqueda.

OBJETIVO:
Dada una conversación, genera UNA sola pregunta auto-contenida
que represente exactamente lo que el usuario quiere saber ahora.

REGLAS:
- NO respondas la pregunta.
- NO expliques nada.
- NO inventes información.
- Usa solo la información implícita del contexto.
- Devuelve SOLO JSON válido.

FORMATO:
{
  "standalone_question": "..."
}
"""


class QueryRewriter:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def rewrite(
        self,
        last_user: str,
        topic_question: Optional[str] = "",
        last_bot: Optional[str] = "",
        lang: str = "es",
    ) -> str:
        user_block = f"""
IDIOMA: {lang}

PREGUNTA ACTUAL DEL USUARIO:
{last_user}

PREGUNTA COMPLETA ANTERIOR (si existe):
{topic_question or "(ninguna)"}

ÚLTIMA RESPUESTA DEL BOT (si ayuda):
{(last_bot or "")[:400]}
""".strip()

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": REWRITE_PROMPT},
                {"role": "user", "content": user_block},
            ],
            temperature=0.0,
        )

        text = (resp.choices[0].message.content or "").strip()
        try:
            data = json.loads(text)
            return (data.get("standalone_question") or last_user).strip()
        except Exception:
            return last_user
