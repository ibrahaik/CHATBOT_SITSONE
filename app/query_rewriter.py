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
- Devuelve SOLO JSON válido: {"standalone_question": "..."}.
- Si la pregunta usa referencias deícticas (p. ej. "esta pantalla", "aquí", "esto", "ahí", "la anterior"),
  debes resolverlas usando el CONTEXTO (pantalla/tema) proporcionado.
- Si hay un nombre de pantalla disponible, INCLÚYELO literalmente en la pregunta final.

FORMATO:
{"standalone_question":"..."}
""".strip()

DEICTIC_MARKERS = (
    "esta pantalla", "esa pantalla", "esta", "eso", "esto",
    "aquí", "ahi", "ahí", "allí",
    "la anterior", "lo anterior", "lo de antes",
    "esa", "ese", "eso de", "en esa", "en esa pantalla", "en esta", "en esta pantalla",
    "aqui", "acá", "aca",
)

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
        screen_hint: Optional[str] = "",
        root_question: Optional[str] = "",
    ) -> str:

        lu = (last_user or "").strip()
        needs_anchor = any(m in lu.lower() for m in DEICTIC_MARKERS)

        user_block = f"""
IDIOMA: {lang}

CONTEXTO DE PANTALLA / TEMA (si existe):
- screen_hint: {screen_hint or "(ninguno)"}
- root_question: {root_question or "(ninguna)"}

PREGUNTA ACTUAL DEL USUARIO:
{lu}

PREGUNTA COMPLETA ANTERIOR (si existe):
{topic_question or "(ninguna)"}

ÚLTIMA RESPUESTA DEL BOT (si ayuda):
{(last_bot or "")[:400]}

NOTA:
needs_anchor={needs_anchor}
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
            q = (data.get("standalone_question") or lu).strip()

            # Cinturón y tirantes: si sigue deíctico y hay ancla, lo forzamos.
            if needs_anchor and (screen_hint or root_question):
                anchor = (screen_hint or root_question).strip()
                if anchor and anchor.lower() not in q.lower():
                    # no lo hagas feo: añade literal y compacto
                    q = f"{q} (sobre: {anchor})"
            return q or lu
        except Exception:
            return lu
