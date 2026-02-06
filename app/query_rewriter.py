# app/query_rewriter.py
import json
import re
from typing import Optional, List
from openai import OpenAI

DEPENDENCY_PROMPT = """
Decide si la pregunta del usuario DEPENDE de la última respuesta del asistente
o si es una pregunta NUEVA e independiente.

Responde SOLO JSON válido:
{"depends_on_last_answer": true | false}
""".strip()

KEYWORDS_PROMPT = """
Ets un assistent que converteix la pregunta actual en paraules clau per a cerca semàntica.

OBJECTIU
- Retorna NOMÉS paraules clau (keywords) en CATALÀ per cercar a Pinecone.

REGLLES
- NO responguis la pregunta.
- Retorna NOMÉS JSON:
  {"keywords_ca":["...","..."]}
""".strip()

_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ0-9_]+", re.UNICODE)

def _fallback_keywords(text: str, max_k: int = 6) -> List[str]:
    toks = _WORD_RE.findall((text or "").strip())
    out = []
    seen = set()
    for w in toks:
        wl = w.lower()
        if wl in seen:
            continue
        seen.add(wl)
        out.append(wl)
        if len(out) >= max_k:
            break
    return out


class QueryRewriter:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def _depends_on_context(self, last_user: str, last_bot: str) -> bool:
        if not last_bot:
            return False

        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": DEPENDENCY_PROMPT},
                {
                    "role": "user",
                    "content": f"""
Última respuesta del asistente:
{last_bot[:400]}

Pregunta del usuario:
{last_user}
""".strip(),
                },
            ],
        )

        try:
            data = json.loads(resp.choices[0].message.content)
            return bool(data.get("depends_on_last_answer"))
        except Exception:
            return False

    def rewrite(
        self,
        last_user: str,
        last_bot: Optional[str] = "",
    ) -> str:
        use_context = self._depends_on_context(last_user, last_bot or "")

        user_block = f"""
PREGUNTA ACTUAL:
{last_user}

CONTEXTE:
{last_bot[:300] if use_context else "(cap)"}
""".strip()

        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": KEYWORDS_PROMPT},
                {"role": "user", "content": user_block},
            ],
        )

        try:
            data = json.loads(resp.choices[0].message.content)
            kws = data.get("keywords_ca") or []
            return ", ".join(kws) if kws else ", ".join(_fallback_keywords(last_user))
        except Exception:
            return ", ".join(_fallback_keywords(last_user))
