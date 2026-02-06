# app/query_rewriter.py
import json
import re
from typing import Optional, List
from openai import OpenAI

KEYWORDS_PROMPT = """
Ets un assistent que converteix la pregunta actual en paraules clau per a cerca semàntica.

OBJECTIU
- Retorna NOMÉS paraules clau (keywords) en CATALÀ per cercar a Pinecone.
- Les keywords han de ser el mínim necessari per NO perdre informació.

REGLLES (molt estrictes)
- NO responguis la pregunta.
- NO expliquis res.
- NO inventis informació.
- Retorna NOMÉS JSON vàlid:
  {"keywords_ca":["...","..."]}

COM EXTREURE KEYWORDS
- Prioritza paraules que JA apareixen a la pregunta actual de l’usuari.
- NO afegeixis sinònims ni paraules “decoratives”.
- Mantén literals sigles/termes: "IPF", "DNI", "NIE", "PDF", "Word", "Excel", etc.
- Si hi ha CONTEXTE, usa’l NOMÉS per desambiguar i només amb 1–3 paraules clau extra.
- 2 a 8 keywords màxim. Sense duplicats. Sense punts finals.
- No afegeixis "pantalla", "títol", ni noms de pantalla si l’usuari NO ho ha dit explícitament.
""".strip()

_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ0-9_]+", re.UNICODE)

def _fallback_keywords(text: str, max_k: int = 6) -> List[str]:
    toks = _WORD_RE.findall((text or "").strip())
    seen = set()
    out = []
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
    """
    Devuelve effective_question como keywords en catalán (string separado por comas).
    """

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def rewrite(
        self,
        last_user: str,
        topic_question: Optional[str] = "",
        last_bot: Optional[str] = "",
        lang: str = "es",
        screen_hint: Optional[str] = "",  # ignorado
        root_question: Optional[str] = "",  # ignorado (compat)
    ) -> str:
        lu = (last_user or "").strip()

        user_block = f"""
PREGUNTA ACTUAL (usuari):
{lu}

CONTEXTE (opcional; pot ser buit):
- topic_question: {topic_question or "(cap)"}
- last_bot (retallat): {(last_bot or "")[:400]}
""".strip()

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": KEYWORDS_PROMPT},
                {"role": "user", "content": user_block},
            ],
            temperature=0.0,
        )

        text = (resp.choices[0].message.content or "").strip()

        try:
            data = json.loads(text)
            kws = data.get("keywords_ca") or []
            if not isinstance(kws, list):
                kws = []

            cleaned = []
            seen = set()
            for k in kws:
                ks = str(k).strip()
                if not ks:
                    continue
                ksl = ks.lower()
                if ksl in seen:
                    continue
                seen.add(ksl)
                cleaned.append(ks)
                if len(cleaned) >= 8:
                    break

            if cleaned:
                return ", ".join(cleaned)

            fb = _fallback_keywords(lu, max_k=6)
            return ", ".join(fb) if fb else lu

        except Exception:
            fb = _fallback_keywords(lu, max_k=6)
            return ", ".join(fb) if fb else lu
