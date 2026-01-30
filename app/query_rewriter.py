# app/query_rewriter.py
import json
import re
from typing import Optional, List, Dict, Any
from openai import OpenAI

# Extrae KEYWORDS en catalán, sin responder, sin frases largas.
KEYWORDS_PROMPT = """
Ets un assistent que transforma una conversa en paraules clau per a cerca semàntica.

OBJECTIU:
Donada la conversa, retorna NOMÉS paraules clau (keywords) en CATALÀ, curtes i útils per cercar a Pinecone.

REGLLES:
- NO responguis la pregunta.
- NO expliquis res.
- NO inventis informació.
- Retorna NOMÉS JSON vàlid amb aquesta forma:
  {"keywords_ca":["...","...","..."]}

- Sempre en català (encara que l'usuari escrigui en castellà).
- Mantén literalment sigles i termes tècnics (p.ex. "IPF", "DNI", "NIE", "PDF", "Word", "Excel").
- Si la pregunta és massa curta o deíctica ("i això?", "i notificades?"), incorpora 1–3 keywords del context disponible
  (root_question / topic_question / last_bot) però SENSE afegir frases.
- 4 a 10 keywords màxim. Sense duplicats. Sense punts finals.
- No afegeixis "pantalla/tema:" ni parèntesis ni textos llargs.
""".strip()

# Heurística mínima si el model retorna mal JSON
_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ0-9_]+", re.UNICODE)

def _fallback_keywords_ca(text: str, max_k: int = 8) -> List[str]:
    t = (text or "").strip()
    toks = _WORD_RE.findall(t)
    # fallback ultra simple: lower + uniques preservant ordre
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
    ATENCIÓ:
    - Abans: reescrivia a pregunta standalone.
    - Ara: genera un 'effective_question' com a keywords en català, separades per comes.
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
        screen_hint: Optional[str] = "",   # ignorat per disseny
        root_question: Optional[str] = "",
    ) -> str:
        lu = (last_user or "").strip()

        user_block = f"""
IDIOMA_ULTIM_MISSATGE: {lang}

PREGUNTA ACTUAL (usuari):
{lu}

CONTEXTE (si ajuda per desambiguar preguntes curtes):
- root_question: {root_question or "(cap)"}
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
            # neteja
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
                if len(cleaned) >= 10:
                    break

            if cleaned:
                # efectiu: keywords separades per comes (string curt)
                return ", ".join(cleaned)

            # fallback si JSON però buit
            fb = _fallback_keywords_ca(" ".join([lu, topic_question or "", root_question or ""]), max_k=8)
            return ", ".join(fb) if fb else lu

        except Exception:
            fb = _fallback_keywords_ca(" ".join([lu, topic_question or "", root_question or ""]), max_k=8)
            return ", ".join(fb) if fb else lu
