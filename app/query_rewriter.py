# app/query_rewriter.py
import json
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from openai import OpenAI


@dataclass
class RewriteResult:
    effective_query: str
    use_context: bool
    mode: str
    confidence: float
    clarify: bool
    clarify_question_es: str
    keywords_ca: List[str]


REWRITER_PROMPT = (
    "Ets un assistent de reescriptura per a RAG per un sistema intern de Area Metropolitana de Barcelona.\n"
    "\n"
    "IMPORTANT:\n"
    "- NO responguis la pregunta de l'usuari.\n"
    "- Retorna NOMÉS JSON vàlid (cap text extra).\n"
    "\n"
    "FORMAT DE SORTIDA (obligatori):\n"
    "{\n"
    '  "mode": "new_topic" | "follow_up" | "ambiguous",\n'
    '  "use_context": true | false,\n'
    '  "confidence": 0.0,\n'
    '  "keywords_ca": ["..."],\n'
    '  "clarify": true | false,\n'
    '  "clarify_question_es": "..."\n'
    "}\n"
    "\n"
    "DEFINICIONS (noms exactes):\n"
    "- PREGUNTA_ACTUAL: el missatge de l'usuari actual.\n"
    "- ÚLTIMA_RESPOSTA: l'última resposta del sistema.\n"
    "- CONTEXT: PREGUNTA_ACTUAL + ÚLTIMA_RESPOSTA.\n"
    "- FONT: l'origen del coneixement utilitzat a ÚLTIMA_RESPOSTA (p.ex. wiki: \"Gestió Persones\" o faq: \"banda magnètica\").\n"
    "\n"
    "MÈTODE ÚNIC (sense casos):\n"
    "1) Extreu el TEMA PRINCIPAL de ÚLTIMA_RESPOSTA en 1–3 paraules (substantiu/s).\n"
    "2) Extreu el TEMA PRINCIPAL de PREGUNTA_ACTUAL en 1–3 paraules (substantiu/s).\n"
    "3) Si PREGUNTA_ACTUAL tracta d’un detall/element/subfunció dins del mateix tema,\n"
    "   considera que el tema és el mateix.\n"
    "4) Si el tema principal és el mateix → mode=\"follow_up\" i use_context=true.\n"
    "5) Si el tema principal és diferent → mode=\"new_topic\" i use_context=false.\n"
    "6) Si no pots decidir amb seguretat → mode=\"ambiguous\", clarify=true i pregunta curta en castellà.\n"
    "\n"
    "DEFINICIÓ DE TEMA PRINCIPAL:\n"
    "- El tema principal és el concepte central del domini (substantiu).\n"
    "- Ignora connectors, muletilles i detalls accessoris.\n"
    "\n"
    "KEYWORDS (molt estrictes):\n"
    "- 2 a 8 keywords màxim.\n"
    "- Sense duplicats (case-insensitive).\n"
    "- NO afegeixis sinònims ni paraules decoratives.\n"
    "- PRIORITZA paraules literals de PREGUNTA_ACTUAL.\n"
    "- Mantén literals sigles/termes: IPF, DNI, NIE, T4, TMB, etc.\n"
    "- PROHIBIT: no retornis connectors/buides com \"y\", \"sobre\", \"por\", \"la\", \"el\", \"un\", \"una\".\n"
    "\n"
    "COMPOSICIÓ DE KEYWORDS QUAN HI HA FOLLOW_UP:\n"
    "- Si mode=\"follow_up\" i use_context=true:\n"
    "  - keywords_ca ha d'incloure:\n"
    "    a) 2–5 keywords literals de PREGUNTA_ACTUAL\n"
    "    b) +1–2 keywords que representin el TEMA PRINCIPAL de FONT\n"
    "  - Aquestes keywords de FONT poden no aparèixer literalment a PREGUNTA_ACTUAL.\n"
    "  - Prioritza keywords compostes si FONT és una pantalla o mòdul (p.ex. \"Gestió Persones\").\n"
    "- Si mode!=\"follow_up\" o use_context=false:\n"
    "  - keywords_ca només de PREGUNTA_ACTUAL.\n"
    "\n"
    "OBLIGATORI (només si mode=\"follow_up\" i use_context=true):\n"
    "- Si FONT correspon a una pantalla o mòdul amb nom explícit,\n"
    "  keywords_ca HA D'INCLOURE obligatòriament el nom de la pantalla com a keyword composta.\n"
    "- No és vàlid substituir el nom de la pantalla per descripcions funcionals o conceptes genèrics.\n"
    "- Exemple:\n"
    "  FONT = \"Wiki 1.5. Gestió Retorns\"\n"
    "  → keywords_ca ha d'incloure \"Gestió Retorns\".\n"
    "VALIDACIÓ (auto-control):\n"
    "- Si inclous una paraula buida/prohibida a keywords_ca, la resposta és incorrecta.\n"
    "- En aquest cas, corregeix-te i retorna keywords_ca sense paraules prohibides.\n"
    "\n"
    "ACLARIMENT:\n"
    "- Si mode=\"ambiguous\" → clarify=true i clarify_question_es no buit.\n"
    "- La pregunta d'aclariment ha de ser molt curta i en castellà.\n"
).strip()


# Captura la línea completa Fuente/Font/Source:
#   Font: Wiki “1.2. GESTIÓ Persones” — actualitzat 2025-06
#   Fuente: Wiki “1.7 Gestió de lliuraments” — actualizado 2025-06
#   Fuente: FAQ — <tema>
# y extrae el texto después de "Fuente/Font/Source:" recortando el sufijo de actualización si existe.
_SOURCE_LINE_RE = re.compile(r"(?im)^[ \t]*(Fuente|Font|Source)\s*:\s*(?P<rest>.+?)\s*$")
_TAIL_UPDATED_RE = re.compile(
    r"\s*(?:—|–|-)\s*(?:actualitzat|actualizado|updated)\b.*$",
    re.IGNORECASE,
)

# Quita comillas tipográficas o normales alrededor del bloque, si las hay
_SURROUNDING_QUOTES_RE = re.compile(r'^[\s"“”\'‘’]+|[\s"“”\'‘’]+$')


def extract_font_from_last_bot(last_bot: str) -> str:
    """
    Extrae FONT de la respuesta del bot si viene embebida en una línea Fuente/Font/Source.
    Soporta WIKI (con "— actualizado/actualitzat/updated ...") y FAQ (sin sufijo).
    Devuelve "" si no encuentra.
    """
    if not last_bot:
        return ""

    m = _SOURCE_LINE_RE.search(last_bot)
    if not m:
        return ""

    rest = (m.group("rest") or "").strip()

    # Normaliza espacios
    rest = re.sub(r"\s+", " ", rest).strip()

    # Recorta sufijo de actualización si existe
    rest = _TAIL_UPDATED_RE.sub("", rest).strip()

    # Quita comillas envolventes si existen
    rest = _SURROUNDING_QUOTES_RE.sub("", rest).strip()

    return rest


class QueryRewriter:
    """
    Backend ultra-minimal: NO limpia, NO corrige, NO fuerza reglas.
    Solo:
      - envía PREGUNTA_ACTUAL + ÚLTIMA_RESPOSTA + FONT al LLM
      - parsea JSON
      - devuelve exactamente lo que el LLM diga
    """

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def rewrite(
        self,
        last_user: str,
        last_bot: Optional[str] = "",
        font: Optional[str] = "",  # opcional: si no llega, se intenta extraer de last_bot
        lang: str = "es",
    ) -> RewriteResult:
        pregunta_actual = last_user or ""
        ultima_resposta = last_bot or ""

        font_str = (font or "").strip()
        if not font_str:
            font_str = extract_font_from_last_bot(ultima_resposta)

        user_block = (
            "PREGUNTA_ACTUAL:\n"
            f"{pregunta_actual}\n"
            "\n"
            "ÚLTIMA_RESPOSTA:\n"
            f"{ultima_resposta if ultima_resposta else '(cap)'}\n"
            "\n"
            "FONT:\n"
            f"{font_str if font_str else '(cap)'}\n"
        ).strip()

        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": REWRITER_PROMPT},
                {"role": "user", "content": user_block},
            ],
        )

        raw = (resp.choices[0].message.content or "").strip()

        # Defaults "seguros" SOLO si el JSON es inválido.
        mode = "ambiguous"
        use_context = False
        confidence = 0.0
        clarify = True
        clarify_q = "¿Puedes concretar a qué te refieres?"
        keywords: List[str] = []

        try:
            data: Dict[str, Any] = json.loads(raw)

            mode = str(data.get("mode") or "ambiguous").strip()
            if mode not in ("new_topic", "follow_up", "ambiguous"):
                mode = "ambiguous"

            use_context = bool(data.get("use_context", False))

            try:
                confidence = float(data.get("confidence", 0.0))
            except Exception:
                confidence = 0.0

            if confidence < 0.0:
                confidence = 0.0
            if confidence > 1.0:
                confidence = 1.0

            kws = data.get("keywords_ca") or []
            if isinstance(kws, list):
                keywords = [str(k) for k in kws if str(k).strip()]
            else:
                keywords = []

            clarify = bool(data.get("clarify", False))
            clarify_q = str(data.get("clarify_question_es") or "").strip()

        except Exception:
            pass

        effective_query = ", ".join(keywords).strip()

        return RewriteResult(
            effective_query=effective_query,
            use_context=use_context,
            mode=mode,
            confidence=confidence,
            clarify=clarify,
            clarify_question_es=clarify_q,
            keywords_ca=keywords,
        )
