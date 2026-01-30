# app/context.py
from typing import List
from app.config import RAGConfig
from app.retrieval.models import RetrievedItem
from app.utils import safe_str
import re

_ROLE_RE = re.compile(r"(?im)^\s*Rol\s*:\s*.*$", re.MULTILINE)
_KEYWORDS_RE = re.compile(r"(?im)^\s*Keywords\s*:\s*.*$", re.MULTILINE)
_BLOC_RE = re.compile(r"(?im)^\s*Bloc\s*:\s*.*$", re.MULTILINE)
_BDEF_RE = re.compile(r"(?im)^\s*Definició\s+bloc\s*:\s*.*$", re.MULTILINE)
_TITLE_RE = re.compile(r"(?im)^\s*T.{0,2}tol\s*:\s*.*$", re.MULTILINE)
_TYPE_RE = re.compile(r"(?im)^\s*Tipus\s*:\s*.*$", re.MULTILINE)
_CONTEXT_RE = re.compile(r"(?im)^\s*Context\s*:\s*.*$", re.MULTILINE)
_BULLET_RE = re.compile(r"(?im)^\s*[\*\-]\s+.*$")


def _llm_body(text: str) -> str:
    """
    Extrae el cuerpo util del chunk para el LLM:
      - Corta todo lo que venga despues de "Keywords:".
      - Elimina cabeceras Titol/Tipus/Rol/Bloc/Definici bloc/Context.
      - Si hay bullets, devuelve bullets.
      - Si hay Context: en el original, devuelve las lineas tras Context: (sin cabeceras).
      - Si no, devuelve el primer párrafo real.
    """
    t = (text or "").strip()
    if not t:
        return ""

    # 1) Cortar todo lo que venga despues de "Keywords:"
    m_kw = _KEYWORDS_RE.search(t)
    if m_kw:
        t = t[:m_kw.start()].strip()

    lines = [ln.rstrip() for ln in t.splitlines()]

    # 2) Limpieza de cabeceras repetidas
    cleaned = []
    for ln in lines:
        if _TITLE_RE.match(ln) or _TYPE_RE.match(ln) or _ROLE_RE.match(ln):
            continue
        if _BLOC_RE.match(ln) or _BDEF_RE.match(ln):
            continue
        if _KEYWORDS_RE.match(ln):
            continue
        if _CONTEXT_RE.match(ln):
            continue
        cleaned.append(ln)

    # 3) Si hay bullets, devolverlos
    bullets = [ln.strip() for ln in cleaned if _BULLET_RE.match(ln)]
    if bullets:
        return "\n".join(bullets).strip()

    # 4) Si hay "Context:" en el texto original, devolver lineas tras esa cabecera
    if _CONTEXT_RE.search(t):
        after_ctx = []
        seen_ctx = False
        for ln in t.splitlines():
            if _CONTEXT_RE.match(ln):
                seen_ctx = True
                continue
            if not seen_ctx:
                continue
            if _KEYWORDS_RE.match(ln):
                break
            if _BLOC_RE.match(ln) or _BDEF_RE.match(ln):
                continue
            if _TITLE_RE.match(ln) or _TYPE_RE.match(ln) or _ROLE_RE.match(ln):
                continue
            if ln.strip():
                after_ctx.append(ln.strip())
        if after_ctx:
            return "\n".join(after_ctx).strip()

    # 5) Primer parrafo real (no vacio) tras limpieza
    para = []
    for ln in cleaned:
        if not ln.strip():
            if para:
                break
            continue
        if para and ln.strip() == para[-1].strip():
            continue
        para.append(ln.strip())
    return "\n".join(para).strip()


def _to_int(x, default=None):
    try:
        if x is None:
            return default
        return int(str(x).strip())
    except Exception:
        return default


class ContextBuilder:
    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg

    def build_faqs(self, faqs: List[RetrievedItem]) -> str:
        parts: List[str] = []
        if faqs:
            parts.append("FAQs:")
            for i, it in enumerate(faqs[: self.cfg.top_k_faqs], start=1):
                q = safe_str(it.meta.get("question") or it.meta.get("q") or "")
                a = it.text
                parts.append(f"- [FAQ{i}] {('Q: ' + q + ' ' if q else '')}A: {a}".strip())
        out = "\n".join([p for p in parts if p.strip()]).strip()
        return out[: self.cfg.max_context_chars].strip()

    def build_articles(self, arts: List[RetrievedItem]) -> str:
        # orden estable por article_id + chunk_index
        def sort_key(it: RetrievedItem):
            m = it.meta or {}
            aid = safe_str(m.get("article_id"))
            idx = _to_int(m.get("chunk_index"), default=10**9)
            return (aid, idx)

        arts = sorted(arts or [], key=sort_key)

        block_limit = int(self.cfg.top_k_articles_block_context or 15)

        parts: List[str] = []
        if arts:
            parts.append("Articles:")
            for i, it in enumerate(arts[:block_limit], start=1):
                m = it.meta or {}
                titulo = safe_str(m.get("titulo") or "")
                bloc = safe_str(m.get("block_title") or "")
                bdef = safe_str(m.get("block_definition") or "")
                role = safe_str(m.get("chunk_role") or "")
                idx = safe_str(m.get("chunk_index") or "")
                aid = safe_str(m.get("article_id") or "")

                header = f"- [A{i}] article_id={aid} chunk_index={idx}".strip()
                if titulo:
                    header += f" | Títol: {titulo}"
                if bloc:
                    header += f" | Bloc: {bloc}"
                if role:
                    header += f" | Rol: {role}"
                if bdef:
                    header += f" | Definició bloc: {bdef}"
                parts.append(header)

                body = _llm_body(it.text)
                parts.append(f"  Text: {body}".strip())

        out = "\n".join([p for p in parts if p.strip()]).strip()
        return out[: self.cfg.max_context_chars].strip()

    @staticmethod
    def clean_article_text(text: str) -> str:
        return _llm_body(text)
