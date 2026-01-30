# app/context.py
from typing import List
from app.config import RAGConfig
from app.retrieval.models import RetrievedItem
from app.utils import safe_str


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
                a = it.text or ""
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

                # ✅ SIN LIMPIEZA: pasamos el texto RAW tal cual venga de Pinecone
                raw_text = (it.text or "").strip()
                parts.append(f"  Text: {raw_text}".strip())

        out = "\n".join([p for p in parts if p.strip()]).strip()
        return out[: self.cfg.max_context_chars].strip()
