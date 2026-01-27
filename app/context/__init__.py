from typing import List
from app.config import RAGConfig
from app.retrieval.models import RetrievedItem
from app.utils import safe_str

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
        parts: List[str] = []
        if arts:
            parts.append("Articles:")
            for i, it in enumerate(arts[: self.cfg.top_k_articles_final], start=1):
                titulo = safe_str(it.meta.get("titulo") or "")
                bloc = safe_str(it.meta.get("block_title") or "")
                bdef = safe_str(it.meta.get("block_definition") or "")
                role = safe_str(it.meta.get("chunk_role") or "")
                txt = it.text

                header = f"- [A{i}] Títol: {titulo}".strip()
                if bloc:
                    header += f" | Bloc: {bloc}"
                if role:
                    header += f" | Rol: {role}"
                if bdef:
                    header += f" | Definició bloc: {bdef}"
                parts.append(header)
                parts.append(f"  Text: {txt}".strip())

        out = "\n".join([p for p in parts if p.strip()]).strip()
        return out[: self.cfg.max_context_chars].strip()
