from typing import List
from app.config import RAGConfig
from app.retrieval.models import RetrievedItem
from app.utils import safe_str

class ContextBuilder:
    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg

    def build(self, faqs: List[RetrievedItem], arts: List[RetrievedItem]) -> str:
        parts: List[str] = []

        if faqs:
            parts.append("FAQs (prioritat):")
            for i, it in enumerate(faqs[: self.cfg.top_k_faqs], start=1):
                q = safe_str(it.meta.get("question") or it.meta.get("q") or "")
                a = it.text
                parts.append(f"- [FAQ{i}] {('Q: ' + q + ' ' if q else '')}A: {a}".strip())

        if arts:
            parts.append("\nArticles (fallback):")
            for i, it in enumerate(arts[: self.cfg.top_k_articles], start=1):
                titulo = safe_str(it.meta.get("titulo") or "")
                bloc = safe_str(it.meta.get("block_title") or "")
                bdef = safe_str(it.meta.get("block_definition") or "")
                txt = it.text

                header = f"- [A{i}] Títol: {titulo}".strip()
                if bloc:
                    header += f" | Bloc: {bloc}"
                if bdef:
                    header += f" | Definició bloc: {bdef}"
                parts.append(header)
                parts.append(f"  Text: {txt}".strip())

        out = "\n".join([p for p in parts if p.strip()]).strip()
        return out[: self.cfg.max_context_chars].strip()
