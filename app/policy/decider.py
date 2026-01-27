from dataclasses import dataclass
from typing import Optional, List

from app.config import RAGConfig
from app.retrieval.models import RetrievedItem
from app.utils import is_stale, safe_str


@dataclass
class Decision:
    action: str  # "answer" | "clarify"
    reason: str
    caution_note: Optional[str] = None
    citation_hint: Optional[str] = None


class PolicyDecider:
    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg

    def decide(self, question: str, lang: str, faqs: List[RetrievedItem], arts: List[RetrievedItem]) -> Decision:

        # Override: si un artículo es extremadamente relevante, priorízalo siempre
        # (idealmente cfg.min_score_articles_override = 0.92 o 0.95)
        override_thr = self.cfg.overwrite
        if arts and arts[0].score >= override_thr:
            return Decision(
                action="answer",
                reason="articles_override",
                caution_note=self._caution_if_stale(arts[0], lang),
                citation_hint=self._cite_brief(arts[0], lang),
            )

        # Flujo normal: FAQ primero si supera umbral
        if faqs and faqs[0].score >= self.cfg.min_score_faq_to_answer:
            return Decision(
                action="answer",
                reason="faq_confident",
                caution_note=self._caution_if_stale(faqs[0], lang),
                citation_hint=self._cite_brief(faqs[0], lang),
            )

        # Fallback: artículos si superan su umbral
        if arts and arts[0].score >= self.cfg.min_score_articles_to_answer:
            return Decision(
                action="answer",
                reason="articles_confident",
                caution_note=self._caution_if_stale(arts[0], lang),
                citation_hint=self._cite_brief(arts[0], lang),
            )

        return Decision(action="answer", reason="low_signal")

    def _caution_if_stale(self, item: RetrievedItem, lang: str) -> Optional[str]:
        upd = safe_str(item.meta.get("updatedAt"))
        if upd and is_stale(upd, self.cfg.stale_days_warn):
            return (
                "Això pot haver canviat; si tens dubtes, confirma-ho amb la versió actual."
                if lang == "ca"
                else "Esto puede haber cambiado; si tienes dudas, confírmalo con la versión actual."
            )
        return None

    def _cite_brief(self, item: RetrievedItem, lang: str) -> str:
        if item.source == "faq":
            title = safe_str(
                item.meta.get("titulo")
                or item.meta.get("title")
                or item.meta.get("faq_id")
                or item.meta.get("id")
                or ""
            )
            return (f"Font: FAQ {title}" if lang == "ca" else f"Fuente: FAQ {title}").strip()

        titulo = safe_str(item.meta.get("titulo") or item.meta.get("article_id") or item.meta.get("id") or "")
        bloc = safe_str(item.meta.get("block_title") or "")
        if lang == "ca":
            return (f"Font: Wiki “{titulo}” — Bloc “{bloc}”" if bloc else f"Font: Wiki “{titulo}”")
        return (f"Fuente: Wiki “{titulo}” — Bloque “{bloc}”" if bloc else f"Fuente: Wiki “{titulo}”")
