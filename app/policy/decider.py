from dataclasses import dataclass
from typing import Optional, List

from app.config import RAGConfig
from app.retrieval.models import RetrievedItem
from app.utils import is_stale, safe_str


@dataclass
class Decision:
    action: str  # "answer" | "clarify"
    reason: str
    source_kind: str  # "faq" | "article" | "none"
    caution_note: Optional[str] = None


class PolicyDecider:
    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg

    def decide(self, question: str, lang: str, faqs: List[RetrievedItem], arts: List[RetrievedItem]) -> Decision:
        override_thr = self.cfg.overwrite

        top_faq = faqs[0] if faqs else None
        top_art = arts[0] if arts else None

        # 1) Artículos override si son súper relevantes
        if top_art and top_art.score >= override_thr:
            return Decision(
                action="answer",
                reason="articles_override",
                source_kind="article",
                caution_note=self._caution_if_stale(top_art, lang),
            )

        # 2) FAQ si pasa umbral (rápido, sin juez si ya es muy claro)
        if top_faq and top_faq.score >= self.cfg.min_score_faq_to_answer:
            return Decision(
                action="answer",
                reason="faq_confident",
                source_kind="faq",
                caution_note=self._caution_if_stale(top_faq, lang),
            )

        # 3) Artículos si pasan umbral
        if top_art and top_art.score >= self.cfg.min_score_articles_to_answer:
            return Decision(
                action="answer",
                reason="articles_confident",
                source_kind="article",
                caution_note=self._caution_if_stale(top_art, lang),
            )

        # 4) NUEVO: si hay FAQs recuperadas, intenta el pipeline de juez
        # (El juez decidirá si hay evidencia; si no, chatbot degradará a clarify).
        if faqs:
            return Decision(
                action="answer",
                reason="faq_try_judge_low_score",
                source_kind="faq",
                caution_note=self._caution_if_stale(top_faq, lang) if top_faq else None,
            )

        # 5) Si no hay nada, clarify
        return Decision(
            action="clarify",
            reason="low_signal_no_candidates",
            source_kind="none",
            caution_note=None,
        )

    def _caution_if_stale(self, item: RetrievedItem, lang: str) -> Optional[str]:
        upd = safe_str(item.meta.get("updatedAt"))
        if upd and is_stale(upd, self.cfg.stale_days_warn):
            if lang == "ca":
                return "Això pot haver canviat; si tens dubtes, confirma-ho amb la versió actual."
            if lang == "en":
                return "This may have changed; if in doubt, please confirm with the latest version."
            return "Esto puede haber cambiado; si tienes dudas, confírmalo con la versión actual."
        return None
