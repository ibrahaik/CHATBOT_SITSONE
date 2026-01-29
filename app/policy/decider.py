from dataclasses import dataclass
from typing import Optional, List

from app.config import RAGConfig
from app.retrieval.models import RetrievedItem
from app.utils import is_stale, safe_str


@dataclass
class Decision:
    action: str
    reason: str
    source_kind: str  # "faq" | "article" | "none"
    caution_note: Optional[str] = None


class PolicyDecider:
    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg

    def decide(self, question: str, lang: str, faqs: List[RetrievedItem], arts: List[RetrievedItem]) -> Decision:
        top_faq = faqs[0] if faqs else None
        top_art = arts[0] if arts else None

        faq_score = float(top_faq.score) if top_faq else 0.0
        art_score = float(top_art.score) if top_art else 0.0

        has_pantalla = "pantalla" in safe_str(question).lower()

        # 1) Clarify si ambos muy bajos
        if faq_score <= 0.2 and art_score <= 0.2:
            return Decision(
                action="clarify",
                reason="both_below_min_score",
                source_kind="none",
                caution_note=None,
            )

        # 2) Override por articulo
        if top_art and art_score >= 0.78:
            return Decision(
                action="answer",
                reason="articles_override_0_78",
                source_kind="article",
                caution_note=self._caution_if_stale(top_art, lang),
            )

        # 3) Override por FAQ (solo si no aparece "pantalla")
        if top_faq and faq_score >= 0.7 and not has_pantalla:
            return Decision(
                action="answer",
                reason="faq_override_0_70_no_pantalla",
                source_kind="faq",
                caution_note=self._caution_if_stale(top_faq, lang),
            )

        # 4) Si ninguno supera override, gana el que tenga mas score
        if top_art and (art_score >= faq_score or not top_faq):
            return Decision(
                action="answer",
                reason="articles_best_score",
                source_kind="article",
                caution_note=self._caution_if_stale(top_art, lang),
            )
        if top_faq:
            return Decision(
                action="answer",
                reason="faq_best_score",
                source_kind="faq",
                caution_note=self._caution_if_stale(top_faq, lang),
            )

        return Decision(
            action="clarify",
            reason="low_signal_no_candidates",
            source_kind="none",
            caution_note=None,
        )

    def _caution_if_stale(self, item: RetrievedItem, lang: str) -> Optional[str]:
        upd = safe_str((item.meta or {}).get("updatedAt"))
        if upd and is_stale(upd, self.cfg.stale_days_warn):
            if lang == "ca":
                return "Això pot haver canviat; si tens dubtes, confirma-ho amb la versió actual."
            if lang == "en":
                return "This may have changed; if in doubt, please confirm with the latest version."
            return "Esto puede haber cambiado; si tienes dudas, confírmalo con la versión actual."
        return None
