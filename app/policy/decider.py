# app/policy/decider.py
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

        # Mantengo por compatibilidad, pero ya no condiciona la prioridad FAQ vs Wiki en el caso normal.
        has_pantalla = "pantalla" in safe_str(question).lower()

        # 1) Clarify si ambos muy bajos
        if faq_score <= 0.2 and art_score <= 0.2:
            return Decision(
                action="clarify",
                reason="both_below_min_score",
                source_kind="none",
                caution_note=None,
            )

        # 2) Override fuerte por artículo (si gana MUY claro, usamos wiki)
        if top_art and art_score >= 0.78:
            return Decision(
                action="answer",
                reason="articles_override_0_78",
                source_kind="article",
                caution_note=self._caution_if_stale(top_art, lang),
            )

        # 3) Override por FAQ (lo dejo; ahora el caso "normal" ya favorece FAQs igualmente)
        if top_faq and faq_score >= 0.7 and not has_pantalla:
            return Decision(
                action="answer",
                reason="faq_override_0_70_no_pantalla",
                source_kind="faq",
                caution_note=self._caution_if_stale(top_faq, lang),
            )

        # 4) Caso normal: priorizar FAQs si están "cerca" del artículo
        #    margin por defecto 0.05 (si no existe en cfg)
        margin = float(getattr(self.cfg, "faq_over_article_margin", 0.05) or 0.05)

        if top_faq and top_art:
            # Si el FAQ está a <= margin del artículo, gana FAQ
            if faq_score >= (art_score - margin):
                return Decision(
                    action="answer",
                    reason="faq_preferred_within_margin",
                    source_kind="faq",
                    caution_note=self._caution_if_stale(top_faq, lang),
                )
            # Si el artículo gana por más de margin, gana artículo
            return Decision(
                action="answer",
                reason="article_wins_over_margin",
                source_kind="article",
                caution_note=self._caution_if_stale(top_art, lang),
            )

        # 5) Si solo hay uno, usamos el que exista
        if top_faq:
            return Decision(
                action="answer",
                reason="faq_only",
                source_kind="faq",
                caution_note=self._caution_if_stale(top_faq, lang),
            )

        if top_art:
            return Decision(
                action="answer",
                reason="article_only",
                source_kind="article",
                caution_note=self._caution_if_stale(top_art, lang),
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
