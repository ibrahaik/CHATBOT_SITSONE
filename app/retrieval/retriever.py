# app/retrieval/retriever.py  (TE LO DEJO IGUAL, no tocamos lógica aquí)
from typing import Dict, List, Any, Optional
import re

from app.config import RAGConfig
from app.clients.embedder import Embedder
from app.clients.pinecone_gateway import PineconeGateway
from app.retrieval.models import RetrievedItem
from app.utils import safe_str

_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ0-9_]+", re.UNICODE)

def _to_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None:
            return default
        return int(str(x).strip())
    except Exception:
        return default

class Retriever:
    def __init__(self, cfg: RAGConfig, pc: PineconeGateway, embedder: Embedder):
        self.cfg = cfg
        self.pc = pc
        self.embedder = embedder

    def retrieve(self, question: str, expand_articles: bool = True) -> Dict[str, List[RetrievedItem]]:
        emb = self.embedder.embed_one(question)
        if not emb:
            return {"faqs": [], "articles": []}

        # ---- FAQs ----
        faqs_raw = self.pc.query(
            namespace=self.cfg.namespace_faqs,
            embedding=emb,
            top_k=self.cfg.top_k_faq_candidates,
            flt=None,
        )
        faqs = [self._to_item("faq", r) for r in faqs_raw]
        faqs = sorted(faqs, key=lambda x: x.score, reverse=True)

        # ---- Artículos: 1) global ----
        arts_raw = self.pc.query(
            namespace=self.cfg.namespace_articles,
            embedding=emb,
            top_k=self.cfg.top_k_articles_candidates,
            flt=None,
        )
        arts = [self._to_item("article", r) for r in arts_raw]
        arts = sorted(arts, key=lambda x: x.score, reverse=True)

        # ---- Artículos: 2) expansión (mismo block_id del ganador) ----
        if expand_articles:
            arts = self._expand_articles_same_block(embedding=emb, initial=arts)

        return {"faqs": faqs, "articles": arts}

    def expand_articles(self, question: str, initial: List[RetrievedItem]) -> List[RetrievedItem]:
        emb = self.embedder.embed_one(question)
        if not emb:
            return initial or []
        return self._expand_articles_same_block(embedding=emb, initial=initial or [])

    def _to_item(self, source: str, raw: Dict[str, Any]) -> RetrievedItem:
        meta = raw.get("metadata") or {}
        if source == "faq":
            text = safe_str(meta.get("answer") or meta.get("text"))
        else:
            text = safe_str(meta.get("text"))

        return RetrievedItem(
            source=source,
            score=float(raw.get("score") or 0.0),
            text=text,
            meta=meta,
        )

    def _expand_articles_same_block(self, embedding: List[float], initial: List[RetrievedItem]) -> List[RetrievedItem]:
        if not initial:
            return []

        anchor = initial[0]
        meta = anchor.meta or {}

        anchor_block_id = safe_str(meta.get("block_id")).strip()
        anchor_article_id = safe_str(meta.get("article_id")).strip()

        if not anchor_block_id:
            return initial[: self.cfg.top_k_articles_final]

        pool_k = getattr(self.cfg, "top_k_articles_within_article", 60) or 60
        pool_k = max(pool_k, 60)

        flt: Dict[str, Any] = {"block_id": {"$eq": anchor_block_id}}

        raw = self.pc.query(
            namespace=self.cfg.namespace_articles,
            embedding=embedding,
            top_k=pool_k,
            flt=flt,
        )
        items = [self._to_item("article", r) for r in raw]

        by_key: Dict[str, RetrievedItem] = {}
        for it in items:
            by_key[self._article_key(it)] = it
        by_key[self._article_key(anchor)] = anchor

        items = list(by_key.values())
        items = self._sort_article_chunks(items)

        hard_cap = max(int(self.cfg.top_k_articles_block_context or 15), 15)
        hard_cap = max(hard_cap, 15)
        return items[:hard_cap]

    def _sort_article_chunks(self, items: List[RetrievedItem]) -> List[RetrievedItem]:
        def key(it: RetrievedItem):
            m = it.meta or {}
            aid = safe_str(m.get("article_id"))
            idx = _to_int(m.get("chunk_index"), default=10**9)
            return (aid, idx)
        return sorted(items, key=key)

    def _article_key(self, it: RetrievedItem) -> str:
        m = it.meta or {}
        aid = safe_str(m.get("article_id")).strip()
        idx = safe_str(m.get("chunk_index")).strip()
        bid = safe_str(m.get("block_id")).strip()
        role = safe_str(m.get("chunk_role")).strip()
        return f"{aid}::{idx}::{bid}::{role}"
