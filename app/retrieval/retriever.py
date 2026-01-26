from typing import Dict, List, Any
from app.config import RAGConfig
from app.clients.embedder import Embedder
from app.clients.pinecone_gateway import PineconeGateway
from app.retrieval.models import RetrievedItem
from app.utils import safe_str

class Retriever:
    def __init__(self, cfg: RAGConfig, pc: PineconeGateway, embedder: Embedder):
        self.cfg = cfg
        self.pc = pc
        self.embedder = embedder

    def retrieve(self, question: str) -> Dict[str, List[RetrievedItem]]:
        emb = self.embedder.embed_one(question)
        if not emb:
            return {"faqs": [], "articles": []}

        faqs_raw = self.pc.query(
            namespace=self.cfg.namespace_faqs,
            embedding=emb,
            top_k=self.cfg.top_k_faqs,
            flt=None,
        )
        faqs = [self._to_item("faq", r) for r in faqs_raw]

        arts_raw = self.pc.query(
            namespace=self.cfg.namespace_articles,
            embedding=emb,
            top_k=self.cfg.top_k_articles,
            flt={"verificado": True},
        )
        arts = [self._to_item("article", r) for r in arts_raw]

        return {"faqs": sorted(faqs, key=lambda x: x.score, reverse=True),
                "articles": sorted(arts, key=lambda x: x.score, reverse=True)}

    def _to_item(self, source: str, raw: Dict[str, Any]) -> RetrievedItem:
        meta = raw.get("metadata") or {}
        text = safe_str(meta.get("text"))
        return RetrievedItem(source=source, score=float(raw.get("score") or 0.0), text=text, meta=meta)
