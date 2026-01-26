from typing import Any, Dict, List, Optional

from app.config import RAGConfig
from app.retrieval.retriever import Retriever
from app.policy.decider import PolicyDecider
from app.context import ContextBuilder
from app.prompt import LLMResponder
from app.utils import detect_language
from app.clients.openai_client import get_openai_client
from app.clients.embedder import Embedder
from app.clients.pinecone_gateway import PineconeGateway


class ChatbotService:
    def __init__(
        self,
        cfg: RAGConfig,
        retriever: Retriever,
        decider: PolicyDecider,
        ctx_builder: ContextBuilder,
        llm: LLMResponder,
    ):
        self.cfg = cfg
        self.retriever = retriever
        self.decider = decider
        self.ctx_builder = ctx_builder
        self.llm = llm

    # compat: una sola pregunta (sin memoria)
    def handle(self, question: str) -> Dict[str, Any]:
        return self.handle_messages([{"role": "user", "content": question}])

    # conversación sin lógica de aclaramiento
    def handle_messages(
        self,
        messages: List[Dict[str, str]],
        state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        last_user = self._last_user_message(messages)
        lang = detect_language(last_user)

        res = self.retriever.retrieve(last_user)
        faqs = res["faqs"]
        arts = res["articles"]

        decision = self.decider.decide(last_user, lang, faqs, arts)
        context = self.ctx_builder.build(faqs, arts)

        answer = self.llm.run(last_user, lang, context, decision)

        return {
            "language": lang,
            "decision": {"action": decision.action, "reason": decision.reason},
            "answer": answer,
            "debug": {
                "top_faq_score": faqs[0].score if faqs else None,
                "top_article_score": arts[0].score if arts else None,
            },
        }

    def _last_user_message(self, messages: List[Dict[str, str]]) -> str:
        for m in reversed(messages or []):
            if m.get("role") == "user" and (m.get("content") or "").strip():
                return m["content"].strip()
        return ""

def build_app(cfg: RAGConfig) -> ChatbotService:
    client = get_openai_client()
    embedder = Embedder(client=client, model=cfg.openai_embed_model)
    pc = PineconeGateway(index_name=cfg.pinecone_index_name)

    retriever = Retriever(cfg, pc, embedder)
    decider = PolicyDecider(cfg)
    ctx_builder = ContextBuilder(cfg)
    llm = LLMResponder(cfg, client)

    return ChatbotService(cfg, retriever, decider, ctx_builder, llm)
