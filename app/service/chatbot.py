from typing import Any, Dict, List, Optional
import re

from app.config import RAGConfig
from app.retrieval.retriever import Retriever
from app.policy.decider import PolicyDecider
from app.context import ContextBuilder
from app.prompt import LLMResponder
from app.utils import detect_language, safe_str, parse_iso
from app.clients.openai_client import get_openai_client
from app.clients.embedder import Embedder
from app.clients.pinecone_gateway import PineconeGateway
from app.faq_judge import FAQJudge
from app.query_rewriter import QueryRewriter


def _clip(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n].rstrip() + "…"


def _ym(iso: str) -> str:
    dt = parse_iso((iso or "").strip())
    if not dt:
        return ""
    return f"{dt.year:04d}-{dt.month:02d}"


def _faq_theme_until_first_dot(question: str) -> str:
    q = (question or "").strip()
    return q.split(".", 1)[0].strip() if q else ""


def _faq_ref(meta: Dict[str, Any]) -> str:
    if meta.get("faq_id"):
        return meta["faq_id"]
    if meta.get("source_row") is not None:
        return f"row {meta['source_row']}"
    if meta.get("fingerprint"):
        return f"fp {meta['fingerprint'][:10]}"
    return ""


def _pick_source_line(lang: str, decision, faqs: list, arts: list) -> str:
    if decision.source_kind == "faq" and faqs:
        meta = faqs[0].meta or {}
        base = "Font: FAQ" if lang == "ca" else "Source: FAQ" if lang == "en" else "Fuente: FAQ"
        ref = _faq_ref(meta)
        theme = _faq_theme_until_first_dot(meta.get("question", ""))
        return f"{base} {ref} — {theme}".strip(" —")

    if decision.source_kind == "article" and arts:
        meta = arts[0].meta or {}
        titulo = safe_str(meta.get("titulo"))
        ym = _ym(meta.get("updatedAt"))
        base = (
            f"Font: Wiki “{titulo}”" if lang == "ca"
            else f"Source: Wiki “{titulo}”" if lang == "en"
            else f"Fuente: Wiki “{titulo}”"
        )
        return f"{base} — {ym}".strip(" —")

    return ""


_SOURCE_LINE_RE = re.compile(r"(?im)^\s*(Fuente|Font|Source)\s*:\s*.*\s*$")


def _enforce_source_line(answer: str, source_line: str) -> str:
    if not source_line:
        return answer.strip()
    clean = _SOURCE_LINE_RE.sub("", answer or "").strip()
    return f"{clean}\n\n{source_line}".strip()


def _extract_debug_matches(items: list, kind: str, limit: int) -> List[Dict[str, Any]]:
    out = []
    for it in (items or [])[:limit]:
        meta = it.meta or {}
        out.append({
            "kind": kind,
            "score": it.score,
            "faq_id": meta.get("faq_id"),
            "question": meta.get("question"),
            "text_preview": _clip(it.text, 260),
        })
    return out


class ChatbotService:
    def __init__(
        self,
        cfg: RAGConfig,
        retriever: Retriever,
        decider: PolicyDecider,
        ctx_builder: ContextBuilder,
        llm: LLMResponder,
        faq_judge: FAQJudge,
        query_rewriter: QueryRewriter,
    ):
        self.cfg = cfg
        self.retriever = retriever
        self.decider = decider
        self.ctx_builder = ctx_builder
        self.llm = llm
        self.faq_judge = faq_judge
        self.query_rewriter = query_rewriter

    def handle_messages(
        self,
        messages: List[Dict[str, str]],
        state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        state = dict(state or {})
        last_user = self._last_user_message(messages)
        lang = detect_language(last_user)

        if last_user and not state.get("root_question"):
            state["root_question"] = last_user

        if last_user and len(last_user.split()) > 4:
            state["topic_question"] = last_user

        last_bot = self._last_bot_message(messages)
        if last_bot:
            state["last_bot"] = last_bot

        state["last_user"] = last_user
        memory = self._build_memory_text(state)

        effective_question = last_user

        res = self.retriever.retrieve(effective_question)
        faqs, arts = res["faqs"], res["articles"]
        decision = self.decider.decide(effective_question, lang, faqs, arts)

        if decision.action == "clarify" and state.get("topic_question"):
            effective_question = self.query_rewriter.rewrite(
                last_user=last_user,
                topic_question=state.get("topic_question"),
                last_bot=state.get("last_bot"),
                lang=lang,
            )
            res = self.retriever.retrieve(effective_question)
            faqs, arts = res["faqs"], res["articles"]
            decision = self.decider.decide(effective_question, lang, faqs, arts)

        selected_faqs = []
        judge_debug = None

        if decision.source_kind == "faq" and faqs:
            candidates = [
                {
                    "id": it.meta.get("faq_id"),
                    "question": it.meta.get("question"),
                    "answer": it.text,
                }
                for it in faqs[: self.cfg.top_k_faq_candidates]
                if it.meta.get("faq_id")
            ]

            judge_res = self.faq_judge.select(last_user, candidates, lang)
            ids = set(judge_res.selected_ids)

            selected_faqs = [
                it for it in faqs
                if it.meta.get("faq_id") in ids
            ][: self.cfg.top_k_faq_final]

            judge_debug = {
                "selected_ids": judge_res.selected_ids,
                "confidence": judge_res.confidence,
            }

            if not selected_faqs:
                decision.action = "clarify"
                decision.source_kind = "none"

        faqs_for_ctx = selected_faqs if decision.source_kind == "faq" else []
        arts_for_ctx = arts if decision.source_kind == "article" else []

        rag_context = (
            self.ctx_builder.build_faqs(faqs_for_ctx)
            if decision.source_kind == "faq"
            else self.ctx_builder.build_articles(arts_for_ctx)
            if decision.source_kind == "article"
            else ""
        )

        source_line = _pick_source_line(lang, decision, faqs_for_ctx, arts_for_ctx)

        raw_answer = self.llm.run(
            last_user,
            lang,
            rag_context,
            decision,
            memory=memory,
            source_line=source_line,
        )

        answer = (
            _enforce_source_line(raw_answer, source_line)
            if rag_context and source_line
            else raw_answer.strip()
        )

        state["last_bot"] = answer

        return {
            "language": lang,
            "decision": {
                "action": decision.action,
                "reason": decision.reason,
                "source_kind": decision.source_kind,
            },
            "answer": answer,
            "state": state,
            "debug": {
                "effective_question": effective_question,
                "faq_judge": judge_debug,
                "retrieval": {
                    "faqs": _extract_debug_matches(faqs, "faq", self.cfg.top_k_faq_candidates),
                    "articles": _extract_debug_matches(arts, "article", self.cfg.top_k_articles_final),
                },
            },
        }

    def _build_memory_text(self, state: Dict[str, Any]) -> str:
        parts = ["MEMORIA (último hilo):"]
        if state.get("root_question"):
            parts.append(f"- Pregunta raíz: {state['root_question']}")
        if state.get("last_bot"):
            parts.append(f"- Última respuesta del bot: {state['last_bot']}")
        if state.get("last_user"):
            parts.append(f"- Último mensaje del usuario: {state['last_user']}")
        return "\n".join(parts)

    def _last_user_message(self, messages: List[Dict[str, str]]) -> str:
        for m in reversed(messages or []):
            if m.get("role") == "user" and m.get("content"):
                return m["content"].strip()
        return ""

    def _last_bot_message(self, messages: List[Dict[str, str]]) -> str:
        for m in reversed(messages or []):
            if m.get("role") in ("assistant", "bot") and m.get("content"):
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
    faq_judge = FAQJudge(client=client, model=cfg.judge_model)
    query_rewriter = QueryRewriter(client=client, model=cfg.judge_model)

    return ChatbotService(
        cfg,
        retriever,
        decider,
        ctx_builder,
        llm,
        faq_judge,
        query_rewriter,
    )
