# app/chatbot.py
from typing import Any, Dict, List, Optional
import re

from app.config import RAGConfig
from app.retrieval.retriever import Retriever
from app.policy.decider import PolicyDecider, Decision
from app.context import ContextBuilder
from app.prompt import LLMResponder
from app.utils import detect_language, safe_str, parse_iso
from app.clients.openai_client import get_openai_client
from app.clients.embedder import Embedder
from app.clients.pinecone_gateway import PineconeGateway
from app.faq_judge import FAQJudge
from app.query_rewriter import QueryRewriter


# ---------------- utils ----------------

def _clip(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n].rstrip() + "…"


def _ym(iso: str) -> str:
    dt = parse_iso((iso or "").strip())
    if not dt:
        return ""
    return f"{dt.year:04d}-{dt.month:02d}"


def _faq_theme_until_second_dot(question: str) -> str:
    q = (question or "").strip()
    if not q:
        return ""

    parts = q.split(".")
    if len(parts) >= 3:
        a = parts[0].strip()
        b = parts[1].strip()
        if a and b:
            return f"{a}. {b}."

    if len(parts) >= 2 and parts[0].strip():
        return f"{parts[0].strip()}."

    return q


def _pick_source_line(lang: str, decision: Decision, faqs: list, arts: list) -> str:
    if decision.source_kind == "faq" and faqs:
        meta = faqs[0].meta or {}
        base = "Font: FAQ" if lang == "ca" else "Source: FAQ" if lang == "en" else "Fuente: FAQ"
        theme = _faq_theme_until_second_dot(safe_str(meta.get("question")))
        return f"{base} — {theme}".strip() if theme else base

    if decision.source_kind == "article" and arts:
        meta = arts[0].meta or {}
        titulo = safe_str(meta.get("titulo")).strip()
        ym = _ym(safe_str(meta.get("updatedAt")))

        if lang == "ca":
            base = f'Font: Wiki “{titulo}”' if titulo else "Font: Wiki"
            return f"{base} — actualitzat {ym}".strip() if ym else base
        if lang == "en":
            base = f'Source: Wiki “{titulo}”' if titulo else "Source: Wiki"
            return f"{base} — updated {ym}".strip() if ym else base

        base = f'Fuente: Wiki “{titulo}”' if titulo else "Fuente: Wiki"
        return f"{base} — actualizado {ym}".strip() if ym else base

    return ""


_SOURCE_LINE_RE = re.compile(r"(?im)^\s*(Fuente|Font|Source)\s*:\s*.*\s*$")


def _enforce_source_line(answer: str, source_line: str) -> str:
    ans = (answer or "").strip()
    if not source_line:
        return ans
    ans = _SOURCE_LINE_RE.sub("", ans).strip()
    return f"{ans}\n\n{source_line}".strip()


def _extract_debug(items: list, kind: str, limit: int) -> List[Dict[str, Any]]:
    out = []
    for it in (items or [])[:limit]:
        meta = it.meta or {}
        out.append(
            {
                "kind": kind,
                "score": float(it.score or 0.0),
                "meta": meta,
                "text_preview": _clip(it.text, 260),
            }
        )
    return out


# ---------------- service ----------------

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
        last_bot = self._last_bot_message(messages)
        lang = detect_language(last_user)

        # -------- rewriter manda --------
        effective_question = last_user
        rewrite_used = False

        if last_bot:
            effective_question = self.query_rewriter.rewrite(
                last_user=last_user,
                last_bot=last_bot,
            )
            rewrite_used = True

        # -------- retrieval --------
        res = self.retriever.retrieve(effective_question, expand_articles=False)
        faqs, arts = res["faqs"], res["articles"]
        arts_global = list(arts)

        decision = self.decider.decide(effective_question, lang, faqs, arts)

        # -------- FAQ judge --------
        selected_faqs: List[Any] = []
        judge_debug: Optional[Dict[str, Any]] = None

        if decision.source_kind == "faq" and faqs:
            candidates = []
            for it in faqs[: self.cfg.top_k_faq_candidates]:
                fid = safe_str((it.meta or {}).get("faq_id")).strip()
                if not fid:
                    continue
                candidates.append(
                    {
                        "id": fid,
                        "question": safe_str((it.meta or {}).get("question")),
                        "answer": safe_str(it.text),
                    }
                )

            judge_res = self.faq_judge.select(effective_question, candidates, lang)
            ids = set(judge_res.selected_ids)

            selected_faqs = [
                it for it in faqs
                if safe_str((it.meta or {}).get("faq_id")).strip() in ids
            ][: self.cfg.top_k_faq_final]

            if not selected_faqs:
                selected_faqs = faqs[: self.cfg.top_k_faq_final]

            judge_debug = {
                "selected_ids": judge_res.selected_ids,
                "confidence": judge_res.confidence,
                "reason": judge_res.reason,
            }

        # -------- expand articles --------
        arts_expanded: List[Any] = []
        if decision.source_kind == "article":
            arts_expanded = self.retriever.expand_articles(effective_question, arts)
            arts = arts_expanded

        # -------- context --------
        if decision.source_kind == "faq":
            rag_context = self.ctx_builder.build_faqs(selected_faqs)
            faqs_for_ctx, arts_for_ctx = selected_faqs, []
        elif decision.source_kind == "article":
            rag_context = self.ctx_builder.build_articles(arts)
            faqs_for_ctx, arts_for_ctx = [], arts
        else:
            rag_context = ""
            faqs_for_ctx, arts_for_ctx = [], []

        source_line = _pick_source_line(lang, decision, faqs_for_ctx, arts_for_ctx)

        raw_answer = self.llm.run(
            last_user,
            lang,
            rag_context,
            decision,
            memory="",
            source_line=source_line,
        )

        answer = (
            _enforce_source_line(raw_answer, source_line)
            if rag_context and source_line
            else (raw_answer or "").strip()
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
                "original_last_user": last_user,
                "effective_question": effective_question,
                "rewrite_used": rewrite_used,
                "faq_judge": judge_debug,
                "faqs": _extract_debug(faqs_for_ctx, "faq", self.cfg.top_k_faq_final),
                "articles": {
                    "global": _extract_debug(arts_global, "article_global", self.cfg.top_k_articles_candidates),
                    "used": _extract_debug(arts_for_ctx, "article_used", self.cfg.top_k_articles_block_context),
                },
            },
        }

    def _last_user_message(self, messages: List[Dict[str, str]]) -> str:
        for m in reversed(messages or []):
            if m.get("role") == "user" and (m.get("content") or "").strip():
                return m["content"].strip()
        return ""

    def _last_bot_message(self, messages: List[Dict[str, str]]) -> str:
        for m in reversed(messages or []):
            if m.get("role") in ("assistant", "bot") and (m.get("content") or "").strip():
                return m["content"].strip()
        return ""


# ---------------- factory ----------------

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
