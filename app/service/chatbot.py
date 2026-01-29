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
    fid = safe_str(meta.get("faq_id")).strip()
    if fid:
        return fid
    row = meta.get("source_row")
    if row is not None and str(row).strip():
        return f"row {row}"
    fp = safe_str(meta.get("fingerprint")).strip()
    if fp:
        return f"fp {fp[:10]}"
    return ""


def _pick_source_line(lang: str, decision: Decision, faqs: list, arts: list) -> str:
    if decision.source_kind == "faq" and faqs:
        meta = faqs[0].meta or {}
        base = "Font: FAQ" if lang == "ca" else "Source: FAQ" if lang == "en" else "Fuente: FAQ"
        ref = _faq_ref(meta)
        theme = _faq_theme_until_first_dot(safe_str(meta.get("question")))
        out = base
        if ref:
            out += f" {ref}"
        if theme:
            out += f" — {theme}"
        return out.strip()

    if decision.source_kind == "article" and arts:
        meta = arts[0].meta or {}
        titulo = safe_str(meta.get("titulo")).strip()
        ym = _ym(safe_str(meta.get("updatedAt")))
        if lang == "ca":
            base = f'Font: Wiki “{titulo}”' if titulo else "Font: Wiki"
            if ym:
                base += f" — actualitzat {ym}"
            return base.strip()
        if lang == "en":
            base = f'Source: Wiki “{titulo}”' if titulo else "Source: Wiki"
            if ym:
                base += f" — updated {ym}"
            return base.strip()
        base = f'Fuente: Wiki “{titulo}”' if titulo else "Fuente: Wiki"
        if ym:
            base += f" — actualizado {ym}"
        return base.strip()

    return ""


_SOURCE_LINE_RE = re.compile(r"(?im)^\s*(Fuente|Font|Source)\s*:\s*.*\s*$")


def _enforce_source_line(answer: str, source_line: str) -> str:
    ans = (answer or "").strip()
    src = (source_line or "").strip()
    if not src:
        return ans
    ans = _SOURCE_LINE_RE.sub("", ans).strip()
    return f"{ans}\n\n{src}".strip() if ans else src


def _extract_debug_faqs(items: list, kind: str, limit: int) -> List[Dict[str, Any]]:
    out = []
    for it in (items or [])[:limit]:
        meta = it.meta or {}
        out.append({
            "kind": kind,
            "score": float(it.score or 0.0),
            "question": meta.get("question"),
            "text_preview": _clip(it.text, 260),
        })
    return out


def _extract_debug_articles(items: list, kind: str, limit: int) -> List[Dict[str, Any]]:
    out = []
    for it in (items or [])[:limit]:
        meta = it.meta or {}
        out.append({
            "kind": kind,
            "score": float(it.score or 0.0),
            "titulo": meta.get("titulo"),
            "chunk_index": meta.get("chunk_index"),
            "block_id": meta.get("block_id"),
            "text_preview": _clip(it.text, 260),
        })
    return out


# --------- NUEVO: helpers de tema/pantalla ---------

_SCREEN_RE = re.compile(r"(?:pantalla|screen)\s+([^\n\?]+)", re.IGNORECASE)

def _extract_screen_name(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""

    # Caso explícito "pantalla X"
    m = _SCREEN_RE.search(t)
    if m:
        return m.group(1).strip(" \"'”")

    # Heurística mínima (NO inventar): si el texto menciona claramente el nombre
    tl = t.lower()
    if "gestió" in tl and "lliur" in tl:
        return "Gestió de lliuraments"

    return ""


def _should_update_topic(q: str) -> bool:
    ql = (q or "").lower()
    # Solo si hay señal de "tema real", no por longitud
    return any(k in ql for k in ["pantalla", "mòdul", "módulo", "gestió", "gestio", "lliur", "provisionals"])


# --------------------------------------------------


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

    def handle_messages(self, messages: List[Dict[str, str]], state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        state = dict(state or {})

        last_user = self._last_user_message(messages)
        lang = detect_language(last_user)

        # root_question solo una vez
        if last_user and not state.get("root_question"):
            state["root_question"] = last_user

        # screen_hint persistente si lo detectamos
        detected_screen = _extract_screen_name(last_user) or _extract_screen_name(state.get("last_bot", ""))
        if detected_screen:
            state["screen_hint"] = detected_screen

        # topic_question: NO se pisa con “ahí/esto”
        if last_user and _should_update_topic(last_user):
            state["topic_question"] = last_user

        last_bot = self._last_bot_message(messages)
        if last_bot:
            state["last_bot"] = last_bot

        state["last_user"] = last_user
        memory = self._build_memory_text(state)

        # --------- NUEVO: rewriter siempre salvo primer turno ---------
        is_first_turn = not bool(state.get("last_bot"))  # si no hay last_bot, asumimos 1er turno real
        effective_question = last_user
        rewrite_used = False

        if not is_first_turn:
            effective_question = self.query_rewriter.rewrite(
                last_user=last_user,
                topic_question=state.get("topic_question") or "",
                last_bot=state.get("last_bot") or "",
                lang=lang,
                screen_hint=state.get("screen_hint") or "",
                root_question=state.get("root_question") or "",
            )
            rewrite_used = True
        # ------------------------------------------------------------

        # 1) retrieval usando effective_question (reescrita)
        res = self.retriever.retrieve(effective_question, expand_articles=False)
        faqs, arts = res["faqs"], res["articles"]
        arts_global = list(arts)  # <-- NUEVO snapshot
        decision = self.decider.decide(effective_question, lang, faqs, arts)

        # 2) judge FAQs (solo si el decider eligió FAQ)
        selected_faqs: List[Any] = []
        judge_debug: Optional[Dict[str, Any]] = None

        if decision.source_kind == "faq" and faqs:
            candidates = []
            for it in faqs[: self.cfg.top_k_faq_candidates]:
                fid = safe_str((it.meta or {}).get("faq_id")).strip()
                if not fid:
                    continue
                candidates.append({
                    "id": fid,
                    "question": safe_str((it.meta or {}).get("question")),
                    "answer": safe_str(it.text),
                })

            # ✅ CLAVE: el judge debe ver la pregunta efectiva (con pantalla)
            judge_res = self.faq_judge.select(effective_question, candidates, lang)
            ids = set(judge_res.selected_ids)

            selected_faqs = [
                it for it in faqs
                if safe_str((it.meta or {}).get("faq_id")).strip() in ids
            ][: self.cfg.top_k_faq_final]

            judge_debug = {
                "selected_ids": judge_res.selected_ids,
                "confidence": judge_res.confidence,
                "reason": judge_res.reason,
                "candidates_count": len(candidates),
                "selected_count": len(selected_faqs),
            }

            # Si el juez no ve evidencia, usamos top FAQs sin cambiar la decision
            if not selected_faqs:
                selected_faqs = faqs[: self.cfg.top_k_faq_final]
                if judge_debug is not None:
                    judge_debug["fallback_used"] = True

        arts_expanded = []
        if decision.source_kind == "article":
            arts_expanded = self.retriever.expand_articles(effective_question, arts)
            arts = arts_expanded

        # context según decisión final
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
            last_user,  # OJO: aquí mantengo last_user para responder al usuario como tal
            lang,
            rag_context,
            decision,
            memory=memory,
            source_line=source_line,
        )

        answer = _enforce_source_line(raw_answer, source_line) if (rag_context and source_line) else (raw_answer or "").strip()
        state["last_bot"] = answer

        return {
            "language": lang,
            "decision": {"action": decision.action, "reason": decision.reason, "source_kind": decision.source_kind},
            "answer": answer,
            "state": state,
            "debug": {
                "original_last_user": last_user,
                "effective_question": effective_question,
                "rewrite_used": rewrite_used,
                "screen_hint": state.get("screen_hint"),
                "topic_question": state.get("topic_question"),
                "faq_judge": judge_debug,
                "faqs": {
                    "candidates": _extract_debug_faqs(faqs, "faq", self.cfg.top_k_faq_candidates),
                    "selected": _extract_debug_faqs(faqs_for_ctx, "faq_selected", self.cfg.top_k_faq_final),
                },
                "articles": {
                "global_candidates": _extract_debug_articles(arts_global, "article_global", self.cfg.top_k_articles_candidates),
                "expanded_pool": _extract_debug_articles(arts_expanded, "article_expanded", self.cfg.top_k_articles_final),
                "cited_chunks": _extract_debug_articles(arts_for_ctx, "article_cited", self.cfg.top_k_articles_final),
},

            },
        }

    def _build_memory_text(self, state: Dict[str, Any]) -> str:
        parts = ["MEMORIA (último hilo):"]
        if state.get("root_question"):
            parts.append(f"- Pregunta raíz: {_clip(state['root_question'], 280)}")
        if state.get("screen_hint"):
            parts.append(f"- Pantalla/tema: {_clip(state['screen_hint'], 140)}")
        if state.get("last_bot"):
            parts.append(f"- Última respuesta del bot: {_clip(state['last_bot'], 420)}")
        if state.get("last_user"):
            parts.append(f"- Último mensaje del usuario: {_clip(state['last_user'], 280)}")
        return "\n".join(parts).strip()

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

    return ChatbotService(cfg, retriever, decider, ctx_builder, llm, faq_judge, query_rewriter)
