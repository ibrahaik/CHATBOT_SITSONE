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


def _clip(s: str, n: int) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[:n].rstrip() + "…"


def _ym(iso: str) -> str:
    dt = parse_iso((iso or "").strip())
    if not dt:
        return ""
    return f"{dt.year:04d}-{dt.month:02d}"


def _faq_theme_until_first_dot(question: str) -> str:
    q = (question or "").strip()
    if not q:
        return ""
    return q.split(".", 1)[0].strip()


def _faq_ref(meta: Dict[str, Any]) -> str:
    """
    Referencia FAQ robusta (evita el caso "FAQ1" siempre):
    - 1) faq_id
    - 2) source_row (muy útil y único)
    - 3) fingerprint corto
    """
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


def _pick_source_line(lang: str, decision, faqs: list, arts: list) -> str:
    """
    Fuente calculada en backend y dependiente de Decision.
    Regla:
      - faq_confident -> usa FAQ top1
      - articles_confident -> usa Artículo top1
      - low_signal -> "" (sin fuente)
    """
    reason = safe_str(getattr(decision, "reason", "")).lower()
    use_faq = reason.startswith("faq")
    use_art = reason.startswith("articles")

    if use_faq and faqs:
        meta = faqs[0].meta or {}
        ref = _faq_ref(meta)
        theme = _faq_theme_until_first_dot(safe_str(meta.get("question")))
        if lang == "ca":
            base = "Font: FAQ"
            if ref:
                base += f" {ref}"
            if theme:
                base += f" — {theme}"
            return base.strip()
        base = "Fuente: FAQ"
        if ref:
            base += f" {ref}"
        if theme:
            base += f" — {theme}"
        return base.strip()

    if use_art and arts:
        meta = arts[0].meta or {}
        titulo = safe_str(meta.get("titulo")).strip()
        ym = _ym(safe_str(meta.get("updatedAt")))
        if lang == "ca":
            base = f"Font: Wiki “{titulo}”" if titulo else "Font: Wiki"
            if ym:
                base += f" — actualitzat {ym}"
            return base.strip()
        base = f"Fuente: Wiki “{titulo}”" if titulo else "Fuente: Wiki"
        if ym:
            base += f" — actualizado {ym}"
        return base.strip()

    return ""


_SOURCE_LINE_RE = re.compile(r"(?im)^\s*(Fuente|Font)\s*:\s*.*\s*$")


def _enforce_source_line(answer: str, source_line: str) -> str:
    """
    Si hay fuente backend:
      - elimina cualquier línea Fuente:/Font: del LLM
      - añade la fuente backend al final
    """
    ans = (answer or "").strip()
    src = (source_line or "").strip()
    if not src:
        return ans

    ans = _SOURCE_LINE_RE.sub("", ans).strip()
    if not ans:
        return src
    return f"{ans}\n\n{src}".strip()


def _extract_debug_matches(items: list, kind: str, limit: int = 5) -> List[Dict[str, Any]]:
    out = []
    for it in (items or [])[:limit]:
        meta = it.meta or {}
        out.append({
            "kind": kind,
            "score": float(it.score or 0.0),
            "faq_id": meta.get("faq_id"),
            "source_row": meta.get("source_row"),
            "fingerprint": meta.get("fingerprint"),
            "titulo": meta.get("titulo") or meta.get("title"),
            "question": meta.get("question") or meta.get("q"),
            "block_title": meta.get("block_title"),
            "block_definition": meta.get("block_definition"),
            "updatedAt": meta.get("updatedAt"),
            "verificado": meta.get("verificado") if "verificado" in meta else None,
            "text_preview": _clip(safe_str(meta.get("text") or it.text), 260),
            "metadata": meta,
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
    ):
        self.cfg = cfg
        self.retriever = retriever
        self.decider = decider
        self.ctx_builder = ctx_builder
        self.llm = llm

    def handle(self, question: str) -> Dict[str, Any]:
        return self.handle_messages([{"role": "user", "content": question}], state=None)

    def handle_messages(
        self,
        messages: List[Dict[str, str]],
        state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        last_user = self._last_user_message(messages)
        lang = detect_language(last_user)

        # --- state barato ---
        state = dict(state or {})
        if not state.get("root_question") and last_user:
            state["root_question"] = last_user
        last_bot_from_messages = self._last_bot_message(messages)
        if last_bot_from_messages:
            state["last_bot"] = last_bot_from_messages
        state["last_user"] = last_user
        memory = self._build_memory_text(state)

        # --- retrieval ---
        res = self.retriever.retrieve(last_user)
        faqs = res["faqs"]
        arts = res["articles"]

        decision = self.decider.decide(last_user, lang, faqs, arts)

        rag_context = self.ctx_builder.build(faqs, arts)
        context = (memory + "\n\n" + rag_context).strip()

        # Fuente backend (depende del decision)
        source_line = _pick_source_line(lang, decision, faqs, arts)

        # LLM
        raw_answer = self.llm.run(
            last_user, lang, context, decision, memory=memory, source_line=source_line
        )

        # Forzar fuente SOLO si realmente hay contexto (evita "Fuente:" en low_signal/out-of-domain)
        has_context = bool((rag_context or "").strip())
        answer = _enforce_source_line(raw_answer, source_line) if (has_context and source_line) else raw_answer.strip()

        # actualizar state
        state["last_bot"] = answer

        # Debug extendido
        picked_kind = "faq" if safe_str(decision.reason).lower().startswith("faq") else ("article" if safe_str(decision.reason).lower().startswith("articles") else "none")
        picked_meta = (faqs[0].meta if (picked_kind == "faq" and faqs) else (arts[0].meta if (picked_kind == "article" and arts) else {})) or {}

        debug = {
            "decision": {"action": decision.action, "reason": decision.reason},
            "top_faq_score": faqs[0].score if faqs else None,
            "top_article_score": arts[0].score if arts else None,
            "picked_source_kind": picked_kind,
            "picked_source_line": source_line,
            "picked_source_meta": picked_meta,
            "used_context": has_context,
            "memory_preview": _clip(memory, 600),
            "retrieval": {
                "faqs": _extract_debug_matches(faqs, "faq", limit=self.cfg.top_k_faqs),
                "articles": _extract_debug_matches(arts, "article", limit=self.cfg.top_k_articles),
            },
        }

        return {
            "language": lang,
            "decision": {"action": decision.action, "reason": decision.reason},
            "answer": answer,
            "state": state,
            "debug": debug,
        }

    def _build_memory_text(self, state: Dict[str, Any]) -> str:
        root_q = _clip(state.get("root_question", ""), 280)
        last_bot = _clip(state.get("last_bot", ""), 420)
        last_user = _clip(state.get("last_user", ""), 280)

        parts = ["MEMORIA (último hilo):"]
        if root_q:
            parts.append(f"- Pregunta raíz: {root_q}")
        if last_bot:
            parts.append(f"- Última respuesta del bot: {last_bot}")
        if last_user:
            parts.append(f"- Último mensaje del usuario: {last_user}")

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

    return ChatbotService(cfg, retriever, decider, ctx_builder, llm)
