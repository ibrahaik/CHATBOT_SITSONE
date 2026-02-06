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


def _faq_theme_until_second_dot(question: str) -> str:
    """
    Extrae el "tema" del FAQ hasta el segundo punto.
    Ej:
      "FAQs T-metropolitana. CANVI DE TARGETA ROSA A T-METROPOLITANA. 1. ..." =>
      "FAQs T-metropolitana. CANVI DE TARGETA ROSA A T-METROPOLITANA."
    """
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
        out = base
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
        out.append(
            {
                "kind": kind,
                "score": float(it.score or 0.0),
                "question": meta.get("question"),
                "text_preview": _clip(it.text, 260),
            }
        )
    return out


def _extract_debug_articles(items: list, kind: str, limit: int) -> List[Dict[str, Any]]:
    out = []
    for it in (items or [])[:limit]:
        meta = it.meta or {}
        raw = it.text or ""
        out.append(
            {
                "kind": kind,
                "score": float(it.score or 0.0),
                "titulo": meta.get("titulo"),
                "chunk_index": meta.get("chunk_index"),
                "block_id": meta.get("block_id"),
                "text_preview": _clip(raw, 260),  # solo RAW
            }
        )
    return out


# --------- helpers de tema/pantalla ---------
_SCREEN_RE = re.compile(r"(?im)\b(?:pantalla|screen)\b\s*[:\-]?\s*([^\n\?]+)")


def _extract_screen_name(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    m = _SCREEN_RE.search(t)
    if m:
        name = m.group(1).strip().strip(" \"'”")
        if 2 <= len(name) <= 80:
            return name
    return ""


def _should_update_topic(q: str) -> bool:
    ql = (q or "").lower()
    return any(k in ql for k in ["pantalla", "mòdul", "módulo", "gestió", "gestio", "lliur", "provisionals"])


# --------- NUEVO: detección de follow-up (sin IA) ---------
_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ0-9_]+", re.UNICODE)

_DEICTIC_TOKENS = {
    # ES
    "esto", "eso", "esa", "ese", "aqui", "aquí", "ahi", "ahí", "alli", "allí",
    "asi", "así", "entonces", "vale", "ok", "ojo",
    "lo", "la", "los", "las", "ello",
    "tambien", "también",
    # CA
    "aixo", "això", "aqui", "aquí", "alla", "allà", "ahi", "ahí",
    "doncs", "vale", "dacord", "d'acord",
    # EN (por si acaso)
    "this", "that", "here", "there", "then",
}

# tokens típicos “de seguimiento” muy frecuentes
_FOLLOWUP_PATTERNS = [
    r"^(y\s+)?(entonces|vale|ok)\b",
    r"^(i\s+)?(això|aixo|doncs)\b",
    r"^(y\s+)?eso\b",
    r"^(y\s+)?esto\b",
    r"^(i\s+)?aquí\b",
    r"^(y\s+)?aquí\b",
    r"^(y\s+)?ahí\b",
    r"^\?$",
]
_FOLLOWUP_RE = re.compile("|".join(_FOLLOWUP_PATTERNS), re.IGNORECASE)

# entidades “claras” mínimas (si aparecen, asumimos tema nuevo)
_ENTITY_HINTS = [
    "pantalla", "screen",
    "t-metropolitana", "tmetropolitana", "t-metropolitana", "rosa", "tarjeta rosa", "passe acompanyant", "pase acompañante"
    "ipf", "dni", "nie", "passaport", "pasaporte",
    "t4", "t-4",
    "amb", "tmb", "atm",
    "carta", "cartes", "denegació", "denegacion", "denegació",
    "provisional", "provisionals",
]


def _has_clear_entities(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False

    tl = t.lower()

    # contiene algún “hint” del dominio
    if any(h in tl for h in _ENTITY_HINTS):
        return True

    # tiene números o formato tipo "1.6", "2.2", etc.
    if re.search(r"\b\d+(\.\d+)+\b", t):
        return True

    # tiene acrónimos (IPF, DNI, etc.) o tokens con mayúsculas tipo "TMB"
    toks = _WORD_RE.findall(t)
    for w in toks:
        if len(w) >= 2 and w.isupper():
            return True

    return False


def _is_short(text: str, max_words: int = 4) -> bool:
    toks = _WORD_RE.findall((text or "").strip())
    return 0 < len(toks) <= max_words


def _is_deictic(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False

    if _FOLLOWUP_RE.search(t):
        return True

    toks = [w.lower() for w in _WORD_RE.findall(t)]
    if any(w in _DEICTIC_TOKENS for w in toks):
        return True

    # preguntas extremadamente vagas tipo "¿y?" / "¿por qué?"
    if len(toks) <= 2 and any(w in {"porque", "por", "qué", "que", "why", "how", "qué", "que", "como", "cómo", "dónde", "cuando", "cuándo", "por qué", "porque", "esto", "eso", "aquí", "ahí", "vale", "ok", "y", "entonces"} for w in toks):
        return True

    return False


def _should_use_context(last_user: str) -> bool:
    """
    Contexto SOLO si:
      - pregunta corta, o deíctica
      Y
      - NO hay entidades claras
    """
    lu = (last_user or "").strip()
    if not lu:
        return False

    if _has_clear_entities(lu):
        return False

    return _is_short(lu, max_words=4) or _is_deictic(lu)


# --------- NUEVO: recorte/limpieza de last_bot ---------
_CLOSINGS_RE = re.compile(
    r"(?is)\b(si\s+necesitas.*?$|si\s+vols.*?$|if\s+you\s+need.*?$|do\s+you\s+have.*?$)\s*",
    re.UNICODE,
)

def _clean_last_bot_for_context(text: str, max_chars: int = 280) -> str:
    t = (text or "").strip()
    if not t:
        return ""

    # quitar source line y similares
    t = _SOURCE_LINE_RE.sub("", t).strip()

    # quitar cierres típicos
    t = _CLOSINGS_RE.sub("", t).strip()

    # colapsar espacios
    t = re.sub(r"\s+", " ", t).strip()

    return _clip(t, max_chars)


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

        # screen_hint persistente SOLO si se detecta explícitamente (pantalla X)
        detected_screen = _extract_screen_name(last_user) or _extract_screen_name(state.get("last_bot", ""))
        if detected_screen:
            state["screen_hint"] = detected_screen

        # topic_question: solo si el usuario realmente menciona tema/pantalla
        if last_user and _should_update_topic(last_user):
            state["topic_question"] = last_user

        last_bot_prev = self._last_bot_message(messages)
        if last_bot_prev:
            state["last_bot"] = last_bot_prev

        state["last_user"] = last_user
        memory = self._build_memory_text(state)

        # rewriter siempre salvo primer turno
        is_first_turn = not bool(state.get("last_bot"))
        effective_question = last_user
        rewrite_used = False
        used_context = False

        if not is_first_turn:
            use_ctx = _should_use_context(last_user)
            used_context = use_ctx

            topic_for_rewrite = state.get("topic_question") or ""
            last_bot_for_rewrite = _clean_last_bot_for_context(state.get("last_bot") or "") if use_ctx else ""

            # si NO toca contexto, vaciamos también topic (evita contaminación)
            if not use_ctx:
                topic_for_rewrite = ""

            effective_question = self.query_rewriter.rewrite(
                last_user=last_user,
                topic_question=topic_for_rewrite,
                last_bot=last_bot_for_rewrite,
                lang=lang,
                screen_hint=state.get("screen_hint") or "",
                root_question="",  # NO usamos root_question
            )
            rewrite_used = True

        # 1) retrieval usando effective_question
        res = self.retriever.retrieve(effective_question, expand_articles=False)
        faqs, arts = res["faqs"], res["articles"]
        arts_global = list(arts)
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
                it
                for it in faqs
                if safe_str((it.meta or {}).get("faq_id")).strip() in ids
            ][: self.cfg.top_k_faq_final]

            judge_debug = {
                "selected_ids": judge_res.selected_ids,
                "confidence": judge_res.confidence,
                "reason": judge_res.reason,
                "candidates_count": len(candidates),
                "selected_count": len(selected_faqs),
            }

            if not selected_faqs:
                selected_faqs = faqs[: self.cfg.top_k_faq_final]
                if judge_debug is not None:
                    judge_debug["fallback_used"] = True

        arts_expanded: List[Any] = []
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
            last_user,
            lang,
            rag_context,
            decision,
            memory=memory,
            source_line=source_line,
        )

        answer = _enforce_source_line(raw_answer, source_line) if (rag_context and source_line) else (raw_answer or "").strip()
        state["last_bot"] = answer

        block_limit = int(self.cfg.top_k_articles_block_context or 15)

        return {
            "language": lang,
            "decision": {"action": decision.action, "reason": decision.reason, "source_kind": decision.source_kind},
            "answer": answer,
            "state": state,
            "debug": {
                "original_last_user": last_user,
                "effective_question": effective_question,
                "rewrite_used": rewrite_used,
                "rewrite_used_context": used_context,
                "screen_hint": state.get("screen_hint"),
                "topic_question": state.get("topic_question"),
                "faq_judge": judge_debug,
                "faqs": {
                    "candidates": _extract_debug_faqs(faqs, "faq", self.cfg.top_k_faq_candidates),
                    "selected": _extract_debug_faqs(faqs_for_ctx, "faq_selected", self.cfg.top_k_faq_final),
                },
                "articles": {
                    "global_candidates": _extract_debug_articles(arts_global, "article_global", self.cfg.top_k_articles_candidates),
                    "expanded_pool": _extract_debug_articles(arts_expanded, "article_expanded", block_limit),
                    "cited_chunks": _extract_debug_articles(arts_for_ctx, "article_cited", block_limit),
                },
            },
        }

    def _build_memory_text(self, state: Dict[str, Any]) -> str:
        parts = ["MEMORIA (último hilo):"]
        if state.get("screen_hint"):
            parts.append(f"- Pantalla/tema: {_clip(state['screen_hint'], 140)}")
        if state.get("topic_question"):
            parts.append(f"- Último tema explícito: {_clip(state['topic_question'], 280)}")
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
