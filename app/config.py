# app/config.py
from dataclasses import dataclass
import os

@dataclass(frozen=True)
class RAGConfig:
    pinecone_index_name: str
    namespace_faqs: str
    namespace_articles: str

    top_k_faqs: int

    # Artículos: recupera muchos, luego decide + expande
    top_k_articles_candidates: int
    top_k_articles_final: int  # <-- top global (candidatos iniciales que pasan al decider)

    # NUEVO: límite de chunks del bloque expandido (cited_chunks / contexto)
    top_k_articles_block_context: int

    # Umbrales policy
    min_score_faq_to_answer: float
    min_score_articles_to_answer: float
    overwrite: float

    openai_model: str
    openai_embed_model: str
    temperature: float

    max_context_chars: int
    stale_days_warn: int

    # Diversity caps
    max_chunks_per_titulo: int
    max_chunks_per_block: int

    top_k_faq_candidates: int
    top_k_faq_final: int

    judge_model: str

    top_k_articles_within_article: int
    article_neighbor_window: int
    article_min_chunks: int

    @staticmethod
    def from_env() -> "RAGConfig":
        return RAGConfig(
            pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "wiki"),
            namespace_faqs=os.getenv("PINECONE_NAMESPACE_FAQS", "wiki_faqs"),
            namespace_articles=os.getenv("PINECONE_NAMESPACE_ARTICLES", "wiki_articulos"),

            top_k_faqs=int(os.getenv("TOP_K_FAQS", "5")),

            # top global (antes de expandir)
            top_k_articles_candidates=int(os.getenv("TOP_K_ARTICLES_CANDIDATES", "30")),
            top_k_articles_final=int(os.getenv("TOP_K_ARTICLES_FINAL", "8")),

            # bloque expandido (después del filtro block_id)
            top_k_articles_block_context=int(os.getenv("TOP_K_ARTICLES_BLOCK_CONTEXT", "15")),

            min_score_faq_to_answer=float(os.getenv("MIN_SCORE_FAQ", "0.45")),
            min_score_articles_to_answer=float(os.getenv("MIN_SCORE_ARTICLES", "0.45")),
            overwrite=float(os.getenv("OVERWRITE", "0.75")),

            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            openai_embed_model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),

            max_context_chars=int(os.getenv("MAX_CONTEXT_CHARS", "12000")),
            stale_days_warn=int(os.getenv("STALE_DAYS_WARN", "540")),

            max_chunks_per_titulo=int(os.getenv("MAX_CHUNKS_PER_TITULO", "2")),
            max_chunks_per_block=int(os.getenv("MAX_CHUNKS_PER_BLOCK", "2")),

            top_k_faq_candidates=int(os.getenv("TOP_K_FAQ_CANDIDATES", "25")),
            top_k_faq_final=int(os.getenv("TOP_K_FAQ_FINAL", "5")),

            judge_model=os.getenv("JUDGE_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),

            top_k_articles_within_article=int(os.getenv("TOP_K_ARTICLES_WITHIN_ARTICLE", "30")),
            article_neighbor_window=int(os.getenv("ARTICLE_NEIGHBOR_WINDOW", "1")),
            article_min_chunks=int(os.getenv("ARTICLE_MIN_CHUNKS", "3")),
        )
