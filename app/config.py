from dataclasses import dataclass
import os

@dataclass(frozen=True)
class RAGConfig:
    pinecone_index_name: str
    namespace_faqs: str
    namespace_articles: str

    top_k_faqs: int
    top_k_articles: int

    min_score_faq_to_answer: float
    min_score_articles_to_answer: float
    overwrite: float
    
    openai_model: str
    openai_embed_model: str
    temperature: float

    max_context_chars: int
    stale_days_warn: int

    @staticmethod
    def from_env() -> "RAGConfig":
        return RAGConfig(
            pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "wiki"),
            namespace_faqs=os.getenv("PINECONE_NAMESPACE_FAQS", "wiki_faqs"),
            namespace_articles=os.getenv("PINECONE_NAMESPACE_ARTICLES", "wiki_articulos"),
            top_k_faqs=int(os.getenv("TOP_K_FAQS", "5")),
            top_k_articles=int(os.getenv("TOP_K_ARTICLES", "8")),
            min_score_faq_to_answer=float(os.getenv("MIN_SCORE_FAQ", "0.80")),
            min_score_articles_to_answer=float(os.getenv("MIN_SCORE_ARTICLES", "0.75")),
            overwrite=float(os.getenv("OVERWRITE","0.92")),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            openai_embed_model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
            max_context_chars=int(os.getenv("MAX_CONTEXT_CHARS", "12000")),
            stale_days_warn=int(os.getenv("STALE_DAYS_WARN", "540")),
        )
