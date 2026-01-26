from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class RetrievedItem:
    source: str  # "faq" | "article"
    score: float
    text: str
    meta: Dict[str, Any]
