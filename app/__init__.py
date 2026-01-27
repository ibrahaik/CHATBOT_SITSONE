import re
from datetime import datetime, timezone
from typing import Any, Optional

def safe_str(x: Any) -> str:
    return "" if x is None else str(x)

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def parse_iso(dt: Optional[str]) -> Optional[datetime]:
    if not dt:
        return None
    try:
        s = dt.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        return None

def is_stale(updated_at_iso: Optional[str], stale_days: int) -> bool:
    dt = parse_iso(updated_at_iso)
    if not dt:
        return False
    return (now_utc() - dt).days >= stale_days

_LANG_CA_HINT = re.compile(r"\b(què|com|on|aquesta|aquest|cerca|puc|he|cal|si us plau)\b", re.IGNORECASE)
_LANG_ES_HINT = re.compile(r"\b(qué|como|dónde|esta|este|buscar|puedo|tengo|debo|por favor)\b", re.IGNORECASE)
_LANG_EN_HINT = re.compile(r"\b(what|how|where|please|can you|i have|do i|is it|free)\b", re.IGNORECASE)

def detect_language(text: str) -> str:
    """
    Heurística simple: 'es' | 'ca' | 'en'.
    Reglas:
      - si EN claro -> en
      - si ES claro -> es
      - si CA claro -> ca
      - si duda entre ES/CA -> ES (mejor UX y evita mezcla)
      - si vacío -> ES
    """
    t = (text or "").strip()
    if not t:
        return "es"

    en = bool(_LANG_EN_HINT.search(t))
    ca = bool(_LANG_CA_HINT.search(t))
    es = bool(_LANG_ES_HINT.search(t))

    if en and not (es or ca):
        return "en"
    if es and not ca:
        return "es"
    if ca and not es:
        return "ca"

    # Duda: español por defecto
    return "es"
