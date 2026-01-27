import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from openai import OpenAI


@dataclass
class JudgeResult:
    selected_ids: List[str]
    reason: str = ""
    confidence: float = 0.0  # 0..1


def _clip(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else (s[:n].rstrip() + "…")


def build_judge_prompt(question: str, candidates: List[Dict[str, Any]], lang: str) -> str:
    """
    candidates: list of {id, question, answer}
    """
    # Mantén el prompt corto y determinista: seleccionar evidencias.
    return (
        "Eres un juez/reranker. NO respondas al usuario.\n"
        "Tarea: seleccionar hasta 5 FAQs que contengan evidencia DIRECTA para responder la pregunta.\n"
        "Reglas:\n"
        "- Devuelve SOLO JSON válido.\n"
        "- Selecciona FAQs por relevancia a la pregunta, priorizando las que contienen datos concretos (importes, condiciones, pasos).\n"
        "- Si ninguna FAQ contiene evidencia suficiente, devuelve selected_ids vacío.\n"
        "- No inventes. No asumas.\n\n"
        f"Idioma de la conversación: {lang}\n\n"
        f"PREGUNTA:\n{question}\n\n"
        "CANDIDATOS (id | question | answer):\n"
        + "\n".join(
            f"- {c['id']} | { _clip(c.get('question',''), 160) } | { _clip(c.get('answer',''), 260) }"
            for c in candidates
        )
        + "\n\n"
        "FORMATO JSON (obligatorio):\n"
        '{\n  "selected_ids": ["faq_..", "..."],\n  "reason": "breve",\n  "confidence": 0.0\n}\n'
    )


def parse_judge_json(text: str) -> JudgeResult:
    t = (text or "").strip()
    # intenta extraer el primer JSON
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return JudgeResult(selected_ids=[], reason="invalid_json", confidence=0.0)

    try:
        data = json.loads(t[start : end + 1])
        ids = data.get("selected_ids") or []
        ids = [str(x).strip() for x in ids if str(x).strip()]
        ids = ids[:5]
        reason = str(data.get("reason") or "").strip()
        conf = float(data.get("confidence") or 0.0)
        if conf < 0:
            conf = 0.0
        if conf > 1:
            conf = 1.0
        return JudgeResult(selected_ids=ids, reason=reason, confidence=conf)
    except Exception:
        return JudgeResult(selected_ids=[], reason="json_parse_error", confidence=0.0)


class FAQJudge:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def select(self, question: str, candidates: List[Dict[str, Any]], lang: str) -> JudgeResult:
        prompt = build_judge_prompt(question, candidates, lang)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a precise reranking judge."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        out = (resp.choices[0].message.content or "").strip()
        return parse_judge_json(out)
