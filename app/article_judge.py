# app/article_judge.py
import json
from dataclasses import dataclass
from typing import List, Dict, Any
from openai import OpenAI

@dataclass
class ArticleJudgeResult:
    best_index: int
    reason: str = ""
    confidence: float = 0.0  # 0..1

def _clip(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else (s[:n].rstrip() + "…")

ARTICLE_JUDGE_PROMPT = """
Ets un reranker. NO responguis a l'usuari.

TASCA
- Donada la PREGUNTA ORIGINAL de l'usuari i una llista de 8 fragments d'articles (candidats),
  tria el candidat que conté evidència més directa per respondre.

REGLLES
- NO inventis.
- Retorna NOMÉS JSON vàlid: {"best_index":0,"reason":"...","confidence":0.0}
- best_index és un enter 0..7.
- Si cap candidat encaixa, tria el més proper (però sense inventar).
""".strip()

class ArticleJudge:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def pick_best(self, question_original: str, candidates: List[Dict[str, Any]]) -> ArticleJudgeResult:
        # candidates: [{titulo, block_id, chunk_index, text_clean_preview}]
        prompt = (
            f"PREGUNTA ORIGINAL:\n{question_original}\n\n"
            "CANDIDATS (index | titulo | block_id | chunk_index | text_clean_preview):\n"
            + "\n".join(
                f"- {i} | {c.get('titulo','')} | {c.get('block_id','')} | {c.get('chunk_index','')} | "
                f"{_clip(c.get('text_clean_preview',''), 220)}"
                for i, c in enumerate(candidates[:8])
            )
            + "\n\nFORMAT JSON:\n"
            '{ "best_index": 0, "reason": "breu", "confidence": 0.0 }\n'
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": ARTICLE_JUDGE_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )

        out = (resp.choices[0].message.content or "").strip()
        start = out.find("{")
        end = out.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return ArticleJudgeResult(best_index=0, reason="invalid_json", confidence=0.0)

        try:
            data = json.loads(out[start:end+1])
            bi = int(data.get("best_index", 0))
            if bi < 0:
                bi = 0
            if bi > 7:
                bi = 7
            reason = str(data.get("reason") or "").strip()
            conf = float(data.get("confidence") or 0.0)
            conf = max(0.0, min(1.0, conf))
            return ArticleJudgeResult(best_index=bi, reason=reason, confidence=conf)
        except Exception:
            return ArticleJudgeResult(best_index=0, reason="json_parse_error", confidence=0.0)
