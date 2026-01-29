from openai import OpenAI
from app.config import RAGConfig
from app.policy.decider import Decision

def system_prompt(lang: str) -> str:
    return (
        "Eres un asistente experto interno de AMB (operativo, práctico y preciso).\n\n"
        "IDIOMA (muy importante):\n"
        "1) Responde SIEMPRE en el mismo idioma que use el usuario en su último mensaje.\n"
        "2) Si el usuario mezcla idiomas, responde en el idioma predominante del último mensaje.\n"
        "3) Si no está claro, responde en español.\n\n"
        "ESTILO:\n"
        "- Tono cercano y natural. Directo.\n"
        "- Prioriza ayudar a usar la aplicación y resolver dudas concretas.\n"
        "- Por defecto 1–4 frases. Si el usuario pide pasos, usa lista numerada.\n"
        "- No cierres con 'Si quieres dime...'. Da el siguiente paso útil.\n"
        "- Máximo 1 pregunta y solo si desbloquea la respuesta.\n"
        "- Prohibido prometer/garantizar ('te lo prometo', '100% seguro'). Usa lenguaje prudente.\n\n"
        "REGLAS RAG:\n"
        "- Usa SOLO la información del CONTEXTO cuando exista.\n"
        "- Usa MEMORIA solo para continuidad.\n\n"
        "MODO CLARIFY (si Mode=clarify):\n"
        "- No inventes. Explica en 1 frase qué falta y haz 1 sola pregunta concreta.\n\n"
        "REGLA ANTI-GENERALIZACIONES:\n"
        "- Si el usuario generaliza ('nunca', 'nada', 'siempre', 'gratis para todo'):\n"
        "  corrige con 'En general..., pero...' si el CONTEXTO tiene excepciones.\n\n"
        "COSTES / TARIFAS:\n"
        "- Responde en 2 capas: (A) regla general (B) excepciones del CONTEXTO.\n"
        "- NO calcules totales ni sumes importes. Presenta importes por concepto (p.ej., tasa, soporte).\n"
        "- Solo menciona importes si aparecen en el CONTEXTO.\n\n"
        "CITAS:\n"
        "- Si usas CONTEXTO, termina con EXACTAMENTE la línea source_line.\n"
        "- Si NO usas CONTEXTO, NO pongas Fuente/Font.\n\n"
    )

def user_prompt(question: str, context: str, decision: Decision, memory: str = "", source_line: str = "") -> str:
    caution = (decision.caution_note or "").strip()
    source_line = (source_line or "").strip()

    caution_line = f"- Cautela (si aplica): {caution}\n" if caution else ""
    source_line_line = f"- source_line (si usas CONTEXTO): {source_line}\n" if source_line else ""

    return (
        f"PREGUNTA:\n{question}\n\n"
        f"{(memory.strip() + '\n\n') if memory else ''}"
        f"CONTEXTO:\n{context if context else '(sin contexto suficiente)'}\n\n"
        f"INSTRUCCIONES:\n"
        f"- Mode: {decision.action}\n"
        f"{caution_line}"
        f"{source_line_line}"
        f"- Si NO usas CONTEXTO, NO pongas 'Fuente:' / 'Font:'.\n"
    ).strip()

class LLMResponder:
    def __init__(self, cfg: RAGConfig, client: OpenAI):
        self.cfg = cfg
        self.client = client

    def run(self, question: str, lang: str, context: str, decision: Decision, memory: str = "", source_line: str = "") -> str:
        resp = self.client.chat.completions.create(
            model=self.cfg.openai_model,
            messages=[
                {"role": "system", "content": system_prompt(lang)},
                {"role": "user", "content": user_prompt(question, context, decision, memory=memory, source_line=source_line)},
            ],
            temperature=self.cfg.temperature,
        )
        return (resp.choices[0].message.content or "").strip()
