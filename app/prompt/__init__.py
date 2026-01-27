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
        "- Prioriza ayudar a usar la aplicación y resolver dudas concretas, no solo describir pantallas.\n"
        "- Por defecto 1–4 frases. Si el usuario pide pasos, usa lista numerada.\n"
        "- No cierres con coletillas tipo 'Si quieres dime...'. En su lugar, ofrece el siguiente paso útil.\n"
        "- Máximo 1 pregunta y solo si desbloquea la respuesta.\n"
        "- No empieces con negaciones tipo 'no puedo / no dispongo'. Empieza con lo que SÍ puedes hacer.\n\n"
        "REGLAS RAG:\n"
        "- Usa SOLO la información del CONTEXTO cuando exista.\n"
        "- Usa MEMORIA solo para referencias ('eso', 'lo anterior') y continuidad.\n"
        "- Si el CONTEXTO es insuficiente: di qué sí puedes afirmar + qué dato falta (1 pregunta máx).\n\n"
        "REGLA ANTI-GENERALIZACIONES (crítica):\n"
        "- Si el usuario generaliza o absolutiza (p. ej., 'nunca', 'nada', 'ninguna tarifa', 'siempre'):\n"
        "  1) NO confirmes la generalización automáticamente.\n"
        "  2) Verifica si en el CONTEXTO hay matices, condiciones o excepciones.\n"
        "  3) Si las hay, corrige la generalización explícitamente con un 'En general..., pero...' y menciona la(s) excepción(es).\n"
        "  4) Si NO hay evidencia en CONTEXTO, dilo con cautela (sin inventar) y, como mucho, haz 1 pregunta.\n\n"
        "COSTES / PAGOS / TARIFAS (prioridad alta):\n"
        "- Cuando la pregunta trate de dinero (coste, pago, tasa, tarifa, anualidad, recarga, gratis):\n"
        "  • Responde SIEMPRE en 2 capas:\n"
        "    (A) Regla general (lo más habitual).\n"
        "    (B) Excepciones/condiciones relevantes del CONTEXTO (aunque el usuario no las haya mencionado).\n"
        "- Si el usuario pregunta por 'nunca pagar' o 'gratis para todo', debes aclarar límites y casos especiales si existen en CONTEXTO.\n"
        "- Si hay importes concretos en CONTEXTO, inclúyelos tal cual.\n\n"
        "CUANDO EL USUARIO PIDA 'TODO' / 'TOT' O ALGO GENERAL:\n"
        "- Responde con viñetas y esta estructura:\n"
        "  • qué es / què és\n"
        "  • para qué sirve / per a què serveix\n"
        "  • cómo se usa (pasos) / com s’utilitza (passos)\n"
        "  • renovación/caducidad / renovació/caducitat\n"
        "  • incidencias comunes / incidències comunes\n\n"
        "CITAS:\n"
        "- Si usas información del CONTEXTO, al final debes añadir UNA sola línea de fuente.\n"
        "- Esa línea debe ser EXACTAMENTE la que recibes en INSTRUCCIONES (source_line).\n"
        "- Si NO usas CONTEXTO, NO pongas 'Fuente:' ni 'Font:'.\n\n"
        "DESACTUALIZACIÓN:\n"
        "- Si hay riesgo de desactualización (según INSTRUCCIONES), añade 1 frase breve de cautela.\n"
    )


def user_prompt(
    question: str,
    context: str,
    decision: Decision,
    memory: str = "",
    source_line: str = "",
) -> str:
    caution = (decision.caution_note or "").strip()
    source_line = (source_line or "").strip()

    caution_line = f"- Avís/Avís (si aplica): {caution}\n" if caution else ""
    source_line_line = f"- source_line (si uses CONTEXTO): {source_line}\n" if source_line else ""

    return (
        f"PREGUNTA (último mensaje del usuario):\n{question}\n\n"
        f"{(memory.strip() + '\n\n') if memory else ''}"
        f"CONTEXTO:\n{context if context else '(sin contexto suficiente)'}\n\n"
        f"INSTRUCCIONES:\n"
        f"- Mode: {decision.action}\n"
        f"- Integra la MEMORIA con la PREGUNTA para mantener el hilo.\n"
        f"- Responde usando el CONTEXTO cuando exista (no inventes).\n"
        f"{caution_line}"
        f"{source_line_line}"
        f"- Si usas CONTEXTO, termina con EXACTAMENTE la línea source_line.\n"
        f"- Si NO usas CONTEXTO, NO pongas 'Fuente:' / 'Font:'.\n"
    ).strip()


class LLMResponder:
    def __init__(self, cfg: RAGConfig, client: OpenAI):
        self.cfg = cfg
        self.client = client

    def run(
        self,
        question: str,
        lang: str,
        context: str,
        decision: Decision,
        memory: str = "",
        source_line: str = "",
    ) -> str:
        resp = self.client.chat.completions.create(
            model=self.cfg.openai_model,
            messages=[
                {"role": "system", "content": system_prompt(lang)},
                {"role": "user", "content": user_prompt(question, context, decision, memory=memory, source_line=source_line)},
            ],
            temperature=self.cfg.temperature,
        )
        return (resp.choices[0].message.content or "").strip()
