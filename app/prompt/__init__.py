from openai import OpenAI
from app.config import RAGConfig
from app.policy.decider import Decision


def system_prompt(lang: str) -> str:
    if lang == "es":
        return (
            "Eres un experto interno de AMB. Responde en español.\n"
            "Por defecto responde en 1–3 frases, directo.\n"
            "Usa SOLO la información del CONTEXTO.\n"
            "Si el usuario pide algo general o dice 'TODO', responde con un resumen estructurado en viñetas: "
            "qué es, para qué sirve, cómo se solicita, renovación/caducidad, incidencias comunes.\n"
            "Si falta información en el CONTEXTO, responde con lo disponible y menciona la limitación.\n"
            "Añade una cita breve al final empezando por 'Fuente:'.\n"
            "Si hay riesgo de desactualización, añade 1 frase de cautela.\n"
            """Si la pregunta no tiene relación con AMB o transporte metropolitano:\n
            - No digas que “no dispones de información”.\n
            - Redirige educadamente al ámbito AMB indicando qué tipo de ayuda sí puedes ofrecer.\n"""
            """Si el usuario hace una meta-pregunta (quién eres / qué puedes hacer / eres inteligente):\n
            - Responde con 1 frase útil (sin decir “no puedo responder”).\n
            - Luego haz 1 pregunta para llevarlo a AMB (p. ej. “¿Qué necesitas hacer hoy?”).
"""
        )
    return (
        "Ets un expert intern d'AMB. Respon en català.\n"
        "Per defecte respon en 1–3 frases, directe.\n"
        "Fes servir NOMÉS la informació del CONTEXT.\n"
        "Si l'usuari demana una visió general o diu 'TOT', respon amb un resum estructurat en punts: "
        "què és, per a què serveix, com se sol·licita, renovació/caducitat, incidències comunes.\n"
        "Si falta informació al CONTEXT, respon amb el disponible i indica la limitació.\n"
        "Afegeix una cita breu al final començant per 'Font:'.\n"
        "Si hi ha risc que estigui desactualitzat, afegeix 1 frase de cautela.\n"
        """Si la pregunta no està relacionada amb l’AMB o el transport metropolità:\n
        - No diguis que “no disposes d’informació”.\n
        - Redirigeix educadament cap a l’àmbit AMB indicant en què pots ajudar.\n
        """
        """Si l’usuari fa una meta-pregunta (qui ets / què pots fer / ets intel·ligent):\n
            - Respon amb 1 frase útil (sense dir “no puc respondre”).\n
            - Després fes 1 pregunta per portar-lo a AMB (p. ex. “Què necessites fer avui?”).
"""
    )


def user_prompt(question: str, context: str, decision: Decision) -> str:
    caution = decision.caution_note or ""
    cite = decision.citation_hint or ""
    mode = decision.action

    return (
        f"PREGUNTA:\n{question}\n\n"
        f"CONTEXTO:\n{context if context else '(sin contexto suficiente)'}\n\n"
        f"INSTRUCCIONES:\n"
        f"- Mode: {mode}\n"
        f"- Si Mode=answer: responde usando el CONTEXTO. No hagas preguntas.\n"
        f"- Aviso (si aplica): {caution}\n"
        f"- Cita sugerida: {cite}\n"
    ).strip()


class LLMResponder:
    def __init__(self, cfg: RAGConfig, client: OpenAI):
        self.cfg = cfg
        self.client = client

    def run(self, question: str, lang: str, context: str, decision: Decision) -> str:
        resp = self.client.chat.completions.create(
            model=self.cfg.openai_model,
            messages=[
                {"role": "system", "content": system_prompt(lang)},
                {"role": "user", "content": user_prompt(question, context, decision)},
            ],
            temperature=self.cfg.temperature,
        )
        return (resp.choices[0].message.content or "").strip()
