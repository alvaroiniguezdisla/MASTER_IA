from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APIError, AuthenticationError, BadRequestError, RateLimitError
from pydantic import ValidationError

from core.rag_system import RAGSystem


SYSTEM_PROMPT = """
Eres un chatbot interno de una empresa ficticia que responde usando técnica RAG.
Reglas obligatorias:
1. Responde únicamente con la información presente en el CONTEXTO recuperado.
2. Si el contexto no contiene la información suficiente, dilo claramente.
3. No inventes datos, cifras, políticas, fechas ni nombres.
4. Mantén un tono profesional, claro y útil.
5. Si el usuario pregunta algo ambiguo, solicita una aclaración breve.
""".strip()


@dataclass
class Chatbot:
    """Chatbot conversacional que integra retrieval + generación."""

    rag_system: RAGSystem
    model_name: str = "gpt-4.1"
    temperature: float = 0.1
    max_history_items: int = 8
    history: List[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
        )

    def _format_history(self) -> str:
        if not self.history:
            return "No hay historial previo."

        recent_history = self.history[-self.max_history_items :]
        lines: list[str] = []
        for message in recent_history:
            role = "Usuario" if message["role"] == "user" else "Asistente"
            lines.append(f"{role}: {message['content']}")
        return "\n".join(lines)

    def answer(self, user_query: str) -> str:
        """Ejecuta el flujo RAG: consulta -> retrieval -> prompt aumentado -> respuesta."""
        retrieved_docs = self.rag_system.retrieve(user_query)
        context = self.rag_system.format_context(retrieved_docs)

        if not context:
            return "No he encontrado fragmentos relevantes en los documentos cargados."

        augmented_prompt = f"""
HISTORIAL RECIENTE DE LA CONVERSACIÓN:
{self._format_history()}

CONTEXTO RECUPERADO DE LOS DOCUMENTOS MARKDOWN:
{context}

PREGUNTA DEL USUARIO:
{user_query}

INSTRUCCIÓN FINAL:
Responde en español, de forma clara y estructurada. Usa solo el contexto recuperado.
""".strip()

        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=augmented_prompt),
                ]
            )
            answer_text = str(response.content).strip()

        except AuthenticationError:
            answer_text = (
                "Error de autenticación: revisa que OPENAI_API_KEY esté correctamente "
                "configurada en el archivo .env."
            )
        except RateLimitError:
            answer_text = (
                "Se ha alcanzado un límite de uso o tokens de la API. Espera unos minutos "
                "o revisa la configuración de tu cuenta de OpenAI."
            )
        except APIConnectionError:
            answer_text = (
                "No se ha podido conectar con la API de OpenAI. Revisa tu conexión a internet."
            )
        except BadRequestError as exc:
            answer_text = (
                "La API ha rechazado la petición. Puede deberse a límite de contexto, "
                f"contenido no permitido o configuración inválida. Detalle: {exc}"
            )
        except APIError as exc:
            answer_text = f"Error de la API de OpenAI: {exc}"
        except ValidationError as exc:
            answer_text = f"Error de validación de datos: {exc}"
        except Exception as exc:
            answer_text = f"Error inesperado al generar la respuesta: {exc}"

        self.history.append({"role": "user", "content": user_query})
        self.history.append({"role": "assistant", "content": answer_text})
        return answer_text

    def reset(self) -> None:
        """Limpia el historial conversacional de la sesión actual."""
        self.history.clear()
