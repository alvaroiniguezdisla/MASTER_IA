"""Lógica principal del chatbot de terminal."""

from __future__ import annotations

from pydantic import ValidationError

from agents import MaxTurnsExceeded, Runner
from agents.main_agent import main_agent
from agents.specialized_agents import LinkedinPost
from core.conversation import ConversationHistory


class LinkedInAgentsChatbot:
    """Chatbot multiagente para generar posts de LinkedIn."""

    def __init__(self) -> None:
        self.history = ConversationHistory()

    def generate_post(self, user_message: str) -> tuple[LinkedinPost, str]:
        """Ejecuta el agente principal y devuelve el post y el agente final usado."""
        prompt = self.history.build_context_prompt(user_message)
        self.history.add_user_message(user_message)

        try:
            result = Runner.run_sync(main_agent, prompt, max_turns=6)
        except MaxTurnsExceeded as exc:
            raise RuntimeError(
                "Se alcanzó el límite de turnos entre agentes. Prueba con una petición más concreta."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                "No se pudo completar la llamada al modelo. Revisa OPENAI_API_KEY y la conexión."
            ) from exc

        final_output = result.final_output
        agent_name = getattr(result.last_agent, "name", "Agente desconocido")

        try:
            if isinstance(final_output, LinkedinPost):
                post = final_output
            elif isinstance(final_output, dict):
                post = LinkedinPost.model_validate(final_output)
            else:
                post = LinkedinPost.model_validate_json(str(final_output))
        except ValidationError as exc:
            raise RuntimeError(
                "El modelo no devolvió una publicación con la estructura esperada."
            ) from exc

        self.history.add_assistant_message(self.format_post(post))
        return post, agent_name

    @staticmethod
    def format_post(post: LinkedinPost) -> str:
        """Formatea la salida estructurada para terminal."""
        hashtags = " ".join(post.hashtags)
        return (
            f"Título: {post.title}\n\n"
            f"Contenido:\n{post.content}\n\n"
            f"Hashtags: {hashtags}\n"
            f"Categoría: {post.category}"
        )
