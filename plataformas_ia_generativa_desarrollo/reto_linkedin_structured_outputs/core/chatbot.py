"""Lógica principal del chatbot de generación de posts."""

from models.linkedin_post import LinkedinPost
from core.api_client import OpenAIStructuredClient


class LinkedinPostChatbot:
    """Chatbot por terminal para generar posts de LinkedIn."""

    def __init__(self) -> None:
        self.api_client = OpenAIStructuredClient()

    def generate_post(self, idea: str) -> LinkedinPost:
        """Valida la entrada y solicita la generación del post."""
        clean_idea = idea.strip()
        if not clean_idea:
            raise ValueError("La idea del post no puede estar vacía.")
        return self.api_client.generate_linkedin_post(clean_idea)

    @staticmethod
    def format_post(post: LinkedinPost) -> str:
        """Devuelve una representación legible del post generado."""
        hashtags = " ".join(post.hashtags)
        return (
            "\n" + "=" * 70 + "\n"
            "POST DE LINKEDIN GENERADO\n"
            + "=" * 70 + "\n\n"
            f"Título:\n{post.title}\n\n"
            f"Contenido:\n{post.content}\n\n"
            f"Hashtags:\n{hashtags}\n\n"
            f"Categoría:\n{post.category}\n"
            + "=" * 70 + "\n"
        )
