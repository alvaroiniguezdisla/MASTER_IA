"""Modelo Pydantic para validar posts de LinkedIn generados con Structured Outputs."""

from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LinkedinPost(BaseModel):
    """Estructura estricta esperada para un post de LinkedIn.

    Todos los campos son obligatorios y se prohíben propiedades adicionales
    para que la respuesta de la API cumpla exactamente con este esquema.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    title: str = Field(
        description="Título breve, claro y atractivo para el post de LinkedIn."
    )
    content: str = Field(
        description="Contenido completo del post, redactado de forma profesional."
    )
    hashtags: List[str] = Field(
        description="Lista de hashtags relevantes para acompañar el post."
    )
    category: str = Field(
        description="Categoría temática principal del post."
    )

    @field_validator("title", "content", "category")
    @classmethod
    def text_fields_cannot_be_empty(cls, value: str) -> str:
        """Evita campos de texto vacíos o solo con espacios."""
        if not value.strip():
            raise ValueError("El campo no puede estar vacío.")
        return value.strip()

    @field_validator("hashtags")
    @classmethod
    def hashtags_must_be_valid(cls, value: List[str]) -> List[str]:
        """Valida y normaliza la lista de hashtags."""
        if not value:
            raise ValueError("Debe incluir al menos un hashtag.")

        normalized_hashtags: List[str] = []
        for hashtag in value:
            clean_hashtag = hashtag.strip()
            if not clean_hashtag:
                raise ValueError("Los hashtags no pueden estar vacíos.")
            if not clean_hashtag.startswith("#"):
                clean_hashtag = f"#{clean_hashtag}"
            normalized_hashtags.append(clean_hashtag.replace(" ", ""))

        return normalized_hashtags
