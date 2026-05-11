"""Cliente de OpenAI usando Responses API y Structured Outputs."""

import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from openai import APIConnectionError, APIError, APITimeoutError, AuthenticationError, BadRequestError, RateLimitError
from pydantic import ValidationError

from models.linkedin_post import LinkedinPost


DEFAULT_MODEL = "gpt-4o-2024-08-06"

SYSTEM_PROMPT = """
Eres un especialista en copywriting para LinkedIn.
Tu tarea es transformar la idea del usuario en un post profesional, claro y útil.

Reglas de estilo:
- Escribe en español.
- Mantén un tono profesional, cercano y creíble.
- Evita promesas exageradas o afirmaciones no verificables.
- El contenido debe parecer escrito por una persona, no por una plantilla genérica.
- Incluye una apertura atractiva, desarrollo con valor y cierre con llamada a la conversación.
- Los hashtags deben ser relevantes y específicos.
- La categoría debe resumir el tema principal del post.
""".strip()


class ChatbotError(Exception):
    """Error base de la aplicación."""


class ConfigurationError(ChatbotError):
    """Error de configuración local, por ejemplo API key ausente."""


class ModelRefusalError(ChatbotError):
    """La API rechazó generar la respuesta por motivos de seguridad."""


class ResponseValidationError(ChatbotError):
    """La respuesta no pudo validarse contra el modelo Pydantic."""


class OpenAIStructuredClient:
    """Encapsula la llamada a OpenAI con Structured Outputs."""

    def __init__(self, model: Optional[str] = None) -> None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "No se encontró OPENAI_API_KEY. Añade tu clave en el archivo .env."
            )

        self.model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
        self.client = OpenAI(api_key=api_key)

    def generate_linkedin_post(self, idea: str) -> LinkedinPost:
        """Genera y valida un post de LinkedIn a partir de una idea del usuario."""
        try:
            response = self.client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Genera un post de LinkedIn a partir de esta idea o descripción. "
                            "Devuelve todos los campos solicitados de forma completa y coherente.\n\n"
                            f"Idea del usuario: {idea}"
                        ),
                    },
                ],
                text_format=LinkedinPost,
                max_output_tokens=1200,
            )

            self._raise_if_incomplete(response)

            post = getattr(response, "output_parsed", None)
            if post is not None:
                return post

            refusal = self._extract_refusal(response)
            if refusal:
                raise ModelRefusalError(refusal)

            raise ResponseValidationError(
                "La API no devolvió un objeto LinkedinPost válido."
            )

        except ValidationError as error:
            raise ResponseValidationError(
                "La respuesta no cumple el esquema Pydantic definido para LinkedinPost."
            ) from error
        except AuthenticationError as error:
            raise ConfigurationError(
                "La API key no es válida o no tiene permisos suficientes."
            ) from error
        except RateLimitError as error:
            raise ChatbotError(
                "Se ha alcanzado un límite de uso, cuota o velocidad de la API."
            ) from error
        except (APIConnectionError, APITimeoutError) as error:
            raise ChatbotError(
                "No se pudo conectar con la API de OpenAI o la petición agotó el tiempo de espera."
            ) from error
        except BadRequestError as error:
            raise ChatbotError(
                "La petición fue rechazada. Revisa el modelo elegido, el esquema o el tamaño de la entrada."
            ) from error
        except APIError as error:
            raise ChatbotError(
                "OpenAI devolvió un error inesperado durante la generación."
            ) from error

    @staticmethod
    def _raise_if_incomplete(response: object) -> None:
        """Detecta respuestas incompletas, por ejemplo por límite de tokens."""
        status = getattr(response, "status", None)
        if status != "incomplete":
            return

        details = getattr(response, "incomplete_details", None)
        reason = getattr(details, "reason", "motivo desconocido")
        raise ChatbotError(
            f"La respuesta quedó incompleta antes de terminar. Motivo: {reason}."
        )

    @staticmethod
    def _extract_refusal(response: object) -> Optional[str]:
        """Extrae un posible refusal de la respuesta de la API de forma defensiva."""
        for item in getattr(response, "output", []) or []:
            for content_item in getattr(item, "content", []) or []:
                refusal = getattr(content_item, "refusal", None)
                if refusal:
                    return str(refusal)
                if getattr(content_item, "type", None) == "refusal":
                    text = getattr(content_item, "text", None)
                    return str(text or "El modelo rechazó la solicitud.")
        return None
