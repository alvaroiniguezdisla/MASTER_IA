import json
import logging
from typing import Type, TypeVar

from fastapi import HTTPException
from openai import APIConnectionError, APITimeoutError, RateLimitError, APIStatusError
from openai import AsyncAzureOpenAI
from pydantic import BaseModel, ValidationError

from app.config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class AzureOpenAIService:
    """
    Servicio común para invocar Azure OpenAI usando salidas estructuradas.

    Cada endpoint construye un prompt específico y pasa un modelo Pydantic.
    El servicio fuerza la respuesta JSON mediante response_format=json_schema
    y valida la salida con Pydantic antes de devolverla al router.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client: AsyncAzureOpenAI | None = None

    def _get_client(self) -> AsyncAzureOpenAI:
        if not self.settings.azure_openai_api_key or not self.settings.azure_openai_endpoint:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Azure OpenAI no está configurado. Define AZURE_OPENAI_API_KEY, "
                    "AZURE_OPENAI_ENDPOINT y AZURE_OPENAI_DEPLOYMENT en el archivo .env."
                ),
            )

        if self._client is None:
            self._client = AsyncAzureOpenAI(
                api_key=self.settings.azure_openai_api_key,
                azure_endpoint=self.settings.azure_openai_endpoint,
                api_version=self.settings.azure_openai_api_version,
                timeout=self.settings.request_timeout_seconds,
            )

        return self._client

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        temperature: float = 0.4,
    ) -> T:
        """
        Ejecuta una llamada a Azure OpenAI y devuelve un objeto Pydantic validado.
        """

        json_schema = response_model.model_json_schema()
        client = self._get_client()

        try:
            completion = await client.chat.completions.create(
                model=self.settings.azure_openai_deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "strict": True,
                        "schema": json_schema,
                    },
                },
            )

            raw_content = completion.choices[0].message.content
            if not raw_content:
                raise HTTPException(
                    status_code=502,
                    detail="Azure OpenAI devolvió una respuesta vacía.",
                )

            try:
                return response_model.model_validate_json(raw_content)
            except (ValidationError, json.JSONDecodeError) as exc:
                logger.exception("La respuesta del modelo no cumple el esquema Pydantic.")
                raise HTTPException(
                    status_code=502,
                    detail="La respuesta generada por la IA no cumple el esquema esperado.",
                ) from exc

        except RateLimitError as exc:
            logger.exception("Límite de uso o cuota de Azure OpenAI superado.")
            raise HTTPException(
                status_code=429,
                detail="Se ha superado el límite de uso de Azure OpenAI. Inténtalo más tarde.",
            ) from exc

        except (APIConnectionError, APITimeoutError) as exc:
            logger.exception("Error de conexión o timeout con Azure OpenAI.")
            raise HTTPException(
                status_code=503,
                detail="No se ha podido conectar con Azure OpenAI en este momento.",
            ) from exc

        except APIStatusError as exc:
            logger.exception("Azure OpenAI devolvió un error HTTP.")
            raise HTTPException(
                status_code=exc.status_code,
                detail=f"Azure OpenAI devolvió un error: {exc.message}",
            ) from exc
