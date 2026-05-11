"""Cliente OpenAI que usa Responses API con herramientas MCP remotas/locales."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

import requests
from openai import APIConnectionError, APITimeoutError, BadRequestError, OpenAI, OpenAIError, RateLimitError

from config.settings import settings

logger = logging.getLogger(__name__)


class OpenAIMCPClient:
    """Cliente conversacional que conecta OpenAI Responses API con el servidor MCP."""

    def __init__(
        self,
        model: str | None = None,
        mcp_server_url: str | None = None,
        server_label: str = "weather_currency_tools",
    ) -> None:
        if not settings.OPENAI_API_KEY:
            raise ValueError("Falta OPENAI_API_KEY. Configúrala en .env antes de iniciar el cliente.")

        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = model or settings.OPENAI_MODEL
        self.mcp_server_url = mcp_server_url or settings.MCP_SERVER_URL
        self.server_label = server_label
        self.history: list[dict[str, str]] = []

    @property
    def tools(self) -> list[dict[str, Any]]:
        """Configuración MCP para la API Responses."""
        return [
            {
                "type": "mcp",
                "server_label": self.server_label,
                "server_description": (
                    "Servidor MCP local con herramientas de conversión de monedas y clima. "
                    "Incluye convert_currency, get_exchange_rates, geocode_city, "
                    "get_current_weather y get_weather_forecast."
                ),
                "server_url": self.mcp_server_url,
                "allowed_tools": [
                    "convert_currency",
                    "get_exchange_rates",
                    "geocode_city",
                    "get_current_weather",
                    "get_weather_forecast",
                ],
                "require_approval": "never",
            }
        ]

    @staticmethod
    def _health_url_from_mcp_url(mcp_url: str) -> str:
        parsed = urlparse(mcp_url)
        return f"{parsed.scheme}://{parsed.netloc}/health"

    def check_server_health(self) -> bool:
        """Comprueba si el servidor MCP local expone el endpoint /health."""
        try:
            health_url = self._health_url_from_mcp_url(self.mcp_server_url)
            response = requests.get(health_url, timeout=5)
            return response.ok
        except requests.RequestException:
            return False

    def ask(self, user_message: str) -> str:
        """
        Envía un mensaje a OpenAI y permite que el modelo use las herramientas MCP.

        Mantiene historial conversacional durante la sesión para dar continuidad a las respuestas.
        """
        clean_message = user_message.strip()
        if not clean_message:
            return "No he recibido ninguna consulta."

        input_messages: list[dict[str, str]] = [*self.history, {"role": "user", "content": clean_message}]

        try:
            logger.info("Enviando consulta a OpenAI con MCP server_url=%s", self.mcp_server_url)
            response = self.client.responses.create(
                model=self.model,
                instructions=(
                    "Eres un asistente útil para consultas de clima y conversión de monedas. "
                    "Usa las herramientas MCP cuando necesites datos reales o actualizados. "
                    "Para clima por ciudad: primero geocode_city y después get_current_weather o get_weather_forecast. "
                    "No inventes datos de clima ni tasas de cambio. Si una herramienta devuelve error, explícalo de forma clara. "
                    "Responde siempre en español, de manera breve y ordenada."
                ),
                tools=self.tools,
                input=input_messages,
                max_output_tokens=900,
            )

            answer = response.output_text or "No se recibió texto final del modelo."
            self.history.extend([
                {"role": "user", "content": clean_message},
                {"role": "assistant", "content": answer},
            ])
            return answer

        except RateLimitError as exc:
            logger.exception("Límite de uso de OpenAI alcanzado")
            return f"Se ha alcanzado un límite de uso de OpenAI. Detalle: {exc}"
        except APITimeoutError as exc:
            logger.exception("Timeout de OpenAI")
            return f"OpenAI tardó demasiado en responder. Detalle: {exc}"
        except APIConnectionError as exc:
            logger.exception("Error de conexión con OpenAI")
            return f"No se pudo conectar con OpenAI. Detalle: {exc}"
        except BadRequestError as exc:
            logger.exception("Petición inválida a OpenAI")
            return (
                "OpenAI rechazó la petición. Revisa que el servidor MCP sea accesible desde la API "
                "y que MCP_SERVER_URL apunte al endpoint correcto, por ejemplo http://localhost:8000/mcp/ "
                "o una URL pública tipo Devtunnel/ngrok. "
                f"Detalle: {exc}"
            )
        except OpenAIError as exc:
            logger.exception("Error de OpenAI")
            return f"Error de OpenAI: {exc}"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error inesperado en el cliente")
            return f"Error inesperado: {exc}"

    def reset_history(self) -> None:
        """Limpia el historial de conversación."""
        self.history.clear()
