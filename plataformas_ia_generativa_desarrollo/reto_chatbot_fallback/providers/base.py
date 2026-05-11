"""Contrato común para los proveedores de modelos de lenguaje."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from core.conversation import Message


class ProviderError(RuntimeError):
    """Error controlado al invocar un proveedor de IA."""


class BaseProvider(ABC):
    """Interfaz común que implementan OpenAI, Anthropic y Google Gemini."""

    name: str

    @abstractmethod
    def stream_response(self, system_prompt: str, messages: list[Message]) -> Iterable[str]:
        """Genera una respuesta en streaming a partir del historial recibido."""
        raise NotImplementedError
