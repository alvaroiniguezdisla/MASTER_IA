"""Lógica principal del chatbot con Ollama SDK."""

from __future__ import annotations

from typing import Generator, Iterable, Optional

import ollama

from core.conversation import ConversationManager
from experts.expert_prompts import EXPERTS, get_expert_name


class OllamaChatbotError(Exception):
    """Error base del chatbot."""


class OllamaConnectionError(OllamaChatbotError):
    """Error cuando no se puede conectar con el servicio local de Ollama."""


class OllamaModelNotAvailableError(OllamaChatbotError):
    """Error cuando el modelo solicitado no está disponible localmente."""


class ExpertChatbot:
    """Chatbot de expertos temáticos usando Ollama SDK en Python."""

    def __init__(self, model: str = "gemma3:1b") -> None:
        self.model = model
        self.conversation = ConversationManager()

    @property
    def active_expert_key(self) -> str:
        return self.conversation.active_expert

    @property
    def active_expert_name(self) -> str:
        return get_expert_name(self.active_expert_key)

    def ensure_ollama_ready(self) -> None:
        """Comprueba que Ollama está activo y que el modelo existe localmente."""
        try:
            response = ollama.list()
        except Exception as exc:  # El SDK puede envolver errores HTTP de varias formas.
            raise OllamaConnectionError(
                "No se pudo conectar con Ollama. Comprueba que el servicio local está iniciado "
                "con el comando: ollama serve"
            ) from exc

        model_names = self._extract_model_names(response)
        if self.model not in model_names:
            raise OllamaModelNotAvailableError(
                f"El modelo '{self.model}' no está disponible localmente. "
                f"Descárgalo con: ollama pull {self.model}"
            )

    def change_expert(self, expert_key: str, reset_history: bool = False) -> None:
        """Cambia el experto activo."""
        self.conversation.set_active_expert(expert_key, reset_history=reset_history)

    def reset_current_history(self) -> None:
        """Reinicia el historial del experto activo."""
        self.conversation.reset_current_history()

    def reset_all_histories(self) -> None:
        """Reinicia todos los historiales."""
        self.conversation.reset_all_histories()

    def stream_answer(self, user_message: str) -> Generator[str, None, str]:
        """Envía un mensaje del usuario y devuelve la respuesta en streaming.

        El método va generando fragmentos de texto para que la interfaz pueda
        imprimirlos en tiempo real. Al finalizar, guarda la respuesta completa en
        el historial del experto activo.
        """
        clean_message = user_message.strip()
        if not clean_message:
            return ""

        self.conversation.add_user_message(clean_message)
        messages = self.conversation.get_messages_for_model()

        full_answer = ""

        try:
            stream = ollama.chat(
                model=self.model,
                messages=messages,
                stream=True,
            )

            for chunk in stream:
                piece = self._extract_content_from_chunk(chunk)
                if piece:
                    full_answer += piece
                    yield piece

        except Exception as exc:
            raise OllamaChatbotError(
                "Se produjo un error al generar la respuesta con Ollama. "
                "Comprueba que Ollama está activo y que el modelo local está disponible."
            ) from exc

        self.conversation.add_assistant_message(full_answer.strip())
        return full_answer

    def get_status_text(self) -> str:
        """Devuelve un resumen breve del estado actual."""
        turns = self.conversation.current_turns_count()
        return (
            f"Modelo local: {self.model} | "
            f"Experto activo: {self.active_expert_name} | "
            f"Mensajes en este historial: {turns}"
        )

    @staticmethod
    def _extract_model_names(response: object) -> set[str]:
        """Extrae nombres de modelos de ollama.list() soportando varias versiones."""
        models: Iterable[object]

        if isinstance(response, dict):
            models = response.get("models", [])
        else:
            models = getattr(response, "models", [])

        names: set[str] = set()
        for model_info in models:
            if isinstance(model_info, dict):
                name = model_info.get("model") or model_info.get("name")
            else:
                name = getattr(model_info, "model", None) or getattr(model_info, "name", None)

            if name:
                names.add(str(name))

        return names

    @staticmethod
    def _extract_content_from_chunk(chunk: object) -> str:
        """Extrae texto de cada fragmento de streaming de Ollama."""
        if isinstance(chunk, dict):
            message = chunk.get("message", {})
            if isinstance(message, dict):
                return message.get("content", "") or ""
            return ""

        message = getattr(chunk, "message", None)
        if message is None:
            return ""

        if isinstance(message, dict):
            return message.get("content", "") or ""

        return getattr(message, "content", "") or ""
