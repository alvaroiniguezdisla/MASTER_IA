"""Lógica principal del chatbot y cascada de fallback."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass

from core.conversation import Conversation
from providers import AnthropicProvider, GeminiProvider, OpenAIProvider


@dataclass
class ProviderAttempt:
    """Resultado interno de un intento de proveedor."""

    provider_name: str
    answer: str


class FallbackChatbot:
    """
    Chatbot con fallback automático OpenAI -> Anthropic -> Google Gemini.

    La conversación se mantiene en memoria mediante Conversation. Cada petición
    nueva envía el historial completo al proveedor activo, por lo que el contexto
    se conserva aunque haya cambio de proveedor por fallback.
    """

    def __init__(self) -> None:
        self.conversation = Conversation()
        self.system_prompt = os.getenv(
            "SYSTEM_PROMPT",
            (
                "Eres un asistente de IA profesional, claro y útil. "
                "Responde en español, mantén el contexto de la conversación "
                "y reconoce cuando no tengas información suficiente."
            ),
        )
        self.temperature = float(os.getenv("TEMPERATURE", "0.3"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "800"))
        self.fallback_response = os.getenv(
            "FALLBACK_RESPONSE",
            (
                "Lo siento, ahora mismo no puedo contactar con ningún proveedor "
                "de IA. Inténtalo de nuevo más tarde."
            ),
        )
        self.providers = [OpenAIProvider(), AnthropicProvider(), GeminiProvider()]

    def stream_chat(self, user_message: str) -> Iterator[str]:
        """
        Procesa un mensaje de usuario, aplica fallback y emite texto en streaming.

        La función produce tanto avisos de cambio de proveedor como fragmentos de
        respuesta del asistente. El historial solo guarda la respuesta final válida.
        """
        self.conversation.add_user_message(user_message)
        last_error: Exception | None = None

        for index, provider in enumerate(self.providers):
            if index == 0:
                yield f"\n[sistema] Intentando responder con {provider.name}...\n"
            else:
                yield f"\n\n[sistema] Fallback activado. Probando con {provider.name}...\n"

            try:
                attempt = self._attempt_provider(provider)
                self.conversation.add_assistant_message(attempt.answer)
                yield f"\n\n[sistema] Respuesta completada con {attempt.provider_name}.\n"
                return
            except Exception as exc:  # noqa: BLE001 - se capturan fallos de red/API/cuota/autenticación.
                last_error = exc
                yield (
                    f"\n[sistema] {provider.name} ha fallado: "
                    f"{exc.__class__.__name__}: {exc}\n"
                )

        # Ningún proveedor ha funcionado: respuesta preconfigurada.
        self.conversation.add_assistant_message(self.fallback_response)
        yield "\n[sistema] Ningún proveedor respondió correctamente.\n"
        yield f"Asistente: {self.fallback_response}\n"
        if last_error is not None:
            yield f"[sistema] Último error capturado: {last_error.__class__.__name__}.\n"

    def _attempt_provider(self, provider) -> ProviderAttempt:
        """Ejecuta un proveedor y devuelve el texto completo generado."""
        chunks: list[str] = []
        yield_prefix_printed = False

        for chunk in provider.stream_response(
            system_prompt=self.system_prompt,
            conversation=self.conversation,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ):
            if not yield_prefix_printed:
                # Este print se realiza aquí para que el streaming sea real dentro
                # del intento. La CLI llama a stream_chat e imprime los fragmentos.
                yield_prefix_printed = True
            chunks.append(chunk)
            print(chunk, end="", flush=True)

        answer = "".join(chunks).strip()
        if not answer:
            raise RuntimeError("El proveedor no devolvió contenido de texto.")
        return ProviderAttempt(provider_name=provider.name, answer=answer)
