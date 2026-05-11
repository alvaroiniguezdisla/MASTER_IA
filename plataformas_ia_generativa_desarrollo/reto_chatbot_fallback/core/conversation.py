"""Gestión del historial de conversación en memoria."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Role = Literal["user", "assistant"]


@dataclass
class Message:
    """Mensaje simple de la conversación."""

    role: Role
    content: str


@dataclass
class Conversation:
    """
    Mantiene el historial completo usuario-asistente en memoria.

    No usa base de datos externa porque el requisito del ejercicio permite
    resolver la gestión de conversación con una clase o estructura Python.
    """

    messages: list[Message] = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        self._add_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        self._add_message("assistant", content)

    def _add_message(self, role: Role, content: str) -> None:
        cleaned_content = content.strip()
        if cleaned_content:
            self.messages.append(Message(role=role, content=cleaned_content))

    def to_openai_input(self) -> list[dict[str, str]]:
        """Formato compatible con la API Responses de OpenAI."""
        return [
            {"role": message.role, "content": message.content}
            for message in self.messages
        ]

    def to_anthropic_messages(self) -> list[dict[str, str]]:
        """Formato compatible con la API Messages de Anthropic."""
        return [
            {"role": message.role, "content": message.content}
            for message in self.messages
        ]

    def to_gemini_contents(self):
        """
        Formato compatible con google-genai.

        En Gemini el rol equivalente a assistant es model.
        La importación se hace dentro del método para no obligar a cargar
        google-genai si solo se está revisando el código o usando otro proveedor.
        """
        from google.genai import types

        contents = []
        for message in self.messages:
            gemini_role = "model" if message.role == "assistant" else "user"
            contents.append(
                types.Content(
                    role=gemini_role,
                    parts=[types.Part.from_text(text=message.content)],
                )
            )
        return contents

    def clear(self) -> None:
        """Vacía la conversación actual."""
        self.messages.clear()
