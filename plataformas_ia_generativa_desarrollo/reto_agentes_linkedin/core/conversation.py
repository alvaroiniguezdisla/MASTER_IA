"""Gestión sencilla del historial de conversación."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConversationHistory:
    """Guarda el historial de la sesión en memoria."""

    turns: list[dict[str, str]] = field(default_factory=list)

    def add_user_message(self, message: str) -> None:
        self.turns.append({"role": "user", "content": message})

    def add_assistant_message(self, message: str) -> None:
        self.turns.append({"role": "assistant", "content": message})

    def build_context_prompt(self, current_user_message: str) -> str:
        """Construye un prompt con historial resumido para mantener contexto."""
        if not self.turns:
            return current_user_message

        history_lines: list[str] = []
        for turn in self.turns[-8:]:
            role = "Usuario" if turn["role"] == "user" else "Asistente"
            history_lines.append(f"{role}: {turn['content']}")

        history_text = "\n".join(history_lines)
        return (
            "Historial reciente de la conversación:\n"
            f"{history_text}\n\n"
            "Nueva petición del usuario:\n"
            f"{current_user_message}"
        )

    def clear(self) -> None:
        self.turns.clear()
