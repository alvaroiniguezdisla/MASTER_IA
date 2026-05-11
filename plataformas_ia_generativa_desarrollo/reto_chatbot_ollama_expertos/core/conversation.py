"""Gestión del historial de conversación del chatbot."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from experts.expert_prompts import DEFAULT_EXPERT, EXPERTS, get_system_prompt

Message = Dict[str, str]


@dataclass
class ConversationManager:
    """Mantiene el historial de conversación separado por experto.

    Se usa un historial independiente para cada experto temático. Así, el usuario
    puede cambiar entre Programación, Marketing y Jurídico-Legal sin perder el
    contexto anterior de cada especialista. También se permite reiniciar el
    historial del experto activo o de todos los expertos.
    """

    active_expert: str = DEFAULT_EXPERT
    histories: Dict[str, List[Message]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for expert_key in EXPERTS:
            self.histories.setdefault(expert_key, [])

    def set_active_expert(self, expert_key: str, reset_history: bool = False) -> None:
        """Cambia el experto activo y opcionalmente reinicia su historial."""
        if expert_key not in EXPERTS:
            raise ValueError(f"Experto no válido: {expert_key}")

        self.active_expert = expert_key
        self.histories.setdefault(expert_key, [])

        if reset_history:
            self.reset_current_history()

    def add_user_message(self, content: str) -> None:
        """Añade un mensaje del usuario al historial del experto activo."""
        self.histories[self.active_expert].append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Añade un mensaje del asistente al historial del experto activo."""
        self.histories[self.active_expert].append({"role": "assistant", "content": content})

    def get_current_history(self) -> List[Message]:
        """Devuelve una copia del historial del experto activo."""
        return list(self.histories[self.active_expert])

    def get_messages_for_model(self) -> List[Message]:
        """Construye los mensajes que se enviarán a Ollama.

        El prompt de sistema se incluye en cada llamada para garantizar que el
        modelo mantenga el comportamiento del experto activo.
        """
        return [
            {"role": "system", "content": get_system_prompt(self.active_expert)},
            *self.get_current_history(),
        ]

    def reset_current_history(self) -> None:
        """Reinicia solo el historial del experto activo."""
        self.histories[self.active_expert] = []

    def reset_all_histories(self) -> None:
        """Reinicia el historial de todos los expertos."""
        for expert_key in self.histories:
            self.histories[expert_key] = []

    def current_turns_count(self) -> int:
        """Devuelve el número de mensajes del historial activo."""
        return len(self.histories[self.active_expert])
