"""Proveedor Anthropic usando SDK oficial y streaming."""

from __future__ import annotations

import os
from collections.abc import Iterator

import anthropic


class AnthropicProvider:
    """Implementación del proveedor Anthropic Claude."""

    name = "Anthropic Claude"

    def __init__(self) -> None:
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
        self.client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else None

    def stream_response(
        self,
        *,
        system_prompt: str,
        conversation,
        temperature: float,
        max_tokens: int,
    ) -> Iterator[str]:
        if self.client is None:
            raise RuntimeError("No se encontró ANTHROPIC_API_KEY en el entorno.")

        with self.client.messages.stream(
            model=self.model,
            system=system_prompt,
            messages=conversation.to_anthropic_messages(),
            temperature=temperature,
            max_tokens=max_tokens,
        ) as stream:
            for text in stream.text_stream:
                if text:
                    yield text
