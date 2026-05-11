"""Proveedor OpenAI usando SDK oficial y API Responses en streaming."""

from __future__ import annotations

import os
from collections.abc import Iterator

from openai import OpenAI


class OpenAIProvider:
    """Implementación del proveedor OpenAI."""

    name = "OpenAI"

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def stream_response(
        self,
        *,
        system_prompt: str,
        conversation,
        temperature: float,
        max_tokens: int,
    ) -> Iterator[str]:
        if self.client is None:
            raise RuntimeError("No se encontró OPENAI_API_KEY en el entorno.")

        stream = self.client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=conversation.to_openai_input(),
            stream=True,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        for event in stream:
            # Evento principal de texto incremental en la Responses API.
            if getattr(event, "type", None) == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    yield delta
