"""Proveedor Google Gemini usando google-genai y streaming."""

from __future__ import annotations

import os
from collections.abc import Iterator

from google import genai
from google.genai import types


class GeminiProvider:
    """Implementación del proveedor Google Gemini."""

    name = "Google Gemini"

    def __init__(self) -> None:
        # Se aceptan ambos nombres para facilitar la configuración local.
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None

    def stream_response(
        self,
        *,
        system_prompt: str,
        conversation,
        temperature: float,
        max_tokens: int,
    ) -> Iterator[str]:
        if self.client is None:
            raise RuntimeError("No se encontró GEMINI_API_KEY ni GOOGLE_API_KEY en el entorno.")

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        stream = self.client.models.generate_content_stream(
            model=self.model,
            contents=conversation.to_gemini_contents(),
            config=config,
        )

        for chunk in stream:
            text = getattr(chunk, "text", None)
            if text:
                yield text
