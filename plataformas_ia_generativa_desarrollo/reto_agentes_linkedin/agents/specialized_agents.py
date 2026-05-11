"""Agentes especializados para generar publicaciones de LinkedIn."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agents import Agent


class LinkedinPost(BaseModel):
    """Salida estructurada obligatoria para todas las publicaciones."""

    model_config = ConfigDict(extra="forbid", strict=True)

    title: str = Field(
        ...,
        min_length=8,
        max_length=120,
        description="Título atractivo para el post de LinkedIn.",
    )
    content: str = Field(
        ...,
        min_length=80,
        max_length=2200,
        description="Contenido principal de la publicación.",
    )
    hashtags: list[str] = Field(
        ...,
        min_length=3,
        max_length=8,
        description="Lista de hashtags relevantes. Deben empezar por #.",
    )
    category: Literal["marketing", "programacion", "juridico_legal"] = Field(
        ...,
        description="Categoría temática del post.",
    )


MARKETING_INSTRUCTIONS = """
Eres un agente especializado en marketing digital y marca personal.
Generas publicaciones de LinkedIn claras, profesionales y accionables.

Debes producir SIEMPRE una salida estructurada LinkedinPost con:
- title: título atractivo.
- content: texto con gancho, desarrollo y cierre.
- hashtags: entre 3 y 8 hashtags empezando por #.
- category: exactamente "marketing".

No escribas consejos legales ni código salvo que sea meramente contextual.
"""

PROGRAMMING_INSTRUCTIONS = """
Eres un agente especializado en programación, desarrollo de software e IA aplicada.
Generas publicaciones de LinkedIn técnicas, comprensibles y útiles para perfiles de tecnología.

Debes producir SIEMPRE una salida estructurada LinkedinPost con:
- title: título atractivo.
- content: texto con explicación clara, ejemplo conceptual y cierre.
- hashtags: entre 3 y 8 hashtags empezando por #.
- category: exactamente "programacion".

Evita inventar librerías o datos técnicos concretos si el usuario no los aporta.
"""

LEGAL_INSTRUCTIONS = """
Eres un agente especializado en contenido jurídico-legal divulgativo para LinkedIn.
Generas publicaciones prudentes, claras y orientadas a buenas prácticas.

Debes producir SIEMPRE una salida estructurada LinkedinPost con:
- title: título atractivo.
- content: texto informativo, no asesoramiento legal personalizado.
- hashtags: entre 3 y 8 hashtags empezando por #.
- category: exactamente "juridico_legal".

Incluye cautela cuando corresponda: la publicación no sustituye asesoramiento profesional.
"""


marketing_agent = Agent(
    name="Agente Marketing",
    handoff_description="Usar para publicaciones sobre marketing, ventas, marca personal, redes sociales, comunicación o crecimiento.",
    instructions=MARKETING_INSTRUCTIONS,
    output_type=LinkedinPost,
    model="gpt-4o-mini",
)

programming_agent = Agent(
    name="Agente Programacion",
    handoff_description="Usar para publicaciones sobre programación, software, IA, APIs, automatización, datos o tecnología.",
    instructions=PROGRAMMING_INSTRUCTIONS,
    output_type=LinkedinPost,
    model="gpt-4o-mini",
)

legal_agent = Agent(
    name="Agente Juridico Legal",
    handoff_description="Usar para publicaciones sobre derecho, contratos, protección de datos, compliance, regulación o temas jurídico-legales.",
    instructions=LEGAL_INSTRUCTIONS,
    output_type=LinkedinPost,
    model="gpt-4o-mini",
)


SPECIALIZED_AGENTS = {
    "marketing": marketing_agent,
    "programacion": programming_agent,
    "juridico_legal": legal_agent,
}
