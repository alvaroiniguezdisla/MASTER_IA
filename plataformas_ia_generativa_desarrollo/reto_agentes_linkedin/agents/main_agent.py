"""Agente principal coordinador."""

from __future__ import annotations

from agents import Agent

from agents.specialized_agents import (
    LinkedinPost,
    legal_agent,
    marketing_agent,
    programming_agent,
)

MAIN_AGENT_INSTRUCTIONS = """
Eres el agente principal de un sistema multiagente para generar publicaciones de LinkedIn.

Tu trabajo es recibir la solicitud del usuario y DELEGARLA al agente especializado correcto:
- Marketing: marketing, ventas, marca personal, redes sociales, comunicación, crecimiento.
- Programación: software, desarrollo, IA, APIs, automatización, datos, tecnología.
- Jurídico-legal: derecho, contratos, protección de datos, compliance, regulación.

Si la temática no está perfectamente clara, elige la categoría más cercana según el objetivo principal.
No generes tú el post si hay un agente especializado aplicable; realiza handoff.
La salida final del sistema debe ser un objeto LinkedinPost validado.
"""

main_agent = Agent(
    name="Agente Principal",
    instructions=MAIN_AGENT_INSTRUCTIONS,
    handoffs=[marketing_agent, programming_agent, legal_agent],
    output_type=LinkedinPost,
    model="gpt-4o-mini",
)
