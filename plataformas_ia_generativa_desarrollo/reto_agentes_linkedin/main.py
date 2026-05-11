"""Punto de entrada del chatbot multiagente."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from core.chatbot import LinkedInAgentsChatbot


def print_help() -> None:
    print("""
Comandos disponibles:
  /ayuda   Muestra esta ayuda.
  /salir   Termina el programa.

Ejemplos de peticiones:
  - Crea un post de LinkedIn sobre marca personal para perfiles junior.
  - Genera una publicación sobre buenas prácticas con APIs en Python.
  - Haz un post sobre protección de datos para pequeñas empresas.
""".strip())


def main() -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("Aviso: no se ha encontrado OPENAI_API_KEY en el entorno o en el archivo .env.")
        print("Crea un archivo .env con: OPENAI_API_KEY=tu_clave_aqui\n")

    chatbot = LinkedInAgentsChatbot()

    print("=== Chatbot multiagente para posts de LinkedIn ===")
    print("Escribe una idea y el agente principal delegará al especialista adecuado.")
    print("Comandos: /ayuda, /salir\n")

    while True:
        try:
            user_input = input("Tú > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSaliendo...")
            break

        if not user_input:
            continue

        if user_input.lower() in {"/salir", "salir", "quit", "exit"}:
            print("Hasta luego.")
            break

        if user_input.lower() == "/ayuda":
            print_help()
            continue

        print("\n[Agente Principal] Analizando la solicitud y delegando...\n")

        try:
            post, agent_name = chatbot.generate_post(user_input)
        except RuntimeError as exc:
            print(f"Error: {exc}\n")
            continue

        print(f"[{agent_name}] Publicación generada:\n")
        print(chatbot.format_post(post))
        print("\n" + "-" * 70 + "\n")


if __name__ == "__main__":
    main()
