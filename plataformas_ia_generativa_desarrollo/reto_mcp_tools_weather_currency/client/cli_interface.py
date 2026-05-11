"""Interfaz CLI para conversar con el cliente OpenAI + MCP."""

from __future__ import annotations

import logging

from client.openai_client import OpenAIMCPClient
from server.currency_tools import list_supported_currencies

logger = logging.getLogger(__name__)

HELP_TEXT = """
Comandos disponibles:
  /ayuda     Muestra esta ayuda.
  /monedas   Lista códigos de moneda comunes.
  /reset     Limpia el historial de conversación.
  /salir     Cierra el programa.

Ejemplos de consultas:
  Convierte 100 USD a EUR.
  ¿Cuál es el clima actual en Madrid?
  Dame el pronóstico del tiempo para Nueva York durante 5 días.
  ¿Qué coordenadas tiene Tokio?
  Dame las tasas de cambio de EUR a USD, GBP y JPY.
""".strip()


class CLIInterface:
    """Bucle de conversación por terminal."""

    def __init__(self) -> None:
        self.client = OpenAIMCPClient()

    def print_welcome(self) -> None:
        print("\nChatbot MCP: clima y conversión de monedas")
        print("Escribe /ayuda para ver ejemplos o /salir para terminar.\n")

        if self.client.check_server_health():
            print("Servidor MCP detectado correctamente.\n")
        else:
            print(
                "Aviso: no se detecta el servidor MCP en /health.\n"
                "Arranca primero: python main_server.py\n"
                "Si usas OpenAI Responses con MCP remoto, asegúrate de que MCP_SERVER_URL sea accesible.\n"
            )

    def run(self) -> None:
        self.print_welcome()

        while True:
            try:
                user_input = input("Tú > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nSaliendo...")
                break

            if not user_input:
                continue

            command = user_input.lower()
            if command in {"/salir", "quit", "exit"}:
                print("Hasta luego.")
                break
            if command == "/ayuda":
                print(f"\n{HELP_TEXT}\n")
                continue
            if command == "/monedas":
                print("\nCódigos habituales:")
                print(", ".join(list_supported_currencies()))
                print()
                continue
            if command == "/reset":
                self.client.reset_history()
                print("Historial reiniciado.\n")
                continue

            print("\nAsistente > ", end="", flush=True)
            answer = self.client.ask(user_input)
            print(answer)
            print()
