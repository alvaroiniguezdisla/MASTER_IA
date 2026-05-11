"""Punto de entrada CLI del chatbot con fallback."""

from __future__ import annotations

from dotenv import load_dotenv

from core.chatbot import FallbackChatbot


def main() -> None:
    """Ejecuta la interfaz conversacional por consola."""
    load_dotenv()
    chatbot = FallbackChatbot()

    print("=" * 70)
    print("Chatbot con fallback automático: OpenAI -> Anthropic -> Google Gemini")
    print("Escribe /salir para terminar la conversación.")
    print("=" * 70)

    while True:
        try:
            user_message = input("\nTú: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nSaliendo del chatbot. ¡Hasta luego!")
            break

        if not user_message:
            print("Escribe un mensaje o /salir para terminar.")
            continue

        if user_message.lower() == "/salir":
            print("Saliendo del chatbot. ¡Hasta luego!")
            break

        print("\nAsistente: ", end="", flush=True)
        for output in chatbot.stream_chat(user_message):
            # Los chunks de texto del proveedor se imprimen dentro de _attempt_provider
            # para que aparezcan inmediatamente según llegan. Aquí se imprimen avisos.
            if output.startswith("\n[sistema]") or output.startswith("\n\n[sistema]") or output.startswith("Asistente:") or output.startswith("[sistema]"):
                print(output, end="", flush=True)


if __name__ == "__main__":
    main()
