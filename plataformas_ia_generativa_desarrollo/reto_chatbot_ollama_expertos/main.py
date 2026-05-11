"""Punto de entrada del chatbot de expertos con Ollama."""

from __future__ import annotations

import sys

from core.chatbot import (
    ExpertChatbot,
    OllamaChatbotError,
    OllamaConnectionError,
    OllamaModelNotAvailableError,
)
from experts.expert_prompts import EXPERTS, get_expert_keys

MODEL_NAME = "gemma3:1b"


def print_banner() -> None:
    print("=" * 72)
    print("CHATBOT DE EXPERTOS TEMÁTICOS CON OLLAMA")
    print("Modelo local offline:", MODEL_NAME)
    print("=" * 72)
    print()


def print_help() -> None:
    print("Opciones disponibles:")
    print("  /experto      Cambiar de experto temático")
    print("  /reiniciar    Reiniciar el historial del experto activo")
    print("  /reiniciar todo  Reiniciar el historial de todos los expertos")
    print("  /estado       Mostrar modelo, experto activo y mensajes del historial")
    print("  /ayuda        Mostrar esta ayuda")
    print("  /salir        Finalizar la aplicación")
    print()


def print_experts() -> None:
    print("Expertos disponibles:")
    for index, key in enumerate(get_expert_keys(), start=1):
        expert = EXPERTS[key]
        print(f"  {index}. {expert['name']} - {expert['description']}")
    print()


def select_expert(chatbot: ExpertChatbot, initial: bool = False) -> None:
    """Permite seleccionar experto desde la consola."""
    keys = get_expert_keys()

    while True:
        print_experts()
        option = input("Selecciona un experto por número: ").strip()

        if not option.isdigit():
            print("Introduce un número válido.\n")
            continue

        index = int(option) - 1
        if index < 0 or index >= len(keys):
            print("Opción fuera de rango.\n")
            continue

        selected_key = keys[index]
        reset_history = False

        if not initial:
            answer = input(
                "¿Quieres reiniciar el historial de este experto? [s/N]: "
            ).strip().lower()
            reset_history = answer in {"s", "si", "sí", "y", "yes"}

        chatbot.change_expert(selected_key, reset_history=reset_history)
        print(f"Experto activo: {chatbot.active_expert_name}\n")
        break


def check_system(chatbot: ExpertChatbot) -> bool:
    """Verifica conexión local con Ollama y disponibilidad del modelo."""
    print("Comprobando servicio local de Ollama y modelo descargado...")
    try:
        chatbot.ensure_ollama_ready()
    except OllamaConnectionError as exc:
        print("ERROR DE CONEXIÓN")
        print(exc)
        print("\nPasos sugeridos:")
        print("  1. Instala Ollama si no lo tienes instalado.")
        print("  2. Inicia el servicio local con: ollama serve")
        print("  3. Vuelve a ejecutar: python main.py")
        return False
    except OllamaModelNotAvailableError as exc:
        print("MODELO NO DISPONIBLE")
        print(exc)
        print("\nDescarga el modelo con:")
        print(f"  ollama pull {MODEL_NAME}")
        return False

    print("Sistema listo. Funcionando con modelo local offline.\n")
    return True


def chat_loop(chatbot: ExpertChatbot) -> None:
    """Bucle principal de conversación."""
    print_help()

    while True:
        prompt = f"[{chatbot.active_expert_name}] Tú > "
        user_input = input(prompt).strip()

        if not user_input:
            continue

        command = user_input.lower()

        if command == "/salir":
            print("Saliendo del chatbot. Hasta pronto.")
            break

        if command == "/ayuda":
            print_help()
            continue

        if command == "/estado":
            print(chatbot.get_status_text())
            print()
            continue

        if command == "/experto":
            select_expert(chatbot, initial=False)
            print_help()
            continue

        if command == "/reiniciar":
            chatbot.reset_current_history()
            print(f"Historial reiniciado para el experto: {chatbot.active_expert_name}\n")
            continue

        if command == "/reiniciar todo":
            chatbot.reset_all_histories()
            print("Historial reiniciado para todos los expertos.\n")
            continue

        print(f"[{chatbot.active_expert_name}] Asistente > ", end="", flush=True)
        try:
            for text_piece in chatbot.stream_answer(user_input):
                print(text_piece, end="", flush=True)
            print("\n")
        except OllamaChatbotError as exc:
            print("\n")
            print("No se pudo completar la respuesta.")
            print(exc)
            print("Puedes probar de nuevo, cambiar de experto con /experto o salir con /salir.\n")


def main() -> int:
    print_banner()
    chatbot = ExpertChatbot(model=MODEL_NAME)

    if not check_system(chatbot):
        return 1

    select_expert(chatbot, initial=True)
    chat_loop(chatbot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
