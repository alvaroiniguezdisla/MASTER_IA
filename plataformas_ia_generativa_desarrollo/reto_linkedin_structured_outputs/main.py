"""Punto de entrada de la aplicación."""

from core.api_client import (
    ChatbotError,
    ConfigurationError,
    ModelRefusalError,
    ResponseValidationError,
)
from core.chatbot import LinkedinPostChatbot


EXIT_COMMANDS = {"/salir", "salir", "exit", "quit"}


def print_header() -> None:
    print("=" * 70)
    print("CHATBOT GENERADOR DE POSTS PARA LINKEDIN")
    print("OpenAI Responses API + Structured Outputs + Pydantic")
    print("=" * 70)
    print("Comandos disponibles:")
    print("  /salir  -> finalizar la aplicación")
    print("Escribe una idea y se generará un post estructurado.\n")


def main() -> None:
    print_header()

    try:
        chatbot = LinkedinPostChatbot()
    except ConfigurationError as error:
        print(f"Error de configuración: {error}")
        print("Crea un archivo .env con OPENAI_API_KEY=tu_clave_aqui")
        return

    while True:
        user_input = input("Describe la idea del post: ").strip()

        if user_input.lower() in EXIT_COMMANDS:
            print("Aplicación finalizada correctamente.")
            break

        if not user_input:
            print("Introduce una idea antes de generar el post.\n")
            continue

        print("\nGenerando post estructurado, espera unos segundos...\n")

        try:
            post = chatbot.generate_post(user_input)
            print(chatbot.format_post(post))
        except ModelRefusalError as error:
            print(f"La API rechazó la solicitud: {error}\n")
        except ResponseValidationError as error:
            print(f"Error de validación de Pydantic: {error}\n")
        except ValueError as error:
            print(f"Entrada no válida: {error}\n")
        except ChatbotError as error:
            print(f"No se pudo generar el post: {error}\n")
        except KeyboardInterrupt:
            print("\nAplicación interrumpida por el usuario.")
            break


if __name__ == "__main__":
    main()
