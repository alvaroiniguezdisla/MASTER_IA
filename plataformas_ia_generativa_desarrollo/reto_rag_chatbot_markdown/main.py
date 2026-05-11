from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from core.chatbot import Chatbot
from core.rag_system import RAGSystem


def print_header() -> None:
    print("=" * 72)
    print("Chatbot RAG con LangChain + OpenAI + documentos Markdown")
    print("=" * 72)
    print("Comandos disponibles:")
    print("  /salir  -> terminar la conversación")
    print("  /reset  -> borrar el historial de la sesión")
    print("-" * 72)


def main() -> None:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "tu_api_key_aqui":
        print("Falta configurar OPENAI_API_KEY en el archivo .env.")
        print("Abre .env y sustituye 'tu_api_key_aqui' por tu clave real.")
        return

    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1")
    documents_dir = Path(__file__).resolve().parent / "documents"

    print_header()
    print("Cargando documentos y creando embeddings...")

    try:
        rag_system = RAGSystem(documents_dir=documents_dir)
        total_chunks = rag_system.ingest_documents()
        chatbot = Chatbot(rag_system=rag_system, model_name=model_name)
    except Exception as exc:
        print(f"No se pudo inicializar el sistema RAG: {exc}")
        return

    print(f"Sistema listo. Fragmentos indexados: {total_chunks}")
    print(f"Modelo de chat configurado: {model_name}")
    print("Puedes preguntar sobre la empresa ficticia, sus servicios o políticas.")
    print("-" * 72)

    while True:
        try:
            user_query = input("\nTú: ").strip()
        except KeyboardInterrupt:
            print("\nSesión finalizada.")
            break

        if not user_query:
            continue

        if user_query.lower() in {"/salir", "salir", "quit", "exit"}:
            print("Chatbot: Hasta luego.")
            break

        if user_query.lower() == "/reset":
            chatbot.reset()
            print("Chatbot: Historial de conversación borrado.")
            continue

        print("\nChatbot:")
        answer = chatbot.answer(user_query)
        print(answer)


if __name__ == "__main__":
    main()
