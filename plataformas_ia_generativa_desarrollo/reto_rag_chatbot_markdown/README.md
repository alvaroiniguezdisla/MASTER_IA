# Reto proyecto: Chatbot inteligente con RAG

Este proyecto implementa un chatbot por terminal que usa la técnica **RAG** (*Retrieval-Augmented Generation*) con **LangChain**, **OpenAIEmbeddings** y **OpenAI** para responder preguntas basándose únicamente en documentos Markdown de una empresa ficticia.

## Estructura del proyecto

```text
├── main.py
├── documents/
│   ├── documento1.md
│   └── documento2.md
├── core/
│   ├── __init__.py
│   ├── rag_system.py
│   └── chatbot.py
├── requirements.txt
├── .env
└── README.md
```

## Qué incluye

- Embeddings con `OpenAIEmbeddings` usando el modelo `text-embedding-3-small`.
- Vector store en memoria con `InMemoryVectorStore` de LangChain.
- Procesamiento de documentos Markdown ficticios de empresa.
- Troceado de documentos mediante `RecursiveCharacterTextSplitter`.
- Retrieval por similitud semántica.
- Chatbot conversacional con historial durante la sesión.
- Modelo de OpenAI configurable mediante `.env`, por defecto `gpt-4.1`.
- Interfaz CLI con comandos `/salir` y `/reset`.
- Manejo básico de errores de conexión, autenticación, límites de uso y problemas de API.

## Requisitos previos

- Python 3.10 o superior.
- Una clave de API de OpenAI.

## Instalación

Crea y activa un entorno virtual:

```bash
python -m venv .venv
```

En Windows:

```bash
.venv\Scripts\activate
```

En macOS/Linux:

```bash
source .venv/bin/activate
```

Instala las dependencias:

```bash
pip install -r requirements.txt
```

## Configuración

Abre el archivo `.env` y sustituye el valor de ejemplo por tu clave real:

```env
OPENAI_API_KEY=tu_api_key_real
OPENAI_MODEL=gpt-4.1
```

También puedes usar otro modelo compatible, por ejemplo:

```env
OPENAI_MODEL=gpt-4o
```

## Ejecución

Desde la raíz del proyecto ejecuta:

```bash
python main.py
```

Al iniciar, el sistema cargará los documentos Markdown, los dividirá en fragmentos, generará embeddings y almacenará los vectores en `InMemoryVectorStore`.

## Ejemplos de preguntas

Puedes probar consultas como:

```text
¿Cuál es la misión de NovaTech Solutions?
```

```text
¿Qué beneficios de formación ofrece la empresa?
```

```text
¿Cuál es el procedimiento para proyectos de IA?
```

```text
¿Cuál es el horario de trabajo y la política de teletrabajo?
```

## Comandos disponibles

- `/salir`: termina la conversación.
- `/reset`: borra el historial conversacional de la sesión actual.

## Funcionamiento RAG

El flujo principal es:

1. El usuario introduce una pregunta por terminal.
2. El sistema convierte la pregunta en embedding.
3. `InMemoryVectorStore` recupera los fragmentos Markdown más similares.
4. El chatbot construye un prompt aumentado con el contexto recuperado.
5. El modelo de OpenAI genera una respuesta usando únicamente ese contexto.
6. La respuesta se muestra por terminal y se guarda en el historial de la sesión.

## Notas de seguridad

No subas claves reales a repositorios públicos. El archivo `.gitignore` incluye `.env` para evitar publicar credenciales por accidente.
