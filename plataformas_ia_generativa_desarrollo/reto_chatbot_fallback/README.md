# Reto proyecto: Chatbot con fallback automático

Proyecto en Python para construir un chatbot conversacional con fallback automático entre tres proveedores de IA:

1. OpenAI
2. Anthropic Claude
3. Google Gemini

El objetivo es mantener la continuidad del servicio: por defecto se intenta responder con OpenAI; si falla, se pasa a Anthropic; si también falla, se pasa a Google Gemini; y si ninguno funciona, se muestra una respuesta preconfigurada.

## Estructura

```text
reto_chatbot_fallback/
├── main.py
├── providers/
│   ├── __init__.py
│   ├── openai_provider.py
│   ├── anthropic_provider.py
│   └── gemini_provider.py
├── core/
│   ├── __init__.py
│   ├── chatbot.py
│   └── conversation.py
├── requirements.txt
├── .env
├── .env.example
├── .gitignore
└── README.md
```

## Requisitos cubiertos

### Implementación proveedor OpenAI

Archivo: `providers/openai_provider.py`

- Crea un cliente `OpenAI` con el SDK oficial `openai`.
- Usa la API `Responses` mediante `client.responses.create(...)`.
- Usa streaming con `stream=True`.
- Procesa eventos incrementales `response.output_text.delta`.

### Implementación proveedor Anthropic

Archivo: `providers/anthropic_provider.py`

- Crea un cliente `anthropic.Anthropic` con el SDK oficial.
- Usa `client.messages.stream(...)`.
- Recorre `stream.text_stream` para recibir la respuesta en streaming.

### Implementación proveedor Google

Archivo: `providers/gemini_provider.py`

- Usa el SDK oficial `google-genai`.
- Crea un cliente `genai.Client(...)`.
- Usa `client.models.generate_content_stream(...)`.
- Adapta el rol `assistant` al rol `model`, que es el equivalente en Gemini.

### Lógica de fallback

Archivo: `core/chatbot.py`

Orden implementado:

```text
OpenAI -> Anthropic Claude -> Google Gemini -> respuesta preconfigurada
```

El chatbot captura excepciones de los proveedores y muestra avisos en consola cuando cambia de proveedor.

### Gestión de conversación

Archivo: `core/conversation.py`

- La clase `Conversation` mantiene el historial completo en memoria.
- Guarda mensajes del usuario y del asistente.
- Convierte el historial al formato requerido por OpenAI, Anthropic y Gemini.
- No utiliza base de datos externa.

### Interfaz CLI

Archivo: `main.py`

- Bucle conversacional interactivo.
- El usuario escribe mensajes por terminal.
- La conversación termina limpiamente con `/salir`.
- Las respuestas se imprimen con streaming usando `print(..., end="", flush=True)`.

## Instalación

Desde la carpeta del proyecto:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

## Configuración de credenciales

Edita el archivo `.env` con tus claves reales en local:

```env
OPENAI_API_KEY="sk-..."
ANTHROPIC_API_KEY="sk-ant-..."
GEMINI_API_KEY="..."
```

No subas claves reales al repositorio. El archivo `.gitignore` incluye `.env` para evitar publicar credenciales por accidente.

## Ejecución

```bash
python main.py
```

Ejemplo de uso:

```text
Tú: Hola, ¿puedes resumirme qué es un fallback en un chatbot?
Asistente: [respuesta en streaming]

Tú: /salir
Saliendo del chatbot. ¡Hasta luego!
```

## Prueba del fallback

Para comprobar la cascada de fallback, puedes dejar vacía o poner mal una clave en `.env`:

- Si `OPENAI_API_KEY` falta o es incorrecta, debería intentar Anthropic.
- Si Anthropic también falla, debería intentar Gemini.
- Si los tres fallan, se devolverá `FALLBACK_RESPONSE`.

## Modelos por defecto

```env
OPENAI_MODEL="gpt-4o-mini"
ANTHROPIC_MODEL="claude-3-5-haiku-latest"
GEMINI_MODEL="gemini-2.5-flash"
```

Puedes cambiarlos en `.env` según los modelos disponibles en tus cuentas.

## Notas de diseño

- La lógica del chatbot está separada de la interfaz CLI.
- Cada proveedor tiene su propio archivo.
- El historial se mantiene en una clase propia para que sea fácil revisarlo y ampliarlo.
- Las claves de API se leen desde variables de entorno para no hardcodear credenciales.
