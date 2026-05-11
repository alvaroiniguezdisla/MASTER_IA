# Reto proyecto: Chatbot especializado con Ollama SDK en Python

Este proyecto implementa un chatbot de consola con tres expertos temáticos usando el SDK oficial de Ollama para Python y el modelo local `gemma3:1b`.

El sistema funciona de forma local y offline una vez que Ollama y el modelo están instalados en la máquina.

## Estructura del proyecto

```text
├── main.py
├── experts/
│   ├── __init__.py
│   └── expert_prompts.py
├── core/
│   ├── __init__.py
│   ├── chatbot.py
│   └── conversation.py
├── requirements.txt
└── README.md
```

## Expertos implementados

El chatbot permite conversar con tres especialistas diferenciados:

1. **Programación de software**: desarrollo, arquitectura, depuración, buenas prácticas, APIs, bases de datos y testing.
2. **Marketing**: estrategia comercial, branding, campañas, segmentación, análisis de mercado y copywriting.
3. **Jurídico-legal**: contratos, normativas, riesgos legales, protección de datos, propiedad intelectual y cumplimiento.

Cada experto tiene un prompt de sistema propio en `experts/expert_prompts.py`, lo que modifica el comportamiento, el tono y el enfoque de las respuestas del modelo.

## Requisitos previos

Tener instalado Python 3.10 o superior.

Tener instalado Ollama en local.

Descargar el modelo requerido:

```bash
ollama pull gemma3:1b
```

Comprobar que Ollama está funcionando:

```bash
ollama serve
```

En algunos sistemas Ollama queda iniciado automáticamente, por lo que este comando puede no ser necesario.

## Instalación

Crear y activar un entorno virtual:

```bash
python -m venv .venv
```

En Windows:

```bash
.venv\Scripts\activate
```

En macOS o Linux:

```bash
source .venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución

Desde la carpeta raíz del proyecto:

```bash
python main.py
```

Al iniciar, el programa comprueba:

- Que el servicio local de Ollama está disponible.
- Que el modelo `gemma3:1b` está descargado localmente.

Después permite seleccionar uno de los tres expertos y comenzar la conversación.

## Comandos disponibles en la consola

Durante la conversación se pueden usar estos comandos:

```text
/experto          Cambiar de experto temático
/reiniciar        Reiniciar el historial del experto activo
/reiniciar todo   Reiniciar el historial de todos los expertos
/estado           Mostrar modelo, experto activo y mensajes del historial
/ayuda            Mostrar la ayuda
/salir            Finalizar la aplicación
```

## Gestión del historial

La clase `ConversationManager`, en `core/conversation.py`, mantiene un historial independiente para cada experto.

Esto permite:

- Mantener el contexto entre varios mensajes con el mismo experto.
- Cambiar de experto durante la conversación.
- Conservar el historial anterior de cada experto.
- Reiniciar solo el historial del experto activo.
- Reiniciar todos los historiales.

## Integración con Ollama SDK

La clase `ExpertChatbot`, en `core/chatbot.py`, usa directamente la librería `ollama` de Python:

```python
ollama.chat(
    model="gemma3:1b",
    messages=messages,
    stream=True,
)
```

La respuesta se genera en streaming para que aparezca progresivamente en la consola.

## Manejo de errores

El proyecto gestiona errores habituales:

- Ollama no está iniciado.
- El modelo `gemma3:1b` no está descargado.
- Fallos durante la generación de la respuesta.

Cuando ocurre un error, la interfaz muestra un mensaje claro con instrucciones para resolverlo.

## Funcionamiento offline

El chatbot no usa APIs externas ni claves privadas. Todas las respuestas se generan mediante el servicio local de Ollama y el modelo descargado en la máquina.

Una vez descargado el modelo, no es necesaria conexión a internet para usar la aplicación.
