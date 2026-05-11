# Reto proyecto: chatbot generador de posts de LinkedIn

Este proyecto implementa un chatbot por terminal que genera posts de LinkedIn usando la API de OpenAI con **Structured Outputs** y un modelo **Pydantic** llamado `LinkedinPost`.

## Estructura del proyecto

```text
├── main.py
├── models/
│   ├── __init__.py
│   └── linkedin_post.py
├── core/
│   ├── __init__.py
│   ├── chatbot.py
│   └── api_client.py
├── requirements.txt
├── .env
└── README.md
```

## Requisitos implementados

- Modelo Pydantic `LinkedinPost` con campos obligatorios:
  - `title: str`
  - `content: str`
  - `hashtags: list[str]`
  - `category: str`
- Validación estricta con `extra="forbid"` para evitar propiedades adicionales.
- Uso de `client.responses.parse()` del SDK de OpenAI.
- Uso de `text_format=LinkedinPost` para activar Structured Outputs.
- Interfaz por terminal para introducir la idea del post.
- Visualización estructurada del resultado.
- Manejo de errores de configuración, conexión, límites de API, rechazos del modelo y validación Pydantic.

## Instalación

Desde la carpeta del proyecto:

```bash
python -m venv .venv
```

Activar entorno virtual:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Configuración

Edita el archivo `.env` y añade tu clave privada:

```env
OPENAI_API_KEY=tu_clave_aqui
OPENAI_MODEL=gpt-4o-2024-08-06
```

No subas tu clave real a GitHub. El archivo `.gitignore` está preparado para ignorar `.env`.

## Ejecución

```bash
python main.py
```

La aplicación pedirá una idea para el post. Por ejemplo:

```text
Describe la idea del post: La importancia de validar los datos que devuelve un LLM en aplicaciones reales
```

Para salir:

```text
/salir
```

## Ejemplo de salida esperada

```text
POST DE LINKEDIN GENERADO
======================================================================

Título:
Validar la salida de un LLM no es opcional

Contenido:
En una demo, aceptar texto libre puede funcionar. En una aplicación real, no...

Hashtags:
#IA #Pydantic #OpenAI #StructuredOutputs

Categoría:
Inteligencia Artificial
======================================================================
```

## Notas

El proyecto usa por defecto `gpt-4o-2024-08-06`, un modelo compatible con Structured Outputs. También puede configurarse otro modelo compatible mediante la variable `OPENAI_MODEL`.
