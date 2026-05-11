# Chatbot multiagente para generar posts de LinkedIn

Proyecto de terminal en Python usando **OpenAI Agents SDK**. El sistema tiene un agente principal que recibe la solicitud del usuario y delega en agentes especializados para generar publicaciones de LinkedIn.

## Estructura

```text
├── main.py
├── agents/
│   ├── __init__.py
│   ├── main_agent.py
│   └── specialized_agents.py
├── core/
│   ├── __init__.py
│   ├── chatbot.py
│   └── conversation.py
├── requirements.txt
├── .env
└── README.md
```

## Agentes incluidos

- **Agente Principal**: recibe la petición y delega según la temática.
- **Agente Marketing**: posts sobre marketing, ventas, marca personal y redes sociales.
- **Agente Programacion**: posts sobre software, IA, APIs, datos y tecnología.
- **Agente Juridico Legal**: posts jurídicos, compliance, protección de datos y regulación.

Todas las publicaciones se devuelven con salida estructurada Pydantic:

```python
class LinkedinPost(BaseModel):
    title: str
    content: str
    hashtags: list[str]
    category: Literal["marketing", "programacion", "juridico_legal"]
```

## Instalación

1. Crear entorno virtual:

```bash
python -m venv .venv
```

2. Activar entorno virtual:

En Windows:

```bash
.venv\Scripts\activate
```

En macOS/Linux:

```bash
source .venv/bin/activate
```

3. Instalar dependencias:

```bash
pip install -r requirements.txt
```

4. Configurar la API key en `.env`:

```env
OPENAI_API_KEY=tu_api_key_aqui
```

## Ejecución

```bash
python main.py
```

## Comandos disponibles

```text
/ayuda   Muestra ejemplos y comandos.
/salir   Finaliza el programa.
```

## Ejemplos de uso

```text
Crea un post de LinkedIn sobre marca personal para perfiles junior.
```

```text
Genera una publicación sobre buenas prácticas con APIs en Python.
```

```text
Haz un post sobre protección de datos para pequeñas empresas.
```

## Funcionamiento resumido

1. El usuario escribe una idea para un post.
2. El agente principal analiza la temática.
3. El agente principal delega al especialista correspondiente mediante handoff.
4. El agente especializado genera una salida validada con Pydantic.
5. El chatbot muestra título, contenido, hashtags y categoría.

## Nota técnica sobre la carpeta `agents/`

El ejercicio exige una carpeta local llamada `agents/`, pero el OpenAI Agents SDK también se importa normalmente como `agents`. Para evitar el conflicto de nombres, el archivo `agents/__init__.py` carga el SDK oficial desde `site-packages` con un alias interno y reexporta `Agent`, `Runner` y otras clases necesarias.
