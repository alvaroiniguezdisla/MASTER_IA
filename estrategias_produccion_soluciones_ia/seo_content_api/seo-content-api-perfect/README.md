# SEO Content API

API REST desarrollada con **FastAPI**, **Pydantic** y **Azure OpenAI SDK** para generar contenido SEO mediante inteligencia artificial.

El proyecto implementa los cinco endpoints pedidos en el enunciado:

- `POST /api/keywords/generate`
- `POST /api/articles/generate`
- `POST /api/metadata/generate`
- `POST /api/faqs/extract`
- `POST /api/social/summaries`

## 1. Estructura del proyecto

```text
seo-content-api-perfect/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── dependencies.py
│   ├── models/
│   ├── routers/
│   └── services/
├── requirements.txt
├── .env.example
├── .env
├── .gitignore
└── README.md
```

El ZIP contiene una única carpeta raíz llamada `seo-content-api-perfect`, tal y como pide la plataforma.

## 2. Instalación

Se recomienda usar Python 3.12 o superior.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

En Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Configuración de Azure OpenAI

Copia `.env.example` a `.env` y configura tus credenciales:

```env
AZURE_OPENAI_API_KEY=tu_api_key_de_azure_openai
AZURE_OPENAI_ENDPOINT=https://tu-recurso.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
REQUEST_TIMEOUT_SECONDS=60
```

> Importante: en Azure OpenAI, el valor de `AZURE_OPENAI_DEPLOYMENT` debe ser el nombre del deployment creado en Azure, por ejemplo `gpt-4o-mini`.

## 4. Ejecución

```bash
uvicorn app.main:app --reload
```

Abrir Swagger UI:

```text
http://localhost:8000/docs
```

## 5. Endpoints

### POST /api/keywords/generate

Genera keywords semilla, long-tail, preguntas e intención de búsqueda.

Ejemplo:

```json
{
  "topic": "software de gestión clínica",
  "industry": "salud digital",
  "language": "es"
}
```

### POST /api/articles/generate

Genera un artículo SEO completo con estructura H1/H2/H3, densidad natural de keywords y CTAs.

Ejemplo:

```json
{
  "main_keyword": "software de gestión clínica",
  "secondary_keywords": ["historia clínica electrónica", "gestión hospitalaria"],
  "word_count": 900,
  "tone": "profesional e informativo"
}
```

### POST /api/metadata/generate

Genera 3-5 meta titles y 3-5 meta descriptions optimizados para CTR.

Ejemplo:

```json
{
  "article_title": "Software de gestión clínica para hospitales",
  "main_keyword": "software de gestión clínica",
  "article_excerpt": "Guía sobre cómo elegir una solución de gestión clínica para mejorar procesos hospitalarios."
}
```

Restricciones:
- Meta title: máximo 60 caracteres.
- Meta description: máximo 160 caracteres.

### POST /api/faqs/extract

Extrae FAQs del contenido de un artículo y genera JSON-LD FAQPage válido.

Ejemplo:

```json
{
  "article_content": "Contenido largo del artículo...",
  "max_questions": 5
}
```

### POST /api/social/summaries

Genera contenido adaptado por plataforma.

Ejemplo:

```json
{
  "article_title": "Software de gestión clínica para hospitales",
  "article_content": "Contenido del artículo...",
  "target_platforms": ["twitter", "linkedin", "instagram", "facebook"]
}
```

## 6. Buenas prácticas incluidas

- Arquitectura modular con `models`, `routers` y `services`.
- Pydantic para validación estricta de entrada y salida.
- `extra="forbid"` para evitar campos no esperados.
- Inyección de dependencias con `Depends`.
- Cliente común de Azure OpenAI.
- Manejo de errores de conexión, timeout, rate limit y errores HTTP.
- Salidas estructuradas mediante JSON Schema y validación Pydantic.
- Documentación automática en `/docs`.
- `.env` para configuración sin hardcodear credenciales.

## 7. Nota sobre seguridad

El archivo `.env` incluido está vacío y sirve solo como plantilla local. No debe contener claves reales al subirlo a GitHub.
