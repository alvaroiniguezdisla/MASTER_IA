# Reto proyecto: MCP con herramientas de clima y conversión de monedas

Este proyecto implementa un servidor MCP con **FastMCP** y 5 herramientas conectadas a APIs públicas:

- `convert_currency`: convierte una cantidad entre dos monedas usando ExchangeRate-API.
- `get_exchange_rates`: obtiene tasas de cambio desde una moneda base.
- `geocode_city`: convierte ciudad en coordenadas usando Open-Meteo Geocoding.
- `get_current_weather`: obtiene clima actual desde coordenadas.
- `get_weather_forecast`: obtiene pronóstico de varios días desde coordenadas.

También incluye un cliente CLI que usa la **API Responses de OpenAI** con una herramienta MCP remota/local configurada.

## Estructura

```text
├── server/
│   ├── __init__.py
│   ├── mcp_server.py
│   ├── currency_tools.py
│   ├── weather_tools.py
│   ├── geocoding_tools.py
│   └── api_clients.py
├── client/
│   ├── __init__.py
│   ├── openai_client.py
│   └── cli_interface.py
├── config/
│   ├── __init__.py
│   └── settings.py
├── main_server.py
├── main_client.py
├── requirements.txt
├── .env.example
├── .env
└── README.md
```

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuración

Copia `.env.example` a `.env` si necesitas regenerarlo:

```bash
cp .env.example .env
```

Edita `.env` y configura:

```env
OPENAI_API_KEY=sk-tu_clave_openai
OPENAI_MODEL=gpt-4o-mini
API_KEY_EXCHANGE=tu_api_key_de_exchangerate
MCP_SERVER_URL=http://localhost:8000/mcp/
```

Open-Meteo no requiere API key.

## Ejecución

### 1. Iniciar el servidor MCP

En una terminal:

```bash
python main_server.py
```

Por defecto se levanta en:

```text
http://localhost:8000/mcp/
```

Y expone un endpoint de salud:

```text
http://localhost:8000/health
```

### 2. Iniciar el cliente CLI

En otra terminal:

```bash
python main_client.py
```

Comandos disponibles:

```text
/ayuda
/monedas
/reset
/salir
```

Ejemplos:

```text
Convierte 100 USD a EUR.
¿Cuál es el clima actual en Madrid?
Dame el pronóstico del tiempo para Nueva York durante 5 días.
¿Qué coordenadas tiene Tokio?
Dame las tasas de cambio de EUR a USD, GBP y JPY.
```

## Flujo esperado para clima

Consulta del usuario:

```text
¿Qué tiempo hace en Madrid?
```

Flujo de herramientas:

1. El modelo usa `geocode_city("Madrid")`.
2. Recibe latitud y longitud.
3. Usa `get_current_weather(latitude, longitude)`.
4. Devuelve una respuesta en español con temperatura, humedad, viento y descripción.

## Nota importante sobre MCP y localhost

El servidor corre en `localhost` como pide el ejercicio. Para pruebas con algunos entornos de OpenAI Responses API, el servidor MCP debe ser accesible desde la infraestructura de OpenAI. Si `http://localhost:8000/mcp/` no es alcanzable desde la API, usa un túnel como Devtunnel o ngrok y configura:

```env
MCP_SERVER_URL=https://tu-url-publica/mcp/
```

## Manejo de errores implementado

- Timeout y errores de red en ExchangeRate-API.
- Timeout y errores de red en Open-Meteo.
- Validación de monedas ISO de 3 letras.
- Validación de cantidad positiva para conversiones.
- Validación de coordenadas.
- Validación de días de pronóstico entre 1 y 7.
- Gestión de errores de OpenAI: conexión, timeout, límites de uso y bad requests.
- Mensajes comprensibles para el usuario en CLI.

## APIs utilizadas

- ExchangeRate-API: https://www.exchangerate-api.com/
- Open-Meteo Forecast: https://open-meteo.com/
- Open-Meteo Geocoding: https://open-meteo.com/en/docs/geocoding-api
- FastMCP: https://gofastmcp.com/
- OpenAI Responses API con MCP: https://platform.openai.com/docs/guides/tools-remote-mcp
