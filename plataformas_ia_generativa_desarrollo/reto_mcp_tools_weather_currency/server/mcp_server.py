"""Servidor MCP con 5 herramientas externas de moneda y clima."""

from __future__ import annotations

import logging
from typing import Any

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from server.currency_tools import convert_currency as convert_currency_impl
from server.currency_tools import get_exchange_rates as get_exchange_rates_impl
from server.geocoding_tools import geocode_city as geocode_city_impl
from server.weather_tools import get_current_weather as get_current_weather_impl
from server.weather_tools import get_weather_forecast as get_weather_forecast_impl

logger = logging.getLogger(__name__)

mcp = FastMCP(
    name="weather-currency-mcp",
    instructions=(
        "Servidor MCP con herramientas para convertir monedas y consultar clima. "
        "Para preguntas de clima por ciudad, primero usa geocode_city y después "
        "get_current_weather o get_weather_forecast con las coordenadas obtenidas."
    ),
)


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    """Endpoint HTTP sencillo para comprobar si el servidor está vivo."""
    return JSONResponse({"status": "ok", "server": "weather-currency-mcp"})


@mcp.tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict[str, Any]:
    """
    Convierte una cantidad de una moneda origen a una moneda destino usando ExchangeRate-API.

    Usa esta herramienta cuando el usuario pregunte cosas como "convierte 100 USD a EUR".
    Los códigos de moneda deben ser ISO de 3 letras, por ejemplo USD, EUR, GBP o JPY.
    """
    logger.info("Tool convert_currency llamada con amount=%s from=%s to=%s", amount, from_currency, to_currency)
    return convert_currency_impl(amount, from_currency, to_currency)


@mcp.tool
def get_exchange_rates(base_currency: str, target_currencies: list[str] | None = None) -> dict[str, Any]:
    """
    Devuelve tasas de cambio actuales para una moneda base usando ExchangeRate-API.

    Usa esta herramienta cuando el usuario pida cotizaciones, tasas de cambio o una tabla
    de equivalencias desde una moneda base. target_currencies es opcional.
    """
    logger.info("Tool get_exchange_rates llamada con base=%s targets=%s", base_currency, target_currencies)
    return get_exchange_rates_impl(base_currency, target_currencies)


@mcp.tool
def geocode_city(city: str, country_hint: str | None = None) -> dict[str, Any]:
    """
    Convierte el nombre de una ciudad en coordenadas geográficas con Open-Meteo.

    Usa esta herramienta antes de consultar el clima cuando el usuario dé una ciudad por nombre.
    Devuelve latitud, longitud, país, zona horaria y datos útiles de desambiguación.
    """
    logger.info("Tool geocode_city llamada con city=%s country_hint=%s", city, country_hint)
    return geocode_city_impl(city, country_hint)


@mcp.tool
def get_current_weather(latitude: float, longitude: float) -> dict[str, Any]:
    """
    Devuelve el clima actual para unas coordenadas usando Open-Meteo.

    Requiere latitud y longitud. Si el usuario solo proporciona una ciudad, primero llama a
    geocode_city para obtener las coordenadas y luego a esta herramienta.
    """
    logger.info("Tool get_current_weather llamada con lat=%s lon=%s", latitude, longitude)
    return get_current_weather_impl(latitude, longitude)


@mcp.tool
def get_weather_forecast(latitude: float, longitude: float, days: int = 3) -> dict[str, Any]:
    """
    Devuelve el pronóstico meteorológico de varios días para unas coordenadas.

    Requiere latitud, longitud y un número de días entre 1 y 7. Si el usuario solo da una ciudad,
    primero llama a geocode_city y después a esta herramienta.
    """
    logger.info("Tool get_weather_forecast llamada con lat=%s lon=%s days=%s", latitude, longitude, days)
    return get_weather_forecast_impl(latitude, longitude, days)


def get_mcp_server() -> FastMCP:
    """Devuelve la instancia del servidor MCP para pruebas o ejecución externa."""
    return mcp
