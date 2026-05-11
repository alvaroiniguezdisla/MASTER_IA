"""Clientes HTTP para las APIs externas usadas por las herramientas MCP."""

from __future__ import annotations

import logging
from typing import Any

import requests

from config.settings import settings

logger = logging.getLogger(__name__)


class ExternalAPIError(RuntimeError):
    """Error controlado al llamar a una API externa."""


class ExchangeRateClient:
    """Cliente simple para ExchangeRate-API."""

    def __init__(self, api_key: str | None = None, timeout: int = 15) -> None:
        self.api_key = api_key or settings.API_KEY_EXCHANGE
        self.timeout = timeout

    def latest_rates(self, base_currency: str) -> dict[str, Any]:
        """Obtiene las tasas actuales para una moneda base."""
        base = base_currency.upper().strip()
        if not self.api_key or self.api_key == "tu_api_key_aqui":
            raise ExternalAPIError(
                "Falta API_KEY_EXCHANGE. Configúrala en el archivo .env para usar ExchangeRate-API."
            )

        url = settings.exchange_latest_url(base)
        logger.info("Consultando ExchangeRate-API para moneda base=%s", base)

        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.Timeout as exc:
            raise ExternalAPIError("Timeout consultando ExchangeRate-API.") from exc
        except requests.RequestException as exc:
            raise ExternalAPIError(f"Error de red consultando ExchangeRate-API: {exc}") from exc
        except ValueError as exc:
            raise ExternalAPIError("ExchangeRate-API devolvió una respuesta JSON inválida.") from exc

        if payload.get("result") != "success":
            error_type = payload.get("error-type", "error desconocido")
            raise ExternalAPIError(f"ExchangeRate-API rechazó la petición: {error_type}")

        if "conversion_rates" not in payload:
            raise ExternalAPIError("Respuesta inválida: faltan conversion_rates.")

        return payload


class OpenMeteoClient:
    """Cliente para geocodificación y clima de Open-Meteo."""

    def __init__(self, timeout: int = 15) -> None:
        self.timeout = timeout

    def geocode_city(self, city: str, count: int = 1, language: str = "es") -> dict[str, Any]:
        """Busca coordenadas de una ciudad usando Open-Meteo Geocoding."""
        params = {
            "name": city,
            "count": count,
            "language": language,
            "format": "json",
        }
        logger.info("Geocodificando ciudad=%s", city)

        try:
            response = requests.get(settings.BASE_URL_GEOCODING, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.Timeout as exc:
            raise ExternalAPIError("Timeout consultando Open-Meteo Geocoding.") from exc
        except requests.RequestException as exc:
            raise ExternalAPIError(f"Error de red consultando Open-Meteo Geocoding: {exc}") from exc
        except ValueError as exc:
            raise ExternalAPIError("Open-Meteo Geocoding devolvió JSON inválido.") from exc

        results = payload.get("results") or []
        if not results:
            raise ExternalAPIError(f"No se encontró ninguna ciudad para: {city}")

        return results[0]

    def current_weather(self, latitude: float, longitude: float) -> dict[str, Any]:
        """Obtiene el clima actual para unas coordenadas."""
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            "timezone": "auto",
        }
        logger.info("Consultando clima actual lat=%s lon=%s", latitude, longitude)

        try:
            response = requests.get(f"{settings.BASE_URL_WEATHER.rstrip('/')}/forecast", params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.Timeout as exc:
            raise ExternalAPIError("Timeout consultando Open-Meteo Weather.") from exc
        except requests.RequestException as exc:
            raise ExternalAPIError(f"Error de red consultando Open-Meteo Weather: {exc}") from exc
        except ValueError as exc:
            raise ExternalAPIError("Open-Meteo Weather devolvió JSON inválido.") from exc

        if "current" not in payload:
            raise ExternalAPIError("Respuesta inválida de Open-Meteo: falta current.")

        return payload

    def forecast(self, latitude: float, longitude: float, days: int = 3) -> dict[str, Any]:
        """Obtiene el pronóstico diario para unas coordenadas."""
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "forecast_days": days,
            "timezone": "auto",
        }
        logger.info("Consultando pronóstico lat=%s lon=%s days=%s", latitude, longitude, days)

        try:
            response = requests.get(f"{settings.BASE_URL_WEATHER.rstrip('/')}/forecast", params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.Timeout as exc:
            raise ExternalAPIError("Timeout consultando el pronóstico de Open-Meteo.") from exc
        except requests.RequestException as exc:
            raise ExternalAPIError(f"Error de red consultando pronóstico Open-Meteo: {exc}") from exc
        except ValueError as exc:
            raise ExternalAPIError("Open-Meteo Forecast devolvió JSON inválido.") from exc

        if "daily" not in payload:
            raise ExternalAPIError("Respuesta inválida de Open-Meteo: falta daily.")

        return payload
