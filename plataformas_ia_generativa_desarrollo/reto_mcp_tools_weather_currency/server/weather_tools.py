"""Herramientas meteorológicas para exponer mediante MCP."""

from __future__ import annotations

import logging
from typing import Any

from server.api_clients import ExternalAPIError, OpenMeteoClient

logger = logging.getLogger(__name__)

WEATHER_CODES = {
    0: "cielo despejado",
    1: "principalmente despejado",
    2: "parcialmente nublado",
    3: "nublado",
    45: "niebla",
    48: "niebla con escarcha",
    51: "llovizna ligera",
    53: "llovizna moderada",
    55: "llovizna intensa",
    56: "llovizna helada ligera",
    57: "llovizna helada intensa",
    61: "lluvia ligera",
    63: "lluvia moderada",
    65: "lluvia intensa",
    66: "lluvia helada ligera",
    67: "lluvia helada intensa",
    71: "nieve ligera",
    73: "nieve moderada",
    75: "nieve intensa",
    77: "granos de nieve",
    80: "chubascos ligeros",
    81: "chubascos moderados",
    82: "chubascos violentos",
    85: "chubascos de nieve ligeros",
    86: "chubascos de nieve intensos",
    95: "tormenta",
    96: "tormenta con granizo ligero",
    99: "tormenta con granizo intenso",
}


def _validate_coordinates(latitude: float, longitude: float) -> tuple[float, float]:
    lat = float(latitude)
    lon = float(longitude)
    if not -90 <= lat <= 90:
        raise ValueError("La latitud debe estar entre -90 y 90.")
    if not -180 <= lon <= 180:
        raise ValueError("La longitud debe estar entre -180 y 180.")
    return lat, lon


def weather_code_to_description(code: int | None) -> str:
    """Convierte un código WMO de Open-Meteo en descripción legible."""
    if code is None:
        return "descripción no disponible"
    return WEATHER_CODES.get(int(code), f"código meteorológico desconocido: {code}")


def get_current_weather(latitude: float, longitude: float) -> dict[str, Any]:
    """
    Obtiene el clima actual para unas coordenadas con Open-Meteo.

    Args:
        latitude: Latitud de la ubicación.
        longitude: Longitud de la ubicación.

    Returns:
        Diccionario con temperatura, humedad, viento, descripción meteorológica y hora de medición.
    """
    try:
        lat, lon = _validate_coordinates(latitude, longitude)
        client = OpenMeteoClient()
        payload = client.current_weather(lat, lon)
        current = payload["current"]
        units = payload.get("current_units", {})
        code = current.get("weather_code")

        return {
            "latitude": payload.get("latitude"),
            "longitude": payload.get("longitude"),
            "timezone": payload.get("timezone"),
            "time": current.get("time"),
            "temperature": current.get("temperature_2m"),
            "temperature_unit": units.get("temperature_2m", "°C"),
            "humidity": current.get("relative_humidity_2m"),
            "humidity_unit": units.get("relative_humidity_2m", "%"),
            "wind_speed": current.get("wind_speed_10m"),
            "wind_speed_unit": units.get("wind_speed_10m", "km/h"),
            "weather_code": code,
            "description": weather_code_to_description(code),
        }
    except (ValueError, ExternalAPIError) as exc:
        logger.warning("Error en get_current_weather: %s", exc)
        return {"error": str(exc)}


def get_weather_forecast(latitude: float, longitude: float, days: int = 3) -> dict[str, Any]:
    """
    Obtiene el pronóstico meteorológico diario para unas coordenadas.

    Args:
        latitude: Latitud de la ubicación.
        longitude: Longitud de la ubicación.
        days: Número de días de pronóstico, entre 1 y 7.

    Returns:
        Diccionario con una lista de días, temperaturas mínimas/máximas, probabilidad de precipitación y descripción.
    """
    try:
        lat, lon = _validate_coordinates(latitude, longitude)
        days_int = int(days)
        if not 1 <= days_int <= 7:
            raise ValueError("El número de días debe estar entre 1 y 7.")

        client = OpenMeteoClient()
        payload = client.forecast(lat, lon, days_int)
        daily = payload["daily"]
        units = payload.get("daily_units", {})

        forecast_days = []
        for idx, date in enumerate(daily.get("time", [])):
            code = daily.get("weather_code", [None] * days_int)[idx]
            forecast_days.append(
                {
                    "date": date,
                    "temperature_max": daily.get("temperature_2m_max", [None] * days_int)[idx],
                    "temperature_min": daily.get("temperature_2m_min", [None] * days_int)[idx],
                    "temperature_unit": units.get("temperature_2m_max", "°C"),
                    "precipitation_probability_max": daily.get(
                        "precipitation_probability_max", [None] * days_int
                    )[idx],
                    "precipitation_unit": units.get("precipitation_probability_max", "%"),
                    "weather_code": code,
                    "description": weather_code_to_description(code),
                }
            )

        return {
            "latitude": payload.get("latitude"),
            "longitude": payload.get("longitude"),
            "timezone": payload.get("timezone"),
            "forecast_days": forecast_days,
        }
    except (ValueError, ExternalAPIError) as exc:
        logger.warning("Error en get_weather_forecast: %s", exc)
        return {"error": str(exc)}
