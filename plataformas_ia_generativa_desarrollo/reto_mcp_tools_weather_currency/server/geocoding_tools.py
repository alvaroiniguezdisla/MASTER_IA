"""Herramientas de geocodificación para exponer mediante MCP."""

from __future__ import annotations

import logging
from typing import Any

from server.api_clients import ExternalAPIError, OpenMeteoClient

logger = logging.getLogger(__name__)


def geocode_city(city: str, country_hint: str | None = None) -> dict[str, Any]:
    """
    Convierte el nombre de una ciudad en coordenadas usando Open-Meteo Geocoding.

    Args:
        city: Nombre de la ciudad que se quiere localizar, por ejemplo Madrid.
        country_hint: País opcional para ayudar a desambiguar la ciudad.

    Returns:
        Diccionario con latitud, longitud, país, zona horaria y nombre normalizado.
    """
    try:
        clean_city = city.strip()
        if len(clean_city) < 2:
            raise ValueError("El nombre de la ciudad debe tener al menos 2 caracteres.")

        query = f"{clean_city}, {country_hint.strip()}" if country_hint else clean_city
        client = OpenMeteoClient()
        result = client.geocode_city(query)

        return {
            "name": result.get("name"),
            "latitude": result.get("latitude"),
            "longitude": result.get("longitude"),
            "country": result.get("country"),
            "country_code": result.get("country_code"),
            "admin1": result.get("admin1"),
            "timezone": result.get("timezone"),
            "population": result.get("population"),
        }
    except (ValueError, ExternalAPIError) as exc:
        logger.warning("Error en geocode_city: %s", exc)
        return {"error": str(exc)}
