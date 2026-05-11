"""Herramientas de conversión de monedas para exponer mediante MCP."""

from __future__ import annotations

import logging
from typing import Any

from server.api_clients import ExchangeRateClient, ExternalAPIError

logger = logging.getLogger(__name__)
SUPPORTED_CURRENCIES = {
    "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "CNY", "SEK", "NOK",
    "DKK", "MXN", "BRL", "ARS", "CLP", "COP", "PEN", "UYU", "INR", "KRW", "SGD",
    "HKD", "ZAR", "PLN", "CZK", "HUF", "TRY", "ILS", "AED",
}


def _validate_currency_code(code: str) -> str:
    normalized = code.upper().strip()
    if len(normalized) != 3 or not normalized.isalpha():
        raise ValueError("El código de moneda debe tener 3 letras, por ejemplo USD, EUR o GBP.")
    return normalized


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict[str, Any]:
    """
    Convierte una cantidad entre dos monedas usando tasas actuales de ExchangeRate-API.

    Args:
        amount: Cantidad positiva que se quiere convertir.
        from_currency: Código ISO de 3 letras de la moneda origen, por ejemplo USD.
        to_currency: Código ISO de 3 letras de la moneda destino, por ejemplo EUR.

    Returns:
        Diccionario con cantidad original, tasa de cambio y resultado convertido.
    """
    try:
        amount_float = float(amount)
        if amount_float <= 0:
            raise ValueError("La cantidad debe ser mayor que cero.")

        base = _validate_currency_code(from_currency)
        target = _validate_currency_code(to_currency)

        client = ExchangeRateClient()
        payload = client.latest_rates(base)
        rates = payload["conversion_rates"]

        if target not in rates:
            raise ValueError(f"La moneda destino {target} no está disponible en ExchangeRate-API.")

        rate = float(rates[target])
        converted = amount_float * rate

        logger.info("Conversión realizada: %.2f %s -> %.2f %s", amount_float, base, converted, target)
        return {
            "amount": round(amount_float, 4),
            "from_currency": base,
            "to_currency": target,
            "exchange_rate": rate,
            "converted_amount": round(converted, 4),
            "last_update_utc": payload.get("time_last_update_utc"),
            "next_update_utc": payload.get("time_next_update_utc"),
        }
    except (ValueError, ExternalAPIError) as exc:
        logger.warning("Error en convert_currency: %s", exc)
        return {"error": str(exc)}


def get_exchange_rates(base_currency: str, target_currencies: list[str] | None = None) -> dict[str, Any]:
    """
    Obtiene tasas de cambio actuales para una moneda base.

    Args:
        base_currency: Código ISO de 3 letras de la moneda base, por ejemplo EUR.
        target_currencies: Lista opcional de monedas destino. Si se omite, devuelve un subconjunto habitual.

    Returns:
        Diccionario con la moneda base y las tasas solicitadas.
    """
    try:
        base = _validate_currency_code(base_currency)
        targets = target_currencies or ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "MXN"]
        normalized_targets = [_validate_currency_code(code) for code in targets]

        client = ExchangeRateClient()
        payload = client.latest_rates(base)
        rates = payload["conversion_rates"]

        selected_rates = {}
        missing = []
        for code in normalized_targets:
            if code in rates:
                selected_rates[code] = rates[code]
            else:
                missing.append(code)

        return {
            "base_currency": base,
            "rates": selected_rates,
            "missing_currencies": missing,
            "last_update_utc": payload.get("time_last_update_utc"),
            "next_update_utc": payload.get("time_next_update_utc"),
        }
    except (ValueError, ExternalAPIError) as exc:
        logger.warning("Error en get_exchange_rates: %s", exc)
        return {"error": str(exc)}


def list_supported_currencies() -> list[str]:
    """Devuelve un listado corto de códigos de moneda habituales para la CLI."""
    return sorted(SUPPORTED_CURRENCIES)
