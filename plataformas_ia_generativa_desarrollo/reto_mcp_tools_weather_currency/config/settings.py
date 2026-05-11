"""Configuración central del proyecto."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    """Variables de entorno y valores por defecto."""

    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    API_KEY_EXCHANGE: str | None = os.getenv("API_KEY_EXCHANGE")
    BASE_URL_EXCHANGE: str = os.getenv(
        "BASE_URL_EXCHANGE",
        "https://v6.exchangerate-api.com/v6/{API_KEY_EXCHANGE}/latest/{MONEDA}",
    )

    BASE_URL_WEATHER: str = os.getenv("BASE_URL_WEATHER", "https://api.open-meteo.com/v1/")
    BASE_URL_GEOCODING: str = os.getenv(
        "BASE_URL_GEOCODING", "https://geocoding-api.open-meteo.com/v1/search"
    )

    MCP_HOST: str = os.getenv("MCP_HOST", "127.0.0.1")
    MCP_PORT: int = int(os.getenv("MCP_PORT", "8000"))
    MCP_PATH: str = os.getenv("MCP_PATH", "/mcp/")
    MCP_SERVER_URL: str = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp/")

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    def exchange_latest_url(self, moneda: str) -> str:
        """Construye la URL final de ExchangeRate-API."""
        api_key = self.API_KEY_EXCHANGE or ""
        return self.BASE_URL_EXCHANGE.replace("{API_KEY_EXCHANGE}", api_key).replace("{MONEDA}", moneda.upper())


settings = Settings()
