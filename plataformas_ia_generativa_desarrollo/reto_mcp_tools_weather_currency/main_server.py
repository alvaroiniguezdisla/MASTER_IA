"""Punto de entrada del servidor MCP."""

from __future__ import annotations

import logging

from config.settings import settings
from server.mcp_server import mcp


def configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


if __name__ == "__main__":
    configure_logging()
    logging.getLogger(__name__).info(
        "Iniciando servidor MCP en http://%s:%s%s",
        settings.MCP_HOST,
        settings.MCP_PORT,
        settings.MCP_PATH,
    )
    mcp.run(
        transport="http",
        host=settings.MCP_HOST,
        port=settings.MCP_PORT,
        path=settings.MCP_PATH,
    )
