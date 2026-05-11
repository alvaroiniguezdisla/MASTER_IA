"""Punto de entrada del cliente CLI."""

from __future__ import annotations

import logging

from client.cli_interface import CLIInterface
from config.settings import settings


def configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


if __name__ == "__main__":
    configure_logging()
    CLIInterface().run()
