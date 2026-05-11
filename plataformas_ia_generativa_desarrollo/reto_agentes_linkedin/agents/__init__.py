"""Paquete local de agentes del proyecto.

IMPORTANTE:
El SDK oficial se importa normalmente como `agents`, pero este ejercicio exige
una carpeta local llamada `agents/`. Para evitar el choque de nombres, este
archivo carga el SDK oficial desde site-packages bajo el alias interno
`_openai_agents_sdk` y reexporta sus clases principales.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import pathlib
import sys
from types import ModuleType
from typing import Any


def _load_openai_agents_sdk() -> ModuleType:
    """Carga el paquete oficial `agents` evitando que lo tape este paquete local."""
    alias = "_openai_agents_sdk"
    if alias in sys.modules:
        return sys.modules[alias]

    project_root = pathlib.Path(__file__).resolve().parents[1]
    search_paths: list[str] = []

    for path in sys.path:
        if not path:
            continue
        try:
            resolved = pathlib.Path(path).resolve()
        except OSError:
            continue
        if resolved == project_root:
            continue
        search_paths.append(path)

    spec = importlib.machinery.PathFinder.find_spec("agents", search_paths)
    if spec is None or spec.loader is None:
        raise ImportError(
            "No se encontró el OpenAI Agents SDK. Instálalo con: pip install openai-agents"
        )

    alias_spec = importlib.util.spec_from_file_location(
        alias,
        spec.origin,
        submodule_search_locations=spec.submodule_search_locations,
    )
    if alias_spec is None or alias_spec.loader is None:
        raise ImportError("No se pudo cargar el OpenAI Agents SDK.")

    module = importlib.util.module_from_spec(alias_spec)
    sys.modules[alias] = module
    alias_spec.loader.exec_module(module)
    return module


_sdk = _load_openai_agents_sdk()

# Reexportamos las piezas más usadas del SDK oficial.
Agent: Any = _sdk.Agent
Runner: Any = _sdk.Runner
handoff: Any = getattr(_sdk, "handoff", None)
ModelSettings: Any = getattr(_sdk, "ModelSettings", None)
MaxTurnsExceeded: Any = getattr(_sdk, "MaxTurnsExceeded", Exception)

__all__ = [
    "Agent",
    "Runner",
    "handoff",
    "ModelSettings",
    "MaxTurnsExceeded",
]
