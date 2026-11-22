from __future__ import annotations

from types import ModuleType

from src.prompts.versions import v1, v2

PROMPT_VERSIONS: dict[str, ModuleType] = {
    v1.VERSION: v1,
    v2.VERSION: v2,
}

AVAILABLE_VERSIONS: tuple[str, ...] = tuple(PROMPT_VERSIONS.keys())


def get_version_module(version: str) -> ModuleType:
    """Retorna o módulo correspondente à versão solicitada ou levanta ValueError."""
    normalized = version.strip().lower()
    if normalized not in PROMPT_VERSIONS:
        supported = ", ".join(AVAILABLE_VERSIONS)
        raise ValueError(
            f"Versão de prompt inválida: '{version}'. Versões suportadas: {supported}."
        )
    return PROMPT_VERSIONS[normalized]


__all__ = ["AVAILABLE_VERSIONS", "PROMPT_VERSIONS", "get_version_module", "v1", "v2"]
