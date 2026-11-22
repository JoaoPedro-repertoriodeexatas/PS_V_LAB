from __future__ import annotations

from collections.abc import Callable

from src.models import ContentType, CustomerProfile
from src.prompts.engine import build_prompt
from src.prompts.versions import get_version_module


def _build_strategy_prompt(
    content_type: ContentType,
    profile: CustomerProfile,
    ticket: str,
    version: str,
) -> str:
    version_module = get_version_module(version)
    template = version_module.TEMPLATES[content_type]
    return build_prompt(
        profile=profile,
        ticket=ticket,
        content_type=content_type,
        prompt_version=version_module.VERSION,
        task_instruction=template["task_instruction"],
        output_format=template["output_format"],
    )


def build_diagnosis_prompt(
    profile: CustomerProfile,
    ticket: str,
    version: str = "v2",
) -> str:
    """Monta o prompt da estratégia de diagnóstico estruturado do problema."""
    return _build_strategy_prompt("diagnosis", profile, ticket, version)


def build_resolution_prompt(
    profile: CustomerProfile,
    ticket: str,
    version: str = "v2",
) -> str:
    """Monta o prompt da estratégia de passo a passo de resolução adaptado."""
    return _build_strategy_prompt("resolution", profile, ticket, version)


def build_clarification_prompt(
    profile: CustomerProfile,
    ticket: str,
    version: str = "v2",
) -> str:
    """Monta o prompt da estratégia de perguntas priorizadas de esclarecimento."""
    return _build_strategy_prompt("clarification", profile, ticket, version)


def build_flow_summary_prompt(
    profile: CustomerProfile,
    ticket: str,
    version: str = "v2",
) -> str:
    """Monta o prompt da estratégia de resumo visual do fluxo de decisão."""
    return _build_strategy_prompt("flow_summary", profile, ticket, version)


StrategyBuilder = Callable[[CustomerProfile, str, str], str]

_STRATEGY_DISPATCHER: dict[ContentType, StrategyBuilder] = {
    "diagnosis": build_diagnosis_prompt,
    "resolution": build_resolution_prompt,
    "clarification": build_clarification_prompt,
    "flow_summary": build_flow_summary_prompt,
}


def get_prompt_for_content_type(
    content_type: ContentType,
    profile: CustomerProfile,
    ticket: str,
    version: str = "v2",
) -> str:
    """Roteia dinamicamente para a estratégia correspondente ao tipo de conteúdo."""
    builder = _STRATEGY_DISPATCHER.get(content_type)
    if builder is None:
        supported = ", ".join(_STRATEGY_DISPATCHER.keys())
        raise ValueError(
            f"Tipo de conteúdo inválido: '{content_type}'. Tipos suportados: {supported}."
        )
    return builder(profile, ticket, version)
