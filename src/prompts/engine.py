from __future__ import annotations

from typing import get_args

from src.models import ContentType, CustomerProfile
from src.prompts.versions import get_version_module

VALID_CONTENT_TYPES: tuple[str, ...] = get_args(ContentType)

NO_RAW_COT_RULE = (
    "RESTRIÇÃO CRÍTICA DE RACIOCÍNIO: NÃO exiba raciocínio interno bruto, "
    "cadeia de pensamento não filtrada (Chain-of-Thought), rascunhos mentais ou "
    "tags como <thinking>. Entregue exclusivamente a saída final estruturada e verificável."
)


def _validate_content_type(content_type: str) -> ContentType:
    if content_type not in VALID_CONTENT_TYPES:
        supported = ", ".join(VALID_CONTENT_TYPES)
        raise ValueError(
            f"Tipo de conteúdo inválido: '{content_type}'. Tipos suportados: {supported}."
        )
    return content_type  # type: ignore[return-value]


def _validate_ticket(ticket: str) -> str:
    cleaned = ticket.strip()
    if not cleaned:
        raise ValueError("O texto do ticket não pode estar vazio.")
    return cleaned


def _format_context_block(profile: CustomerProfile, ticket: str) -> str:
    return (
        "Perfil do cliente:\n"
        f"- ID: {profile.customer_id}\n"
        f"- Nome: {profile.name}\n"
        f"- Nível técnico: {profile.technical_level}\n"
        f"- Estilo de comunicação: {profile.communication_style}\n"
        f"- Canal de atendimento: {profile.channel}\n\n"
        "Relato do ticket:\n"
        "<ticket>\n"
        f"{ticket}\n"
        "</ticket>"
    )


def build_prompt(
    profile: CustomerProfile,
    ticket: str,
    content_type: ContentType,
    prompt_version: str = "v2",
    task_instruction: str | None = None,
    output_format: str | None = None,
) -> str:
    """Monta dinamicamente um prompt estruturado com persona, contexto, tarefa, formato e versão."""
    validated_type = _validate_content_type(content_type)
    cleaned_ticket = _validate_ticket(ticket)
    version_module = get_version_module(prompt_version)

    template = version_module.TEMPLATES[validated_type]
    resolved_task = (task_instruction or template["task_instruction"]).strip()
    resolved_format = (output_format or template["output_format"]).strip()

    sections: list[str] = [
        (
            "[PROMPT_METADATA]\n"
            f"prompt_version: {version_module.VERSION}\n"
            f"content_type: {validated_type}\n"
            f"description: {version_module.DESCRIPTION}\n"
            "[/PROMPT_METADATA]"
        ),
        f"[PERSONA E LIMITES]\n{version_module.PERSONA}",
        f"[CONTEXTO DO CLIENTE E DO TICKET]\n{_format_context_block(profile, cleaned_ticket)}",
    ]

    adaptation_builder = getattr(version_module, "build_adaptation_guidelines", None)
    if callable(adaptation_builder):
        adaptation_rules = adaptation_builder(profile)
        sections.append(f"[ADAPTAÇÃO AO PERFIL E CANAL]\n{adaptation_rules}")

    sections.append(f"[TAREFA E DECOMPOSIÇÃO]\n{resolved_task}")

    constraints_lines = [f"- {rule}" for rule in version_module.GENERAL_CONSTRAINTS]
    constraints_lines.append(f"- {NO_RAW_COT_RULE}")
    constraints_block = "\n".join(constraints_lines)

    sections.append(
        "[FORMATO DE SAÍDA E RESTRIÇÕES]\n"
        f"{resolved_format}\n\n"
        "Restrições obrigatórias:\n"
        f"{constraints_block}"
    )

    return "\n\n".join(sections)
