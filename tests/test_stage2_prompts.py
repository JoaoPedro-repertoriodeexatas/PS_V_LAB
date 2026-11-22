from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import ContentType, CustomerProfile  # noqa: E402
from src.prompts import (  # noqa: E402
    AVAILABLE_VERSIONS,
    build_clarification_prompt,
    build_diagnosis_prompt,
    build_flow_summary_prompt,
    build_resolution_prompt,
    get_prompt_for_content_type,
)

SAMPLE_PROFILE_LEIGO_CHAT = CustomerProfile(
    customer_id="cust_001",
    name="Mariana Silva",
    technical_level="leigo",
    communication_style="didático",
    channel="chat",
)

SAMPLE_PROFILE_AVANCADO_EMAIL = CustomerProfile(
    customer_id="cust_005",
    name="Rafael Oliveira",
    technical_level="avançado",
    communication_style="direto",
    channel="e-mail",
)

SAMPLE_TICKET = (
    "Nosso endpoint de webhook (/api/v2/webhooks/payments) começou a receber respostas "
    "HTTP 504 intermitentes durante picos de checkout entre 18h e 21h. O processamento "
    "síncrono do payload está levando entre 28s e 35s antes de retornar 200 OK."
)

ALL_CONTENT_TYPES: tuple[ContentType, ...] = (
    "diagnosis",
    "resolution",
    "clarification",
    "flow_summary",
)


def test_all_four_strategies_generate_for_v1_and_v2() -> None:
    """Garante a geração bem-sucedida das 4 estratégias para v1 e v2."""
    direct_builders = {
        "diagnosis": build_diagnosis_prompt,
        "resolution": build_resolution_prompt,
        "clarification": build_clarification_prompt,
        "flow_summary": build_flow_summary_prompt,
    }

    assert "v1" in AVAILABLE_VERSIONS and "v2" in AVAILABLE_VERSIONS

    for version in ("v1", "v2"):
        for content_type, builder in direct_builders.items():
            direct_prompt = builder(SAMPLE_PROFILE_LEIGO_CHAT, SAMPLE_TICKET, version=version)
            dispatched_prompt = get_prompt_for_content_type(
                content_type=content_type,  # type: ignore[arg-type]
                profile=SAMPLE_PROFILE_LEIGO_CHAT,
                ticket=SAMPLE_TICKET,
                version=version,
            )
            assert isinstance(direct_prompt, str) and len(direct_prompt) > 100
            assert direct_prompt == dispatched_prompt


def test_prompt_contains_profile_ticket_version_and_no_cot_rule() -> None:
    """Verifica presença dos atributos do cliente, ticket, tag de versão e proibição de CoT bruto."""
    for version in ("v1", "v2"):
        for content_type in ALL_CONTENT_TYPES:
            prompt = get_prompt_for_content_type(
                content_type=content_type,
                profile=SAMPLE_PROFILE_AVANCADO_EMAIL,
                ticket=SAMPLE_TICKET,
                version=version,
            )
            assert SAMPLE_PROFILE_AVANCADO_EMAIL.name in prompt
            assert SAMPLE_PROFILE_AVANCADO_EMAIL.technical_level in prompt
            assert SAMPLE_PROFILE_AVANCADO_EMAIL.communication_style in prompt
            assert SAMPLE_PROFILE_AVANCADO_EMAIL.channel in prompt
            assert SAMPLE_TICKET in prompt
            assert f"prompt_version: {version}" in prompt
            assert f"content_type: {content_type}" in prompt
            assert "Chain-of-Thought" in prompt or "raciocínio interno bruto" in prompt


def test_diagnosis_strategy_contains_four_pillars() -> None:
    """Confirma os 4 pilares obrigatórios do diagnóstico em v1 e v2."""
    required_pillars = (
        "Hipóteses",
        "Evidências",
        "Dados Faltantes",
        "Conclusão",
    )
    for version in ("v1", "v2"):
        prompt = build_diagnosis_prompt(
            profile=SAMPLE_PROFILE_LEIGO_CHAT,
            ticket=SAMPLE_TICKET,
            version=version,
        )
        for pillar in required_pillars:
            assert pillar in prompt, f"Pilar '{pillar}' ausente no prompt diagnosis ({version})."


def test_v1_and_v2_have_real_verifiable_differences() -> None:
    """Comprova evolução real entre v1 (baseline) e v2 (otimizada) para o mesmo cenário."""
    for content_type in ALL_CONTENT_TYPES:
        prompt_v1 = get_prompt_for_content_type(
            content_type=content_type,
            profile=SAMPLE_PROFILE_LEIGO_CHAT,
            ticket=SAMPLE_TICKET,
            version="v1",
        )
        prompt_v2 = get_prompt_for_content_type(
            content_type=content_type,
            profile=SAMPLE_PROFILE_LEIGO_CHAT,
            ticket=SAMPLE_TICKET,
            version="v2",
        )
        assert prompt_v1 != prompt_v2
        assert "[ADAPTAÇÃO AO PERFIL E CANAL]" not in prompt_v1
        assert "[ADAPTAÇÃO AO PERFIL E CANAL]" in prompt_v2
        assert len(prompt_v2) > len(prompt_v1)

    flow_v2 = build_flow_summary_prompt(SAMPLE_PROFILE_LEIGO_CHAT, SAMPLE_TICKET, version="v2")
    assert "[SIM]" in flow_v2 and "[NÃO]" in flow_v2

    clarification_v2 = build_clarification_prompt(
        SAMPLE_PROFILE_LEIGO_CHAT, SAMPLE_TICKET, version="v2"
    )
    assert "máximo 5 perguntas" in clarification_v2

    leigo_chat_v2 = build_resolution_prompt(SAMPLE_PROFILE_LEIGO_CHAT, SAMPLE_TICKET, version="v2")
    avancado_email_v2 = build_resolution_prompt(
        SAMPLE_PROFILE_AVANCADO_EMAIL, SAMPLE_TICKET, version="v2"
    )
    assert leigo_chat_v2 != avancado_email_v2
    assert "Zero jargão técnico" in leigo_chat_v2
    assert "e-mail profissional" in avancado_email_v2


def test_invalid_version_and_content_type_raise_value_error() -> None:
    """Garante ValueError para versão inexistente, content_type inválido ou ticket vazio."""
    with pytest.raises(ValueError):
        build_diagnosis_prompt(SAMPLE_PROFILE_LEIGO_CHAT, SAMPLE_TICKET, version="v99")

    with pytest.raises(ValueError):
        get_prompt_for_content_type(
            content_type="diagnosis",
            profile=SAMPLE_PROFILE_LEIGO_CHAT,
            ticket=SAMPLE_TICKET,
            version="v99",
        )

    with pytest.raises(ValueError):
        get_prompt_for_content_type(
            content_type="unknown_type",  # type: ignore[arg-type]
            profile=SAMPLE_PROFILE_LEIGO_CHAT,
            ticket=SAMPLE_TICKET,
            version="v2",
        )

    with pytest.raises(ValueError):
        build_resolution_prompt(SAMPLE_PROFILE_LEIGO_CHAT, "   ", version="v2")


def _run_standalone() -> int:
    """Executa os testes offline, exibe prévia visual comparativa e status em rich."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    console = Console()
    checks = [
        ("Geração das 4 estratégias em v1 e v2", test_all_four_strategies_generate_for_v1_and_v2),
        ("Presença de perfil, ticket, versão e regra anti-CoT", test_prompt_contains_profile_ticket_version_and_no_cot_rule),
        ("4 pilares obrigatórios na estratégia diagnosis", test_diagnosis_strategy_contains_four_pillars),
        ("Diferenças reais e verificáveis entre v1 e v2", test_v1_and_v2_have_real_verifiable_differences),
        ("Validação de ValueError para entradas inválidas", test_invalid_version_and_content_type_raise_value_error),
    ]

    passed: list[str] = []
    try:
        for label, check_fn in checks:
            check_fn()
            passed.append(f"[bold green][OK][/bold green] {label}")
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        console.print(
            Panel(
                f"[bold red]Falha no checkpoint da Etapa 2:[/bold red]\n{exc}",
                title="Checkpoint Etapa 2 - ERRO",
                border_style="red",
            )
        )
        return 1

    preview_leigo_chat = build_resolution_prompt(
        profile=SAMPLE_PROFILE_LEIGO_CHAT,
        ticket=SAMPLE_TICKET,
        version="v2",
    )
    preview_avancado_email = build_resolution_prompt(
        profile=SAMPLE_PROFILE_AVANCADO_EMAIL,
        ticket=SAMPLE_TICKET,
        version="v2",
    )

    console.print(
        Panel(
            Text(preview_leigo_chat),
            title="Prévia Visual (v2 / resolution) - Perfil: Leigo | Didático | Chat",
            border_style="cyan",
        )
    )
    console.print(
        Panel(
            Text(preview_avancado_email),
            title="Prévia Visual (v2 / resolution) - Perfil: Avançado | Direto | E-mail",
            border_style="magenta",
        )
    )

    summary = "\n".join(passed)
    console.print(
        Panel(
            f"{summary}\n\n[bold green]Todos os {len(passed)} checks da Etapa 2 foram validados com sucesso![/bold green]",
            title="Checkpoint Etapa 2 - SUCESSO",
            border_style="green",
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(_run_standalone())
