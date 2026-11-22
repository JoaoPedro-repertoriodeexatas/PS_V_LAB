from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import (  # noqa: E402
    CriterionScore,
    CustomerProfile,
    ExecutionRecord,
    RubricEvaluation,
    TicketItem,
)

DATA_DIR = PROJECT_ROOT / "data"
PROFILES_PATH = DATA_DIR / "profiles.json"
TICKETS_PATH = DATA_DIR / "tickets.json"


def test_required_root_files_exist() -> None:
    """Verifica a existência dos arquivos essenciais de configuração na raiz do projeto."""
    required_files = [
        PROJECT_ROOT / ".env.example",
        PROJECT_ROOT / ".gitignore",
        PROJECT_ROOT / "requirements.txt",
    ]
    for file_path in required_files:
        assert file_path.is_file(), f"Arquivo obrigatório não encontrado: {file_path.name}"
        assert file_path.stat().st_size > 0, f"Arquivo obrigatório está vazio: {file_path.name}"


def test_profiles_json_valid_and_within_bounds() -> None:
    """Valida data/profiles.json com CustomerProfile e garante entre 3 e 5 perfis."""
    assert PROFILES_PATH.is_file(), "Arquivo data/profiles.json não encontrado."

    raw_profiles = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw_profiles, list), "data/profiles.json deve conter uma lista JSON."

    profiles = [CustomerProfile.model_validate(item) for item in raw_profiles]
    assert 3 <= len(profiles) <= 5, f"Esperado entre 3 e 5 perfis, encontrado: {len(profiles)}"

    customer_ids = [profile.customer_id for profile in profiles]
    assert len(set(customer_ids)) == len(customer_ids), "customer_id deve ser único em profiles.json."

    assert {p.technical_level for p in profiles} == {"leigo", "intermediário", "avançado"}
    assert {p.communication_style for p in profiles} == {"direto", "detalhado", "didático"}
    assert {p.channel for p in profiles} == {"chat", "e-mail", "telefone"}


def test_tickets_json_valid() -> None:
    """Valida o carregamento de data/tickets.json com TicketItem."""
    assert TICKETS_PATH.is_file(), "Arquivo data/tickets.json não encontrado."

    raw_tickets = json.loads(TICKETS_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw_tickets, list), "data/tickets.json deve conter uma lista JSON."

    tickets = [TicketItem.model_validate(item) for item in raw_tickets]
    assert len(tickets) >= 1, "data/tickets.json deve conter ao menos 1 ticket."

    ticket_ids = [ticket.ticket_id for ticket in tickets]
    assert len(set(ticket_ids)) == len(ticket_ids), "ticket_id deve ser único em tickets.json."


def test_criterion_score_boundaries() -> None:
    """Confirma que notas 0 e 6 lançam ValidationError e notas entre 1 e 5 passam."""
    for valid_score in range(1, 6):
        criterion = CriterionScore(
            score=valid_score,
            justification=f"Justificativa válida para nota {valid_score}.",
        )
        assert criterion.score == valid_score

    for invalid_score in (0, 6):
        with pytest.raises(ValidationError):
            CriterionScore(
                score=invalid_score,
                justification="Justificativa com nota fora do intervalo.",
            )

    with pytest.raises(ValidationError):
        CriterionScore(score=3, justification="curt")


def test_execution_record_schema_with_rubric() -> None:
    """Valida a construção de ExecutionRecord com RubricEvaluation e timestamp UTC padrão."""
    evaluation = RubricEvaluation(
        clarity=CriterionScore(score=4, justification="Resposta clara e bem estruturada."),
        customer_fit=CriterionScore(score=5, justification="Vocabulário adequado ao perfil."),
        actionability=CriterionScore(score=4, justification="Próximos passos objetivos."),
    )
    record = ExecutionRecord(
        cache_key="sha256:demo-key",
        customer_id="cust_001",
        ticket="TKT-101",
        prompt_version="v1",
        content_type="diagnosis",
        provider="google",
        model="gemini-2.5-flash",
        temperature=0.2,
        output={"hypotheses": ["Timeout na geração síncrona do CSV"]},
        evaluation=evaluation,
    )
    assert record.timestamp.endswith("+00:00") or record.timestamp.endswith("Z")
    assert record.evaluation is not None
    assert record.evaluation.customer_fit.score == 5


def _run_standalone() -> int:
    """Executa a suíte de validação diretamente e exibe o status com rich."""
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    checks = [
        ("Arquivos obrigatórios na raiz", test_required_root_files_exist),
        ("Validação de data/profiles.json", test_profiles_json_valid_and_within_bounds),
        ("Validação de data/tickets.json", test_tickets_json_valid),
        ("Limites da rubrica (CriterionScore)", test_criterion_score_boundaries),
        ("Esquema de ExecutionRecord e RubricEvaluation", test_execution_record_schema_with_rubric),
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
                f"[bold red]Falha no checkpoint da Etapa 1:[/bold red]\n{exc}",
                title="Checkpoint Etapa 1 - ERRO",
                border_style="red",
            )
        )
        return 1

    summary = "\n".join(passed)
    console.print(
        Panel(
            f"{summary}\n\n[bold green]Todos os {len(passed)} checks da Etapa 1 foram validados com sucesso![/bold green]",
            title="Checkpoint Etapa 1 - SUCESSO",
            border_style="green",
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(_run_standalone())
