from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import ContentType, CustomerProfile, ExecutionRecord, TicketItem  # noqa: E402
from src.pipeline import (  # noqa: E402
    PipelineExecutionResult,
    build_rubric_evaluation,
    evaluate_execution,
    execute_pipeline,
)
from src.prompts import AVAILABLE_VERSIONS, get_prompt_for_content_type  # noqa: E402
from src.services import (  # noqa: E402
    BaseLLMClient,
    DEFAULT_HISTORY_PATH,
    JsonHistoryStorage,
    LLMServiceError,
)

DEFAULT_PROFILES_PATH = PROJECT_ROOT / "data" / "profiles.json"
DEFAULT_TICKETS_PATH = PROJECT_ROOT / "data" / "tickets.json"

CONTENT_TYPES: tuple[ContentType, ...] = (
    "diagnosis",
    "resolution",
    "clarification",
    "flow_summary",
)

CONTENT_TYPE_LABELS: dict[ContentType, str] = {
    "diagnosis": "Diagnóstico Estruturado (Hipóteses, Evidências, Lacunas, Conclusão)",
    "resolution": "Passo a Passo de Resolução Adaptado",
    "clarification": "Perguntas Priorizadas de Esclarecimento",
    "flow_summary": "Resumo Visual do Fluxo (Fluxograma ASCII)",
}


def load_profiles(path: Path = DEFAULT_PROFILES_PATH) -> list[CustomerProfile]:
    """Carrega e valida os perfis de cliente a partir do arquivo JSON."""
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo de perfis não encontrado: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Formato inválido em {path}: esperado uma lista JSON.")
    return [CustomerProfile.model_validate(item) for item in raw]


def load_tickets(path: Path = DEFAULT_TICKETS_PATH) -> list[TicketItem]:
    """Carrega e valida os tickets de suporte a partir do arquivo JSON."""
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo de tickets não encontrado: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Formato inválido em {path}: esperado uma lista JSON.")
    return [TicketItem.model_validate(item) for item in raw]


def find_profile_by_id(
    profiles: list[CustomerProfile],
    customer_id: str,
) -> CustomerProfile:
    """Localiza um perfil pelo customer_id ou levanta ValueError informativo."""
    target = customer_id.strip()
    for profile in profiles:
        if profile.customer_id == target:
            return profile
    available = ", ".join(p.customer_id for p in profiles)
    raise ValueError(
        f"Perfil '{customer_id}' não encontrado. IDs disponíveis: {available}."
    )


def find_ticket_by_id(tickets: list[TicketItem], ticket_id: str) -> TicketItem:
    """Localiza um ticket pelo ticket_id ou levanta ValueError informativo."""
    target = ticket_id.strip()
    for ticket in tickets:
        if ticket.ticket_id == target:
            return ticket
    available = ", ".join(t.ticket_id for t in tickets)
    raise ValueError(
        f"Ticket '{ticket_id}' não encontrado. IDs disponíveis: {available}."
    )


def build_parser() -> argparse.ArgumentParser:
    """Cria o parser argparse com suporte a execução direta e avaliação via flags."""
    parser = argparse.ArgumentParser(
        prog="ai-support-cli",
        description=(
            "Assistente de Atendimento Técnico com IA e Engenharia de Prompt. "
            "Execute com flags para modo direto ou sem argumentos obrigatórios para modo interativo."
        ),
    )
    parser.add_argument(
        "--customer",
        type=str,
        default=None,
        help="ID do perfil de cliente em data/profiles.json (ex.: cust_001).",
    )
    parser.add_argument(
        "--ticket-id",
        type=str,
        default=None,
        help="ID de um ticket pré-cadastrado em data/tickets.json (ex.: TKT-101).",
    )
    parser.add_argument(
        "--ticket-text",
        type=str,
        default=None,
        help="Relato de ticket em texto livre (alternativa a --ticket-id).",
    )
    parser.add_argument(
        "--type",
        dest="content_type",
        choices=list(CONTENT_TYPES),
        default=None,
        help="Tipo de conteúdo a gerar.",
    )
    parser.add_argument(
        "--version",
        dest="prompt_version",
        choices=list(AVAILABLE_VERSIONS),
        default=None,
        help="Versão do prompt a utilizar (v1 ou v2; padrão: v2).",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignora o cache e força uma nova chamada à API.",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Exibe o prompt completo montado antes da resposta gerada.",
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=None,
        help=f"Caminho alternativo para o histórico JSON (padrão: {DEFAULT_HISTORY_PATH}).",
    )

    rubric_group = parser.add_argument_group("Rubrica de Avaliação (opcional via flags)")
    rubric_group.add_argument("--clarity", type=int, default=None, help="Nota de Clareza (1 a 5).")
    rubric_group.add_argument(
        "--clarity-justification",
        type=str,
        default=None,
        help="Justificativa para a nota de Clareza (mínimo 5 caracteres).",
    )
    rubric_group.add_argument(
        "--customer-fit",
        type=int,
        default=None,
        help="Nota de Adequação ao Cliente (1 a 5).",
    )
    rubric_group.add_argument(
        "--customer-fit-justification",
        type=str,
        default=None,
        help="Justificativa para a nota de Adequação ao Cliente (mínimo 5 caracteres).",
    )
    rubric_group.add_argument(
        "--actionability",
        type=int,
        default=None,
        help="Nota de Acionabilidade (1 a 5).",
    )
    rubric_group.add_argument(
        "--actionability-justification",
        type=str,
        default=None,
        help="Justificativa para a nota de Acionabilidade (mínimo 5 caracteres).",
    )
    rubric_group.add_argument(
        "--no-eval-prompt",
        action="store_true",
        help="Não pergunta interativamente se deseja avaliar a resposta ao final.",
    )
    return parser


def _prompt_select_profile(console: Console, profiles: list[CustomerProfile]) -> CustomerProfile:
    table = Table(title="Perfis de Cliente Disponíveis", show_lines=True)
    table.add_column("#", style="bold cyan", justify="right")
    table.add_column("ID", style="bold")
    table.add_column("Nome")
    table.add_column("Nível Técnico")
    table.add_column("Estilo")
    table.add_column("Canal")

    for idx, profile in enumerate(profiles, start=1):
        table.add_row(
            str(idx),
            profile.customer_id,
            profile.name,
            profile.technical_level,
            profile.communication_style,
            profile.channel,
        )
    console.print(table)

    while True:
        choice = IntPrompt.ask(
            f"Escolha o perfil [1-{len(profiles)}]",
            default=1,
            console=console,
        )
        if 1 <= choice <= len(profiles):
            return profiles[choice - 1]
        console.print(f"[red]Opção inválida. Informe um número entre 1 e {len(profiles)}.[/red]")


def _prompt_select_ticket(console: Console, tickets: list[TicketItem]) -> str:
    table = Table(title="Tickets de Suporte Disponíveis", show_lines=True)
    table.add_column("#", style="bold cyan", justify="right")
    table.add_column("ID", style="bold")
    table.add_column("Título")

    table.add_row("0", "CUSTOM", "Digitar um relato de ticket personalizado")
    for idx, ticket in enumerate(tickets, start=1):
        table.add_row(str(idx), ticket.ticket_id, ticket.title)
    console.print(table)

    while True:
        choice = IntPrompt.ask(
            f"Escolha o ticket [0-{len(tickets)}]",
            default=1,
            console=console,
        )
        if choice == 0:
            custom_text = Prompt.ask("Digite a descrição completa do ticket", console=console).strip()
            if custom_text:
                return custom_text
            console.print("[red]O relato do ticket não pode ficar vazio.[/red]")
        elif 1 <= choice <= len(tickets):
            selected = tickets[choice - 1]
            return f"[{selected.ticket_id}] {selected.title}: {selected.description}"
        else:
            console.print(f"[red]Opção inválida. Informe um número entre 0 e {len(tickets)}.[/red]")


def _prompt_select_content_type(console: Console) -> ContentType:
    table = Table(title="Tipos de Conteúdo", show_lines=True)
    table.add_column("#", style="bold cyan", justify="right")
    table.add_column("Tipo", style="bold")
    table.add_column("Descrição")

    for idx, content_type in enumerate(CONTENT_TYPES, start=1):
        table.add_row(str(idx), content_type, CONTENT_TYPE_LABELS[content_type])
    console.print(table)

    while True:
        choice = IntPrompt.ask(
            f"Escolha o tipo de conteúdo [1-{len(CONTENT_TYPES)}]",
            default=1,
            console=console,
        )
        if 1 <= choice <= len(CONTENT_TYPES):
            return CONTENT_TYPES[choice - 1]
        console.print(f"[red]Opção inválida. Informe um número entre 1 e {len(CONTENT_TYPES)}.[/red]")


def _prompt_select_version(console: Console) -> str:
    options = list(AVAILABLE_VERSIONS)
    default_version = "v2" if "v2" in options else options[0]
    return Prompt.ask(
        "Escolha a versão do prompt",
        choices=options,
        default=default_version,
        console=console,
    )


def _format_output_for_display(output: dict | str) -> str:
    if isinstance(output, str):
        return output
    return json.dumps(output, indent=2, ensure_ascii=False)


def _render_execution_result(
    console: Console,
    profile: CustomerProfile,
    result: PipelineExecutionResult,
    prompt_text: str | None = None,
) -> None:
    record = result.record

    if prompt_text is not None:
        console.print(
            Panel(
                Text(prompt_text),
                title=f"Prompt Montado ({record.prompt_version} / {record.content_type})",
                border_style="blue",
            )
        )

    cache_status = (
        "[bold cyan]CACHE HIT[/bold cyan] (reutilizado do histórico local)"
        if result.from_cache
        else "[bold green]NOVA EXECUÇÃO API[/bold green] (gerado e persistido)"
    )

    meta_table = Table.grid(padding=(0, 2))
    meta_table.add_column(style="bold")
    meta_table.add_column()
    meta_table.add_row("Status do Cache:", cache_status)
    meta_table.add_row(
        "Cliente:",
        f"{profile.name} ({profile.customer_id}) | {profile.technical_level} | {profile.communication_style} | {profile.channel}",
    )
    meta_table.add_row("Conteúdo / Versão:", f"{record.content_type} | {record.prompt_version}")
    meta_table.add_row(
        "Provedor / Modelo:",
        f"{record.provider} / {record.model} (temperature={record.temperature})",
    )
    if result.from_cache:
        if record.latency_ms is not None:
            latency_label = (
                f"{result.elapsed_ms:.2f} ms (API original: {record.latency_ms:.2f} ms)"
            )
        else:
            latency_label = f"{result.elapsed_ms:.2f} ms (registro legado)"
    else:
        latency_label = f"{result.elapsed_ms:.2f} ms"

    meta_table.add_row("Latência:", latency_label)
    meta_table.add_row("Timestamp (UTC):", record.timestamp)
    meta_table.add_row("Cache Key:", record.cache_key)
    meta_table.add_row("Arquivo Salvo:", str(result.storage_path))

    console.print(
        Panel(
            meta_table,
            title="Metadados da Execução",
            border_style="cyan" if result.from_cache else "green",
        )
    )

    output_str = _format_output_for_display(record.output)
    console.print(
        Panel(
            Text(output_str),
            title=f"Saída Gerada — {record.content_type} ({record.prompt_version})",
            border_style="white",
        )
    )


def _render_evaluation_panel(console: Console, record: ExecutionRecord, storage_path: Path) -> None:
    if record.evaluation is None:
        return
    ev = record.evaluation
    eval_table = Table(show_header=True, header_style="bold green")
    eval_table.add_column("Critério")
    eval_table.add_column("Nota", justify="center")
    eval_table.add_column("Justificativa")
    eval_table.add_row("Clareza (clarity)", f"{ev.clarity.score}/5", ev.clarity.justification)
    eval_table.add_row(
        "Adequação ao Cliente (customer_fit)",
        f"{ev.customer_fit.score}/5",
        ev.customer_fit.justification,
    )
    eval_table.add_row(
        "Acionabilidade (actionability)",
        f"{ev.actionability.score}/5",
        ev.actionability.justification,
    )
    console.print(
        Panel(
            eval_table,
            title=f"Rubrica Salva em {storage_path}",
            border_style="green",
        )
    )


def _prompt_rubric_score(console: Console, label: str) -> tuple[int, str]:
    while True:
        score = IntPrompt.ask(f"Nota para {label} [1-5]", console=console)
        if 1 <= score <= 5:
            break
        console.print("[red]A nota deve ser um inteiro entre 1 e 5.[/red]")

    while True:
        justification = Prompt.ask(
            f"Justificativa curta para {label} (mín. 5 caracteres)",
            console=console,
        ).strip()
        if len(justification) >= 5:
            return score, justification
        console.print("[red]A justificativa precisa ter pelo menos 5 caracteres.[/red]")


def _handle_rubric_evaluation(
    args: argparse.Namespace,
    result: PipelineExecutionResult,
    storage: JsonHistoryStorage,
    console: Console,
    allow_interactive: bool,
) -> ExecutionRecord | None:
    score_flags = (args.clarity, args.customer_fit, args.actionability)
    has_any_flag = any(s is not None for s in score_flags)

    if has_any_flag:
        if not all(s is not None for s in score_flags):
            raise ValueError(
                "Para avaliar via flags, informe todas as 3 notas: "
                "--clarity, --customer-fit e --actionability."
            )
        clarity_just = (
            args.clarity_justification
            or f"Avaliação via CLI para Clareza (nota {args.clarity}/5)."
        )
        fit_just = (
            args.customer_fit_justification
            or f"Avaliação via CLI para Adequação ao Cliente (nota {args.customer_fit}/5)."
        )
        act_just = (
            args.actionability_justification
            or f"Avaliação via CLI para Acionabilidade (nota {args.actionability}/5)."
        )
        evaluation = build_rubric_evaluation(
            clarity_score=args.clarity,
            clarity_justification=clarity_just,
            customer_fit_score=args.customer_fit,
            customer_fit_justification=fit_just,
            actionability_score=args.actionability,
            actionability_justification=act_just,
        )
        updated = evaluate_execution(
            cache_key=result.record.cache_key,
            evaluation=evaluation,
            storage=storage,
        )
        _render_evaluation_panel(console, updated, storage.file_path)
        return updated

    if not allow_interactive or args.no_eval_prompt:
        return None

    wants_eval = Confirm.ask(
        "Deseja preencher a rubrica de avaliação (notas 1 a 5) para este resultado?",
        default=False,
        console=console,
    )
    if not wants_eval:
        return None

    clarity_score, clarity_just = _prompt_rubric_score(console, "Clareza (clarity)")
    fit_score, fit_just = _prompt_rubric_score(console, "Adequação ao Cliente (customer_fit)")
    act_score, act_just = _prompt_rubric_score(console, "Acionabilidade (actionability)")

    evaluation = build_rubric_evaluation(
        clarity_score=clarity_score,
        clarity_justification=clarity_just,
        customer_fit_score=fit_score,
        customer_fit_justification=fit_just,
        actionability_score=act_score,
        actionability_justification=act_just,
    )
    updated = evaluate_execution(
        cache_key=result.record.cache_key,
        evaluation=evaluation,
        storage=storage,
    )
    _render_evaluation_panel(console, updated, storage.file_path)
    return updated


def main(
    argv: Sequence[str] | None = None,
    *,
    llm_client: BaseLLMClient | None = None,
    storage: JsonHistoryStorage | None = None,
    console: Console | None = None,
    profiles_path: Path = DEFAULT_PROFILES_PATH,
    tickets_path: Path = DEFAULT_TICKETS_PATH,
    interactive: bool | None = None,
) -> int:
    """Ponto de entrada da CLI (modo direto via flags ou modo interativo guiado)."""
    active_console = console or Console()
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    allow_interactive = sys.stdin.isatty() if interactive is None else interactive

    try:
        if args.ticket_id and args.ticket_text:
            raise ValueError("Informe apenas --ticket-id OU --ticket-text, não ambos.")

        profiles = load_profiles(profiles_path)
        tickets = load_tickets(tickets_path)

        needs_guided_input = any(
            [
                args.customer is None,
                args.ticket_id is None and args.ticket_text is None,
                args.content_type is None,
            ]
        )

        if needs_guided_input and not allow_interactive:
            raise ValueError(
                "Argumentos obrigatórios ausentes em modo não-interativo. "
                "Informe --customer, (--ticket-id ou --ticket-text) e --type."
            )

        if args.customer is not None:
            profile = find_profile_by_id(profiles, args.customer)
        else:
            profile = _prompt_select_profile(active_console, profiles)

        if args.ticket_id is not None:
            ticket_item = find_ticket_by_id(tickets, args.ticket_id)
            ticket_text = f"[{ticket_item.ticket_id}] {ticket_item.title}: {ticket_item.description}"
        elif args.ticket_text is not None:
            ticket_text = args.ticket_text.strip()
            if not ticket_text:
                raise ValueError("O texto fornecido em --ticket-text não pode estar vazio.")
        else:
            ticket_text = _prompt_select_ticket(active_console, tickets)

        content_type: ContentType = (
            args.content_type
            if args.content_type is not None
            else _prompt_select_content_type(active_console)
        )

        if args.prompt_version is not None:
            prompt_version = args.prompt_version
        elif needs_guided_input:
            prompt_version = _prompt_select_version(active_console)
        else:
            prompt_version = "v2"

        resolved_storage = storage or JsonHistoryStorage(
            args.history_path if args.history_path is not None else DEFAULT_HISTORY_PATH
        )

        prompt_preview: str | None = None
        if args.show_prompt:
            prompt_preview = get_prompt_for_content_type(
                content_type=content_type,
                profile=profile,
                ticket=ticket_text,
                version=prompt_version,
            )

        result = execute_pipeline(
            profile=profile,
            ticket=ticket_text,
            content_type=content_type,
            prompt_version=prompt_version,
            force_refresh=args.force_refresh,
            llm_client=llm_client,
            storage=resolved_storage,
        )

        _render_execution_result(
            console=active_console,
            profile=profile,
            result=result,
            prompt_text=prompt_preview,
        )

        _handle_rubric_evaluation(
            args=args,
            result=result,
            storage=resolved_storage,
            console=active_console,
            allow_interactive=allow_interactive,
        )
        return 0

    except (LLMServiceError, ValueError, ValidationError, KeyError, FileNotFoundError) as exc:
        active_console.print(
            Panel(
                Text(str(exc), style="bold red"),
                title="Erro na Execução da CLI",
                border_style="red",
            )
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
