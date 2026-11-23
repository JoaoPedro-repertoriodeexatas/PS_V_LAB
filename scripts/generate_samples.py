from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cli import find_profile_by_id, find_ticket_by_id, load_profiles, load_tickets  # noqa: E402
from src.models import ContentType, CustomerProfile, RubricEvaluation  # noqa: E402
from src.pipeline import (  # noqa: E402
    PipelineExecutionResult,
    build_rubric_evaluation,
    evaluate_execution,
    execute_pipeline,
)
from src.services import (  # noqa: E402
    BaseLLMClient,
    GeminiClient,
    JsonHistoryStorage,
    LLMServiceError,
)

SAMPLES_DIR = PROJECT_ROOT / "samples"


@dataclass(frozen=True)
class SampleJob:
    """Especificação de um cenário a gerar e avaliar em um arquivo de samples/."""

    filename: str
    customer_id: str
    ticket_id: str
    content_type: ContentType
    prompt_version: str
    evaluation: RubricEvaluation


def _format_ticket_text(ticket_id: str, title: str, description: str) -> str:
    return f"[{ticket_id}] {title}: {description}"


def _execute_with_retry(
    profile: CustomerProfile,
    ticket: str,
    content_type: ContentType,
    prompt_version: str,
    llm_client: BaseLLMClient,
    storage: JsonHistoryStorage,
    force_refresh: bool = False,
    max_attempts: int = 3,
    base_delay_seconds: float = 2.0,
) -> PipelineExecutionResult:
    """Executa o pipeline com retry exponencial curto para picos momentâneos (ex.: HTTP 503)."""
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return execute_pipeline(
                profile=profile,
                ticket=ticket,
                content_type=content_type,
                prompt_version=prompt_version,
                force_refresh=force_refresh,
                llm_client=llm_client,
                storage=storage,
            )
        except LLMServiceError as exc:
            last_error = exc
            if attempt < max_attempts:
                delay = base_delay_seconds * (2 ** (attempt - 1))
                time.sleep(delay)
    assert last_error is not None
    raise last_error


def build_sample_jobs() -> list[SampleJob]:
    """Define os cenários auditáveis da entrega final em samples/."""
    return [
        SampleJob(
            filename="comparison_v1_vs_v2_diagnosis.json",
            customer_id="cust_001",
            ticket_id="TKT-102",
            content_type="diagnosis",
            prompt_version="v1",
            evaluation=build_rubric_evaluation(
                clarity_score=3,
                clarity_justification=(
                    "Estrutura as 4 seções solicitadas, porém mantém tom genérico e "
                    "não prioriza explicitamente hipóteses nem indica nível de confiança."
                ),
                customer_fit_score=2,
                customer_fit_justification=(
                    "Usa jargões técnicos sem tradução (webhook, payload síncrono, HTTP 504) "
                    "e blocos extensos inadequados para cliente leigo em atendimento por chat."
                ),
                actionability_score=3,
                actionability_justification=(
                    "Aponta o gargalo geral de tempo de resposta, mas deixa a próxima validação "
                    "pouco objetiva para o atendente conduzir com o cliente."
                ),
            ),
        ),
        SampleJob(
            filename="comparison_v1_vs_v2_diagnosis.json",
            customer_id="cust_001",
            ticket_id="TKT-102",
            content_type="diagnosis",
            prompt_version="v2",
            evaluation=build_rubric_evaluation(
                clarity_score=4,
                clarity_justification=(
                    "Separa rigorosamente hipóteses priorizadas, evidências do relato, "
                    "dados faltantes e conclusão com nível de confiança explícito (alto); "
                    "descontou-se 1 ponto porque o modelo ecoou parte da instrução descritiva "
                    "nos títulos das seções (achado documentado para futura v3)."
                ),
                customer_fit_score=5,
                customer_fit_justification=(
                    "Traduz termos técnicos para linguagem acessível e didática ('endereço "
                    "digital', 'tempo esgotado', 'mais de meio minuto'), em blocos curtos e "
                    "escaneáveis calibrados ao perfil leigo no canal chat."
                ),
                actionability_score=5,
                actionability_justification=(
                    "Explicita o nível de confiança (alto), lista lacunas objetivas e define "
                    "a primeira validação recomendada sem exigir conhecimento técnico avançado."
                ),
            ),
        ),
        SampleJob(
            filename="sample_resolution_advanced.json",
            customer_id="cust_003",
            ticket_id="TKT-103",
            content_type="resolution",
            prompt_version="v2",
            evaluation=build_rubric_evaluation(
                clarity_score=5,
                clarity_justification=(
                    "Passo a passo numerado, direto e sem rodeios, com critério claro de validação "
                    "em cada etapa e bloco objetivo de escalonamento para N2."
                ),
                customer_fit_score=5,
                customer_fit_justification=(
                    "Emprega terminologia técnica precisa (HTTP 403, escopos de token, rotação de "
                    "credenciais, endpoint) em formato enxuto ideal para perfil avançado no chat."
                ),
                actionability_score=5,
                actionability_justification=(
                    "Cada passo traz ação executável, sinal esperado de sucesso e evidências "
                    "exatas a coletar caso o erro 403 persista."
                ),
            ),
        ),
        SampleJob(
            filename="sample_clarification_layman.json",
            customer_id="cust_004",
            ticket_id="TKT-104",
            content_type="clarification",
            prompt_version="v2",
            evaluation=build_rubric_evaluation(
                clarity_score=5,
                clarity_justification=(
                    "Lista priorizada dentro do limite de 3 a 5 perguntas essenciais, sem repetir "
                    "informações já presentes no relato do ticket."
                ),
                customer_fit_score=5,
                customer_fit_justification=(
                    "Frases curtas, diretas e naturais para fala em voz alta por telefone, sem "
                    "jargões técnicos para um perfil leigo com pouco tempo."
                ),
                actionability_score=5,
                actionability_justification=(
                    "Explica objetivamente por que cada resposta destrava o diagnóstico entre "
                    "falha de Wi-Fi local, terminal PDV ou instabilidade no aplicativo."
                ),
            ),
        ),
        SampleJob(
            filename="sample_flow_summary_ascii.json",
            customer_id="cust_002",
            ticket_id="TKT-101",
            content_type="flow_summary",
            prompt_version="v2",
            evaluation=build_rubric_evaluation(
                clarity_score=5,
                clarity_justification=(
                    "Fluxograma ASCII limpo, bem alinhado com caracteres de caixa e ramificações "
                    "explícitas [SIM] e [NÃO], preservando indentação exata no JSON."
                ),
                customer_fit_score=4,
                customer_fit_justification=(
                    "Equilibra termos funcionais e contexto detalhado adequados ao perfil "
                    "intermediário, servindo como referência clara de triagem."
                ),
                actionability_score=5,
                actionability_justification=(
                    "Conduz o atendente ponta a ponta desde a verificação do intervalo de 90 dias "
                    "até a resolução paliativa ou escalonamento estruturado para N2."
                ),
            ),
        ),
    ]


def generate_samples(
    samples_dir: Path = SAMPLES_DIR,
    force_refresh: bool = False,
    llm_client: BaseLLMClient | None = None,
) -> list[tuple[SampleJob, PipelineExecutionResult]]:
    """Executa todos os cenários configurados e salva os arquivos auditáveis em samples/."""
    samples_dir.mkdir(parents=True, exist_ok=True)
    profiles = load_profiles()
    tickets = load_tickets()
    shared_client = llm_client or GeminiClient()

    jobs = build_sample_jobs()

    if force_refresh:
        unique_files = {samples_dir / job.filename for job in jobs}
        for file_path in unique_files:
            if file_path.exists():
                file_path.unlink()

    completed: list[tuple[SampleJob, PipelineExecutionResult]] = []
    for job in jobs:
        profile = find_profile_by_id(profiles, job.customer_id)
        ticket_item = find_ticket_by_id(tickets, job.ticket_id)
        ticket_text = _format_ticket_text(
            ticket_item.ticket_id,
            ticket_item.title,
            ticket_item.description,
        )
        storage = JsonHistoryStorage(samples_dir / job.filename)

        result = _execute_with_retry(
            profile=profile,
            ticket=ticket_text,
            content_type=job.content_type,
            prompt_version=job.prompt_version,
            llm_client=shared_client,
            storage=storage,
            force_refresh=force_refresh,
        )
        updated_record = evaluate_execution(
            cache_key=result.record.cache_key,
            evaluation=job.evaluation,
            storage=storage,
        )
        final_result = PipelineExecutionResult(
            record=updated_record,
            from_cache=result.from_cache,
            storage_path=storage.file_path,
        )
        completed.append((job, final_result))

    return completed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gera e audita os exemplos JSON versionáveis em samples/."
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Recria os arquivos de samples/ forçando novas chamadas à API Gemini.",
    )
    args = parser.parse_args()

    console = Console()
    try:
        completed = generate_samples(force_refresh=args.force_refresh)
    except (LLMServiceError, ValueError, FileNotFoundError) as exc:
        console.print(
            Panel(
                f"[bold red]Erro ao gerar amostras:[/bold red]\n{exc}",
                title="generate_samples.py - ERRO",
                border_style="red",
            )
        )
        return 1

    table = Table(title="Amostras Geradas em samples/", show_lines=True)
    table.add_column("Arquivo", style="bold cyan")
    table.add_column("Cliente")
    table.add_column("Ticket")
    table.add_column("Tipo")
    table.add_column("Versão")
    table.add_column("Origem")
    table.add_column("Rubrica (C/A/Ac)")

    for job, result in completed:
        ev = result.record.evaluation
        scores = (
            f"{ev.clarity.score}/{ev.customer_fit.score}/{ev.actionability.score}"
            if ev is not None
            else "-"
        )
        origin = "CACHE HIT" if result.from_cache else "API GEMINI"
        table.add_row(
            job.filename,
            job.customer_id,
            job.ticket_id,
            job.content_type,
            job.prompt_version,
            origin,
            scores,
        )

    console.print(table)
    console.print(
        Panel(
            f"[bold green]Foram gerados e avaliados {len(completed)} registros em {SAMPLES_DIR}.[/bold green]",
            title="generate_samples.py - SUCESSO",
            border_style="green",
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
