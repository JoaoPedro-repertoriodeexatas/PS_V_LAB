from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path

import pytest
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import cli as cli_module  # noqa: E402
from src import web_app as web_app_module  # noqa: E402
from src.models import ContentType, ExecutionRecord  # noqa: E402
from src.pipeline import build_rubric_evaluation  # noqa: E402
from src.services import BaseLLMClient, JsonHistoryStorage, LLMServiceError  # noqa: E402

ASCII_SAMPLE = (
    "    +--------------------------+\n"
    "    | Erro 504 no Webhook?     |\n"
    "    +------------+-------------+\n"
    "                 |\n"
    "         +-------+-------+\n"
    "         |               |\n"
    "      [SIM]           [NÃO]\n"
    "         v               v\n"
    "    +----------+   +----------+\n"
    "    | Fila     |   | Verificar|\n"
    "    | Assíncr. |   | Auth     |\n"
    "    +----------+   +----------+"
)


class FakeLLMClient(BaseLLMClient):
    """Cliente LLM determinístico offline para testar CLI e helpers do Streamlit."""

    def __init__(
        self,
        response_text: str | None = None,
        should_fail: bool = False,
    ) -> None:
        self._response_text = response_text
        self._should_fail = should_fail
        self.call_count = 0

    @property
    def provider(self) -> str:
        return "google"

    @property
    def model(self) -> str:
        return "gemini-2.5-flash"

    @property
    def temperature(self) -> float:
        return 0.2

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        if self._should_fail:
            raise LLMServiceError("Erro simulado de rede na API Gemini.")
        if self._response_text is not None:
            return self._response_text
        version_tag = "v1" if "prompt_version: v1" in prompt else "v2"
        return f"Saída simulada ({version_tag}) — chamada #{self.call_count}"


def test_cli_loaders_profiles_and_tickets() -> None:
    """Valida carregamento e busca de perfis e tickets pelos helpers da CLI."""
    profiles = cli_module.load_profiles()
    tickets = cli_module.load_tickets()

    assert 3 <= len(profiles) <= 5
    assert len(tickets) >= 1

    profile = cli_module.find_profile_by_id(profiles, "cust_001")
    ticket = cli_module.find_ticket_by_id(tickets, "TKT-101")
    assert profile.customer_id == "cust_001"
    assert ticket.ticket_id == "TKT-101"

    with pytest.raises(ValueError):
        cli_module.find_profile_by_id(profiles, "cust_inexistente")
    with pytest.raises(ValueError):
        cli_module.find_ticket_by_id(tickets, "TKT-999")


def test_cli_non_interactive_execution_and_cache(tmp_path: Path) -> None:
    """Testa CLI em modo direto, exibição do caminho salvo, fluxo ASCII e cache hit."""
    history_path = tmp_path / "history.json"
    storage = JsonHistoryStorage(history_path)
    fake_llm = FakeLLMClient(response_text=ASCII_SAMPLE)
    console = Console(file=io.StringIO(), record=True, width=140)

    exit_code_1 = cli_module.main(
        [
            "--customer",
            "cust_001",
            "--ticket-id",
            "TKT-102",
            "--type",
            "flow_summary",
            "--version",
            "v2",
            "--show-prompt",
        ],
        llm_client=fake_llm,
        storage=storage,
        console=console,
        interactive=False,
    )
    rendered_1 = console.export_text()

    assert exit_code_1 == 0
    assert fake_llm.call_count == 1
    assert history_path.is_file()
    assert "NOVA EXECUÇÃO API" in rendered_1
    assert str(history_path) in rendered_1
    assert "[SIM]" in rendered_1 and "[NÃO]" in rendered_1
    assert "[PROMPT_METADATA]" in rendered_1

    console_2 = Console(file=io.StringIO(), record=True, width=140)
    exit_code_2 = cli_module.main(
        [
            "--customer",
            "cust_001",
            "--ticket-id",
            "TKT-102",
            "--type",
            "flow_summary",
            "--version",
            "v2",
        ],
        llm_client=fake_llm,
        storage=storage,
        console=console_2,
        interactive=False,
    )
    rendered_2 = console_2.export_text()

    assert exit_code_2 == 0
    assert fake_llm.call_count == 1
    assert "CACHE HIT" in rendered_2


def test_cli_evaluation_flow_and_error_handling(tmp_path: Path) -> None:
    """Valida anexo de rubrica via flags na CLI e tratamento limpo de LLMServiceError."""
    history_path = tmp_path / "history.json"
    storage = JsonHistoryStorage(history_path)
    fake_llm = FakeLLMClient()
    console = Console(file=io.StringIO(), record=True, width=140)

    exit_code = cli_module.main(
        [
            "--customer",
            "cust_002",
            "--ticket-text",
            "Erro 403 ao sincronizar estoque entre ERP e loja virtual.",
            "--type",
            "diagnosis",
            "--version",
            "v2",
            "--clarity",
            "5",
            "--clarity-justification",
            "Diagnóstico claro com 4 seções bem definidas.",
            "--customer-fit",
            "4",
            "--customer-fit-justification",
            "Adequado ao nível intermediário por e-mail.",
            "--actionability",
            "5",
            "--actionability-justification",
            "Próximos passos objetivos para investigação.",
        ],
        llm_client=fake_llm,
        storage=storage,
        console=console,
        interactive=False,
    )
    assert exit_code == 0

    records = storage.load_all()
    assert len(records) == 1
    assert records[0].evaluation is not None
    assert records[0].evaluation.clarity.score == 5
    assert records[0].evaluation.customer_fit.score == 4
    assert records[0].evaluation.actionability.score == 5

    # Testa tratamento amigável de LLMServiceError sem stack trace não tratado
    failing_llm = FakeLLMClient(should_fail=True)
    err_console = Console(file=io.StringIO(), record=True, width=140)
    err_code = cli_module.main(
        [
            "--customer",
            "cust_001",
            "--ticket-text",
            "Falha simulada para validação de tratamento de erro.",
            "--type",
            "resolution",
        ],
        llm_client=failing_llm,
        storage=storage,
        console=err_console,
        interactive=False,
    )
    assert err_code == 1
    assert "Erro simulado de rede na API Gemini." in err_console.export_text()


def test_web_app_helpers_offline(tmp_path: Path) -> None:
    """Valida helpers de geração, comparação v1/v2, rubrica e auditoria de src/web_app.py."""
    storage = JsonHistoryStorage(tmp_path / "web_history.json")
    fake_llm = FakeLLMClient()

    profiles = web_app_module.load_profiles_data()
    tickets = web_app_module.load_tickets_data()
    assert len(profiles) >= 3 and len(tickets) >= 1

    profile = profiles[0]
    ticket_text = tickets[0].description

    # Geração individual
    gen_result, gen_prompt = web_app_module.run_generation(
        profile=profile,
        ticket=ticket_text,
        content_type="diagnosis",
        prompt_version="v2",
        llm_client=fake_llm,
        storage=storage,
    )
    assert gen_result.from_cache is False
    assert "prompt_version: v2" in gen_prompt
    assert fake_llm.call_count == 1

    # Comparação controlada v1 vs v2 (v2 deve reutilizar o cache recém-gerado!)
    comparison = web_app_module.run_comparison(
        profile=profile,
        ticket=ticket_text,
        content_type="diagnosis",
        llm_client=fake_llm,
        storage=storage,
    )
    res_v1, prompt_v1 = comparison["v1"]
    res_v2, prompt_v2 = comparison["v2"]

    assert res_v1.from_cache is False
    assert res_v2.from_cache is True
    assert fake_llm.call_count == 2
    assert "prompt_version: v1" in prompt_v1
    assert "prompt_version: v2" in prompt_v2
    assert res_v1.record.model == res_v2.record.model
    assert res_v1.record.temperature == res_v2.record.temperature

    # Salvar rubrica via helper da web
    updated_v2 = web_app_module.save_rubric_rating(
        cache_key=res_v2.record.cache_key,
        clarity_score=5,
        clarity_justification="Estrutura excelente e fácil de acompanhar.",
        customer_fit_score=5,
        customer_fit_justification="Calibrado perfeitamente ao perfil leigo no chat.",
        actionability_score=4,
        actionability_justification="Ações práticas e verificáveis.",
        storage=storage,
    )
    assert updated_v2.evaluation is not None

    # Auditoria, resumo, filtro e exportação JSON
    all_records = storage.load_all()
    summary = web_app_module.summarize_history(all_records)
    assert summary == {"total": 2, "evaluated": 1, "v1": 1, "v2": 1}

    only_eval = web_app_module.filter_history_records(all_records, only_evaluated=True)
    assert len(only_eval) == 1
    assert only_eval[0].prompt_version == "v2"

    exported_json = web_app_module.export_history_json(only_eval)
    parsed_export = json.loads(exported_json)
    assert isinstance(parsed_export, list) and len(parsed_export) == 1


def _make_record(
    cache_key: str,
    prompt_version: str,
    content_type: ContentType,
    scores: tuple[int, int, int] | None = None,
) -> ExecutionRecord:
    evaluation = (
        build_rubric_evaluation(
            clarity_score=scores[0],
            clarity_justification="Justificativa de clareza válida.",
            customer_fit_score=scores[1],
            customer_fit_justification="Justificativa de adequação válida.",
            actionability_score=scores[2],
            actionability_justification="Justificativa de acionabilidade válida.",
        )
        if scores is not None
        else None
    )
    return ExecutionRecord(
        cache_key=cache_key,
        customer_id="cust_001",
        ticket="Ticket de teste para agregação analítica.",
        prompt_version=prompt_version,
        content_type=content_type,
        provider="google",
        model="gemini-2.5-flash",
        temperature=0.2,
        output="Saída simulada para agregação.",
        evaluation=evaluation,
    )


def test_aggregate_rubric_metrics_empty_and_unevaluated() -> None:
    """Valida aggregate_rubric_metrics para histórico vazio, sem avaliações e versão única."""
    empty_res = web_app_module.aggregate_rubric_metrics([])
    assert empty_res["evaluated_count"] == 0
    assert empty_res["by_version"]["v1"]["count"] == 0
    assert empty_res["by_version"]["v2"]["count"] == 0
    assert empty_res["by_version_and_type"] == []
    assert empty_res["deltas_v2_vs_v1"] is None

    unevaluated_only = [
        _make_record("k1", "v1", "diagnosis", scores=None),
        _make_record("k2", "v2", "resolution", scores=None),
    ]
    uneval_res = web_app_module.aggregate_rubric_metrics(unevaluated_only)
    assert uneval_res["evaluated_count"] == 0
    assert uneval_res["by_version"]["v1"]["count"] == 0
    assert uneval_res["by_version"]["v2"]["count"] == 0
    assert uneval_res["by_version_and_type"] == []
    assert uneval_res["deltas_v2_vs_v1"] is None

    single_version = [_make_record("k3", "v2", "diagnosis", scores=(4, 5, 5))]
    single_res = web_app_module.aggregate_rubric_metrics(single_version)
    assert single_res["evaluated_count"] == 1
    assert single_res["by_version"]["v1"]["count"] == 0
    assert single_res["by_version"]["v2"]["count"] == 1
    assert single_res["by_version"]["v2"]["total_avg"] == 14.0
    assert single_res["deltas_v2_vs_v1"] is None


def test_aggregate_rubric_metrics_averages_deltas_and_filter() -> None:
    """Valida médias exatas, deltas v2 vs v1, filtro por content_type e cruzamento versão x tipo."""
    records = [
        # v1 diagnosis: (3, 2, 3) -> total 8; (4, 3, 2) -> total 9
        _make_record("v1_d1", "v1", "diagnosis", scores=(3, 2, 3)),
        _make_record("v1_d2", "v1", "diagnosis", scores=(4, 3, 2)),
        # v1 resolution: (2, 4, 4) -> total 10
        _make_record("v1_r1", "v1", "resolution", scores=(2, 4, 4)),
        # v2 diagnosis: (4, 5, 5) -> total 14; (5, 5, 4) -> total 14
        _make_record("v2_d1", "v2", "diagnosis", scores=(4, 5, 5)),
        _make_record("v2_d2", "v2", "diagnosis", scores=(5, 5, 4)),
        # v2 resolution: (5, 5, 5) -> total 15
        _make_record("v2_r1", "v2", "resolution", scores=(5, 5, 5)),
        # Registro não avaliado (deve ser ignorado)
        _make_record("v2_none", "v2", "diagnosis", scores=None),
    ]

    overall = web_app_module.aggregate_rubric_metrics(records)
    assert overall["evaluated_count"] == 6

    # v1: clarity=(3+4+2)/3=3.0, fit=(2+3+4)/3=3.0, act=(3+2+4)/3=3.0, total=(8+9+10)/3=9.0
    assert overall["by_version"]["v1"] == {
        "count": 3,
        "clarity_avg": 3.0,
        "customer_fit_avg": 3.0,
        "actionability_avg": 3.0,
        "total_avg": 9.0,
    }
    # v2: clarity=(4+5+5)/3=4.67, fit=(5+5+5)/3=5.0, act=(5+4+5)/3=4.67, total=(14+14+15)/3=14.33
    assert overall["by_version"]["v2"] == {
        "count": 3,
        "clarity_avg": 4.67,
        "customer_fit_avg": 5.0,
        "actionability_avg": 4.67,
        "total_avg": 14.33,
    }
    assert overall["deltas_v2_vs_v1"] == {
        "clarity_avg": 1.67,
        "customer_fit_avg": 2.0,
        "actionability_avg": 1.67,
        "total_avg": 5.33,
    }

    cross = overall["by_version_and_type"]
    assert len(cross) == 4
    assert [(row["prompt_version"], row["content_type"]) for row in cross] == [
        ("v1", "diagnosis"),
        ("v1", "resolution"),
        ("v2", "diagnosis"),
        ("v2", "resolution"),
    ]
    assert cross[0] == {
        "prompt_version": "v1",
        "content_type": "diagnosis",
        "count": 2,
        "clarity_avg": 3.5,
        "customer_fit_avg": 2.5,
        "actionability_avg": 2.5,
        "total_avg": 8.5,
    }

    diag_only = web_app_module.aggregate_rubric_metrics(records, content_type="diagnosis")
    assert diag_only["evaluated_count"] == 4
    assert diag_only["by_version"]["v1"]["total_avg"] == 8.5
    assert diag_only["by_version"]["v2"]["total_avg"] == 14.0
    assert diag_only["deltas_v2_vs_v1"] == {
        "clarity_avg": 1.0,
        "customer_fit_avg": 2.5,
        "actionability_avg": 2.0,
        "total_avg": 5.5,
    }
    assert len(diag_only["by_version_and_type"]) == 2

    flow_empty = web_app_module.aggregate_rubric_metrics(records, content_type="flow_summary")
    assert flow_empty["evaluated_count"] == 0
    assert flow_empty["deltas_v2_vs_v1"] is None


def test_extract_ticket_identifier() -> None:
    """Valida extração de [TKT-XXX] e fallback para 'Personalizado'."""
    assert (
        web_app_module.extract_ticket_identifier("[TKT-102] Erro 504 no webhook de pagamentos")
        == "TKT-102"
    )
    assert (
        web_app_module.extract_ticket_identifier("Relato referenciando [TKT-999] no meio do texto")
        == "TKT-999"
    )
    assert (
        web_app_module.extract_ticket_identifier("Cliente relata lentidão ao gerar relatório")
        == "Personalizado"
    )


def test_build_ticket_analytics_rows_and_fallback() -> None:
    """Valida cruzamento ExecutionRecord x CustomerProfile, rubrica e fallback desconhecido."""
    profiles = web_app_module.load_profiles_data()
    long_ticket = "[TKT-101] " + ("Descrição longa de teste " * 5)

    rec_evaluated = _make_record("k_eval", "v2", "diagnosis", scores=(4, 5, 5)).model_copy(
        update={"customer_id": "cust_001", "ticket": long_ticket, "latency_ms": 1420.5}
    )
    rec_unevaluated = _make_record("k_uneval", "v1", "resolution", scores=None).model_copy(
        update={"customer_id": "cust_antigo_999", "ticket": "Relato livre sem código"}
    )

    rows = web_app_module.build_ticket_analytics_rows(
        [rec_evaluated, rec_unevaluated],
        profiles,
    )
    assert len(rows) == 2

    row_eval = rows[0]
    assert row_eval["ticket_id"] == "TKT-101"
    assert len(row_eval["ticket_preview"]) == 65
    assert row_eval["ticket_full"] == long_ticket
    assert row_eval["customer_name"] == "Mariana Silva"
    assert row_eval["technical_level"] == "leigo"
    assert row_eval["communication_style"] == "didático"
    assert row_eval["channel"] == "chat"
    assert row_eval["latency_ms"] == 1420.5
    assert row_eval["is_evaluated"] is True
    assert row_eval["clarity_score"] == 4
    assert row_eval["customer_fit_score"] == 5
    assert row_eval["actionability_score"] == 5
    assert row_eval["total_score"] == 14

    row_unknown = rows[1]
    assert row_unknown["ticket_id"] == "Personalizado"
    assert row_unknown["customer_name"] == "desconhecido"
    assert row_unknown["technical_level"] == "desconhecido"
    assert row_unknown["communication_style"] == "desconhecido"
    assert row_unknown["channel"] == "desconhecido"
    assert row_unknown["latency_ms"] is None
    assert row_unknown["is_evaluated"] is False
    assert row_unknown["clarity_score"] is None
    assert row_unknown["total_score"] is None

    csv_text = web_app_module._export_analytics_csv(rows)
    assert "latency_ms" in csv_text.splitlines()[0]
    assert "1420.5" in csv_text


def test_summarize_by_ticket_grouping_and_averages() -> None:
    """Valida agrupamento por ticket_id e médias separadas v1 vs v2."""
    profiles = web_app_module.load_profiles_data()
    records = [
        _make_record("a1", "v1", "diagnosis", scores=(3, 2, 3)).model_copy(
            update={"ticket": "[TKT-102] Timeout webhook"}
        ),
        _make_record("a2", "v2", "diagnosis", scores=(4, 5, 5)).model_copy(
            update={"ticket": "[TKT-102] Timeout webhook"}
        ),
        _make_record("a3", "v2", "resolution", scores=(5, 5, 5)).model_copy(
            update={"ticket": "[TKT-102] Timeout webhook"}
        ),
        _make_record("a4", "v1", "clarification", scores=None).model_copy(
            update={"ticket": "Relato livre sem avaliação"}
        ),
    ]
    rows = web_app_module.build_ticket_analytics_rows(records, profiles)
    summary = web_app_module.summarize_by_ticket(rows)

    by_id = {item["ticket_id"]: item for item in summary}
    assert set(by_id.keys()) == {"TKT-102", "Personalizado"}

    tkt_102 = by_id["TKT-102"]
    assert tkt_102["total_executions"] == 3
    assert tkt_102["evaluated_count"] == 3
    # (8 + 14 + 15) / 3 = 12.33
    assert tkt_102["avg_total_score"] == 12.33
    assert tkt_102["v1_avg_total"] == 8.0
    assert tkt_102["v2_avg_total"] == 14.5

    custom = by_id["Personalizado"]
    assert custom["total_executions"] == 1
    assert custom["evaluated_count"] == 0
    assert custom["avg_total_score"] is None
    assert custom["v1_avg_total"] is None
    assert custom["v2_avg_total"] is None


def test_root_cli_shortcut_exists() -> None:
    """Confirma que o atalho cli.py existe na raiz e referencia src.cli.main."""
    root_cli = PROJECT_ROOT / "cli.py"
    assert root_cli.is_file()
    content = root_cli.read_text(encoding="utf-8")
    assert "from src.cli import main" in content


def _run_standalone() -> int:
    """Executa todos os checks da Etapa 4 offline e exibe painel verde com rich."""
    from rich.panel import Panel

    console = Console()
    passed: list[str] = []

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_tmp = Path(tmp_dir)
            checks = [
                (
                    "Carregamento de perfis e tickets pelos helpers da CLI",
                    test_cli_loaders_profiles_and_tickets,
                ),
                (
                    "Execução da CLI não-interativa, caminho salvo, ASCII e cache hit",
                    lambda: test_cli_non_interactive_execution_and_cache(base_tmp / "cli_run"),
                ),
                (
                    "Fluxo de rubrica via CLI e tratamento amigável de erros",
                    lambda: test_cli_evaluation_flow_and_error_handling(base_tmp / "cli_eval"),
                ),
                (
                    "Helpers offline do Streamlit (geração, v1 vs v2, rubrica e histórico)",
                    lambda: test_web_app_helpers_offline(base_tmp / "web"),
                ),
                (
                    "Agregação analítica da rubrica (vazio e sem avaliações)",
                    test_aggregate_rubric_metrics_empty_and_unevaluated,
                ),
                (
                    "Agregação analítica da rubrica (médias, deltas v2 vs v1 e filtro)",
                    test_aggregate_rubric_metrics_averages_deltas_and_filter,
                ),
                (
                    "Extração de identificador de ticket (TKT-XXX e Personalizado)",
                    test_extract_ticket_identifier,
                ),
                (
                    "Cruzamento analítico ExecutionRecord x CustomerProfile",
                    test_build_ticket_analytics_rows_and_fallback,
                ),
                (
                    "Resumo agrupado por ticket (v1 vs v2)",
                    test_summarize_by_ticket_grouping_and_averages,
                ),
                (
                    "Atalho cli.py na raiz do projeto",
                    test_root_cli_shortcut_exists,
                ),
            ]
            for label, check_fn in checks:
                check_fn()
                passed.append(f"[bold green][OK][/bold green] {label}")
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        console.print(
            Panel(
                f"[bold red]Falha no checkpoint da Etapa 4:[/bold red]\n{exc}",
                title="Checkpoint Etapa 4 - ERRO",
                border_style="red",
            )
        )
        return 1

    summary = "\n".join(passed)
    console.print(
        Panel(
            f"{summary}\n\n[bold green]Todos os {len(passed)} checks da Etapa 4 foram validados com sucesso![/bold green]",
            title="Checkpoint Etapa 4 - SUCESSO",
            border_style="green",
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(_run_standalone())
