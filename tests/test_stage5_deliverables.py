from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import ExecutionRecord, RubricEvaluation  # noqa: E402

SAMPLES_DIR = PROJECT_ROOT / "samples"
README_PATH = PROJECT_ROOT / "README.md"
NOTES_PATH = PROJECT_ROOT / "PROMPT_ENGINEERING_NOTES.md"

REQUIRED_SAMPLE_FILES = (
    "comparison_v1_vs_v2_diagnosis.json",
    "sample_resolution_advanced.json",
    "sample_clarification_layman.json",
    "sample_flow_summary_ascii.json",
)


def _load_sample_records(filename: str) -> list[ExecutionRecord]:
    path = SAMPLES_DIR / filename
    assert path.is_file(), f"Arquivo de amostra obrigatório não encontrado: samples/{filename}"
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(raw, list) and len(raw) >= 1, (
        f"samples/{filename} deve conter uma lista JSON não vazia."
    )
    return [ExecutionRecord.model_validate(item) for item in raw]


def test_samples_exist_and_validate_with_pydantic() -> None:
    """Valida existência e aderência ao esquema ExecutionRecord dos 4 arquivos em samples/."""
    assert SAMPLES_DIR.is_dir(), "Diretório samples/ não encontrado."

    seen_content_types: set[str] = set()
    for filename in REQUIRED_SAMPLE_FILES:
        records = _load_sample_records(filename)
        for rec in records:
            assert isinstance(rec.output, (str, dict))
            if isinstance(rec.output, str):
                assert len(rec.output.strip()) > 30
            assert isinstance(rec.evaluation, RubricEvaluation), (
                f"Registro em samples/{filename} deve conter evaluation preenchido."
            )
            seen_content_types.add(rec.content_type)

    assert seen_content_types == {"diagnosis", "resolution", "clarification", "flow_summary"}

    flow_records = _load_sample_records("sample_flow_summary_ascii.json")
    flow_output = str(flow_records[0].output)
    assert flow_records[0].content_type == "flow_summary"
    assert "+" in flow_output and "-" in flow_output and "|" in flow_output
    assert "[SIM]" in flow_output and "[NÃO]" in flow_output


def test_comparison_sample_is_controlled_and_evaluated() -> None:
    """Confirma comparação justa (v1 vs v2) sob mesmo perfil, ticket, modelo e temperatura."""
    records = _load_sample_records("comparison_v1_vs_v2_diagnosis.json")
    assert len(records) == 2, "comparison_v1_vs_v2_diagnosis.json deve conter exatamente 2 execuções."

    by_version = {rec.prompt_version: rec for rec in records}
    assert set(by_version.keys()) == {"v1", "v2"}

    v1_rec = by_version["v1"]
    v2_rec = by_version["v2"]

    assert v1_rec.customer_id == v2_rec.customer_id
    assert v1_rec.ticket == v2_rec.ticket
    assert v1_rec.content_type == "diagnosis"
    assert v2_rec.content_type == "diagnosis"
    assert v1_rec.provider == v2_rec.provider
    assert v1_rec.model == v2_rec.model
    assert v1_rec.temperature == v2_rec.temperature
    assert v1_rec.cache_key != v2_rec.cache_key
    assert v1_rec.evaluation is not None
    assert v2_rec.evaluation is not None


def test_readme_contains_architecture_and_run_instructions() -> None:
    """Valida existência e seções essenciais de README.md."""
    assert README_PATH.is_file(), "README.md não encontrado na raiz do projeto."
    content = README_PATH.read_text(encoding="utf-8")
    assert len(content) > 1500, "README.md está excessivamente curto."

    required_snippets = (
        "requirements.txt",
        ".env.example",
        "cli.py",
        "streamlit run src/web_app.py",
        "pytest",
        "data/history.json",
        "+---",
    )
    for snippet in required_snippets:
        assert snippet in content, f"Trecho obrigatório ausente em README.md: '{snippet}'"


def test_prompt_engineering_notes_covers_engine_versions_and_rubric() -> None:
    """Valida existência e cobertura técnica de PROMPT_ENGINEERING_NOTES.md."""
    assert NOTES_PATH.is_file(), "PROMPT_ENGINEERING_NOTES.md não encontrado na raiz do projeto."
    content = NOTES_PATH.read_text(encoding="utf-8")
    assert len(content) > 1500, "PROMPT_ENGINEERING_NOTES.md está excessivamente curto."

    required_snippets = (
        "src/prompts/engine.py",
        "[PROMPT_METADATA]",
        "<ticket>",
        "NO_RAW_COT_RULE",
        "v1",
        "v2",
        "diagnosis",
        "resolution",
        "clarification",
        "flow_summary",
        "comparison_v1_vs_v2_diagnosis.json",
        "Clareza",
        "Acionabilidade",
    )
    for snippet in required_snippets:
        assert snippet in content, (
            f"Trecho obrigatório ausente em PROMPT_ENGINEERING_NOTES.md: '{snippet}'"
        )


def test_no_secret_api_keys_leaked_in_deliverables() -> None:
    """Garante que nenhuma chave real da API Google (padrão AIza...) foi versionada."""
    secret_pattern = re.compile(r"AIza[0-9A-Za-z_\-]{20,}")
    files_to_scan = [
        PROJECT_ROOT / ".env.example",
        README_PATH,
        NOTES_PATH,
        *(SAMPLES_DIR / name for name in REQUIRED_SAMPLE_FILES),
    ]
    for path in files_to_scan:
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            assert not secret_pattern.search(text), f"Possível segredo vazado em {path.name}!"


def _run_standalone() -> int:
    """Executa a suíte da Etapa 5 offline e exibe painel verde com rich."""
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    checks = [
        ("Validação Pydantic dos 4 arquivos em samples/", test_samples_exist_and_validate_with_pydantic),
        ("Comparação controlada justa (v1 vs v2 + rubrica)", test_comparison_sample_is_controlled_and_evaluated),
        ("Estrutura, diagrama ASCII e instruções no README.md", test_readme_contains_architecture_and_run_instructions),
        ("Cobertura técnica em PROMPT_ENGINEERING_NOTES.md", test_prompt_engineering_notes_covers_engine_versions_and_rubric),
        ("Auditoria de segurança (nenhuma API key vazada)", test_no_secret_api_keys_leaked_in_deliverables),
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
                f"[bold red]Falha no checkpoint da Etapa 5:[/bold red]\n{exc}",
                title="Checkpoint Etapa 5 - ERRO",
                border_style="red",
            )
        )
        return 1

    summary = "\n".join(passed)
    console.print(
        Panel(
            f"{summary}\n\n[bold green]Todos os {len(passed)} checks da Etapa 5 foram validados com sucesso![/bold green]",
            title="Checkpoint Etapa 5 - SUCESSO",
            border_style="green",
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(_run_standalone())
