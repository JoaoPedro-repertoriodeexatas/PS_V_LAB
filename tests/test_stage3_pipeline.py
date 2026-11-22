from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import CustomerProfile  # noqa: E402
from src.pipeline import (  # noqa: E402
    build_rubric_evaluation,
    evaluate_execution,
    execute_pipeline,
    generate_cache_key,
)
from src.services.llm_client import BaseLLMClient, GeminiClient, PLACEHOLDER_API_KEY  # noqa: E402
from src.services.storage import JsonHistoryStorage  # noqa: E402

SAMPLE_PROFILE = CustomerProfile(
    customer_id="cust_001",
    name="Mariana Silva",
    technical_level="leigo",
    communication_style="didático",
    channel="chat",
)

SAMPLE_TICKET = (
    "Ao tentar exportar o relatório consolidado de vendas dos últimos 90 dias, "
    "a tela carrega por 40 segundos e exibe erro genérico."
)

ASCII_FLOWCHART_SAMPLE = (
    "    +---------------------------+\n"
    "    | Relatório > 30 dias?      |\n"
    "    +-------------+-------------+\n"
    "                  |\n"
    "          +-------+-------+\n"
    "          |               |\n"
    "       [SIM]           [NÃO]\n"
    "          v               v\n"
    "    +-----------+   +-----------+\n"
    "    | Dividir   |   | Escalar   |\n"
    "    | período   |   | N2        |\n"
    "    +-----------+   +-----------+"
)


class FakeLLMClient(BaseLLMClient):
    """Dublê offline de cliente LLM para validar cache e pipeline sem chamadas de rede."""

    def __init__(
        self,
        provider: str = "google",
        model: str = "gemini-2.5-flash",
        temperature: float = 0.2,
        response_text: str | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._temperature = temperature
        self._response_text = response_text
        self.call_count = 0
        self.prompts_received: list[str] = []

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    @property
    def temperature(self) -> float:
        return self._temperature

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        self.prompts_received.append(prompt)
        if self._response_text is not None:
            return self._response_text
        return f"Resposta simulada #{self.call_count} para prompt ({len(prompt)} chars)."


def test_generate_cache_key_determinism_and_variations() -> None:
    """Garante determinismo do hash SHA-256 e sensibilidade a cada parâmetro relevante."""
    base_key = generate_cache_key(
        customer_id="cust_001",
        ticket=SAMPLE_TICKET,
        content_type="diagnosis",
        prompt_version="v2",
        provider="google",
        model="gemini-2.5-flash",
        temperature=0.2,
    )
    same_key_normalized = generate_cache_key(
        customer_id=" cust_001 ",
        ticket=f"  {SAMPLE_TICKET}\n ",
        content_type="diagnosis",
        prompt_version="V2",
        provider="GOOGLE",
        model="gemini-2.5-flash",
        temperature=0.2000001,
    )
    assert len(base_key) == 64
    assert base_key == same_key_normalized

    variations = [
        generate_cache_key("cust_002", SAMPLE_TICKET, "diagnosis", "v2", "google", "gemini-2.5-flash", 0.2),
        generate_cache_key("cust_001", f"{SAMPLE_TICKET} extra", "diagnosis", "v2", "google", "gemini-2.5-flash", 0.2),
        generate_cache_key("cust_001", SAMPLE_TICKET, "resolution", "v2", "google", "gemini-2.5-flash", 0.2),
        generate_cache_key("cust_001", SAMPLE_TICKET, "diagnosis", "v1", "google", "gemini-2.5-flash", 0.2),
        generate_cache_key("cust_001", SAMPLE_TICKET, "diagnosis", "v2", "google", "gemini-2.5-flash", 0.7),
    ]
    for varied_key in variations:
        assert varied_key != base_key


def test_pipeline_cache_miss_then_cache_hit(tmp_path: Path) -> None:
    """Valida cache miss na 1ª execução e cache hit sem nova chamada à LLM na 2ª execução."""
    history_file = tmp_path / "history.json"
    storage = JsonHistoryStorage(history_file)
    fake_llm = FakeLLMClient()

    first = execute_pipeline(
        profile=SAMPLE_PROFILE,
        ticket=SAMPLE_TICKET,
        content_type="diagnosis",
        prompt_version="v2",
        llm_client=fake_llm,
        storage=storage,
    )
    assert first.from_cache is False
    assert first.storage_path == history_file
    assert history_file.is_file()
    assert fake_llm.call_count == 1

    second = execute_pipeline(
        profile=SAMPLE_PROFILE,
        ticket=SAMPLE_TICKET,
        content_type="diagnosis",
        prompt_version="v2",
        llm_client=fake_llm,
        storage=storage,
    )
    assert second.from_cache is True
    assert fake_llm.call_count == 1
    assert second.record.cache_key == first.record.cache_key
    assert second.record.output == first.record.output
    assert len(storage.load_all()) == 1


def test_pipeline_force_refresh_ignores_cache(tmp_path: Path) -> None:
    """Confirma que force_refresh=True ignora o cache, gera nova chamada e atualiza hit recente."""
    storage = JsonHistoryStorage(tmp_path / "history.json")
    fake_llm = FakeLLMClient()

    first = execute_pipeline(
        profile=SAMPLE_PROFILE,
        ticket=SAMPLE_TICKET,
        content_type="resolution",
        prompt_version="v1",
        llm_client=fake_llm,
        storage=storage,
    )
    refreshed = execute_pipeline(
        profile=SAMPLE_PROFILE,
        ticket=SAMPLE_TICKET,
        content_type="resolution",
        prompt_version="v1",
        force_refresh=True,
        llm_client=fake_llm,
        storage=storage,
    )
    cached_after_refresh = execute_pipeline(
        profile=SAMPLE_PROFILE,
        ticket=SAMPLE_TICKET,
        content_type="resolution",
        prompt_version="v1",
        force_refresh=False,
        llm_client=fake_llm,
        storage=storage,
    )

    assert first.from_cache is False
    assert refreshed.from_cache is False
    assert fake_llm.call_count == 2
    assert len(storage.load_all()) == 2
    assert cached_after_refresh.from_cache is True
    assert cached_after_refresh.record.output == refreshed.record.output


def test_evaluate_execution_persists_rubric_and_preserves_ascii_indentation(
    tmp_path: Path,
) -> None:
    """Valida persistência da rubrica e preservação exata de indentação em fluxogramas ASCII."""
    storage = JsonHistoryStorage(tmp_path / "history.json")
    fake_llm = FakeLLMClient(response_text=ASCII_FLOWCHART_SAMPLE)

    result = execute_pipeline(
        profile=SAMPLE_PROFILE,
        ticket=SAMPLE_TICKET,
        content_type="flow_summary",
        prompt_version="v2",
        llm_client=fake_llm,
        storage=storage,
    )
    assert result.record.output == ASCII_FLOWCHART_SAMPLE

    evaluation = build_rubric_evaluation(
        clarity_score=5,
        clarity_justification="Fluxograma bem alinhado e decisões explícitas.",
        customer_fit_score=4,
        customer_fit_justification="Linguagem simples e adequada ao contexto.",
        actionability_score=5,
        actionability_justification="Rota clara para resolução e escalonamento.",
    )

    updated = evaluate_execution(
        cache_key=result.record.cache_key,
        evaluation=evaluation,
        storage=storage,
    )
    assert updated.evaluation is not None
    assert updated.evaluation.clarity.score == 5

    reloaded = storage.load_all()
    assert len(reloaded) == 1
    assert reloaded[0].evaluation is not None
    assert reloaded[0].evaluation.actionability.score == 5
    assert reloaded[0].output == ASCII_FLOWCHART_SAMPLE
    assert isinstance(reloaded[0].output, str)
    assert reloaded[0].output.startswith("    +---")

    with pytest.raises(KeyError):
        evaluate_execution(
            cache_key="chave-inexistente",
            evaluation=evaluation,
            storage=storage,
        )


def test_gemini_client_rejects_missing_or_placeholder_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confirma ValueError em GeminiClient quando GEMINI_API_KEY está ausente ou placeholder."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiClient(load_env=False)

    monkeypatch.setenv("GEMINI_API_KEY", PLACEHOLDER_API_KEY)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiClient(load_env=False)

    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiClient(load_env=False)


def _run_standalone() -> int:
    """Executa a suíte da Etapa 3 offline e exibe painel de checkpoint com rich."""
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    passed: list[str] = []

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_tmp = Path(tmp_dir)
            checks = [
                (
                    "Determinismo e sensibilidade de generate_cache_key",
                    test_generate_cache_key_determinism_and_variations,
                ),
                (
                    "Fluxo de cache miss seguido de cache hit sem chamada extra",
                    lambda: test_pipeline_cache_miss_then_cache_hit(base_tmp / "hit"),
                ),
                (
                    "Parâmetro force_refresh=True ignora cache",
                    lambda: test_pipeline_force_refresh_ignores_cache(base_tmp / "refresh"),
                ),
                (
                    "Persistência de RubricEvaluation preservando fluxograma ASCII",
                    lambda: test_evaluate_execution_persists_rubric_and_preserves_ascii_indentation(
                        base_tmp / "eval"
                    ),
                ),
                (
                    "Validação defensiva de GEMINI_API_KEY em GeminiClient",
                    lambda: _run_with_monkeypatch(
                        test_gemini_client_rejects_missing_or_placeholder_api_key
                    ),
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
                f"[bold red]Falha no checkpoint da Etapa 3:[/bold red]\n{exc}",
                title="Checkpoint Etapa 3 - ERRO",
                border_style="red",
            )
        )
        return 1

    summary = "\n".join(passed)
    console.print(
        Panel(
            f"{summary}\n\n[bold green]Todos os {len(passed)} checks da Etapa 3 foram validados com sucesso![/bold green]",
            title="Checkpoint Etapa 3 - SUCESSO",
            border_style="green",
        )
    )
    return 0


def _run_with_monkeypatch(fn) -> None:
    mp = pytest.MonkeyPatch()
    try:
        fn(mp)
    finally:
        mp.undo()


if __name__ == "__main__":
    sys.exit(_run_standalone())
