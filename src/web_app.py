from __future__ import annotations

from collections import defaultdict
import csv
import html
import io
import json
from pathlib import Path
import re
from statistics import fmean
import sys
from typing import Any, get_args

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cli import (  # noqa: E402
    CONTENT_TYPE_LABELS,
    CONTENT_TYPES,
    DEFAULT_PROFILES_PATH,
    DEFAULT_TICKETS_PATH,
    load_profiles,
    load_tickets,
)
from src.models import (  # noqa: E402
    Channel,
    ContentType,
    CustomerProfile,
    ExecutionRecord,
    TechnicalLevel,
    TicketItem,
)
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
    GeminiClient,
    JsonHistoryStorage,
    LLMServiceError,
)

_TICKET_ID_PATTERN = re.compile(r"\[(TKT-[A-Za-z0-9_-]+)\]")
_TECHNICAL_LEVELS: tuple[str, ...] = get_args(TechnicalLevel)
_CHANNELS: tuple[str, ...] = get_args(Channel)

_ARTIFACT_GUIDE: tuple[tuple[ContentType, str], ...] = (
    ("diagnosis", "Hipóteses, evidências, lacunas e conclusão"),
    ("resolution", "Passo a passo acionável calibrado ao perfil"),
    ("clarification", "3 a 5 perguntas priorizadas com justificativa"),
    ("flow_summary", "Fluxograma ASCII com desvios [SIM] / [NÃO]"),
)

_CUSTOM_CSS = """
<style>
  .block-container {
    padding-top: 1.4rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1440px;
  }

  [data-testid="stVerticalBlockBorderWrapper"] {
    border-color: rgba(148, 163, 184, 0.22) !important;
    border-radius: 10px !important;
    background-color: rgba(22, 27, 34, 0.45);
  }

  .console-header {
    padding: 0.25rem 0 0.85rem 0;
    border-bottom: 1px solid rgba(148, 163, 184, 0.18);
    margin-bottom: 1.1rem;
  }

  .console-kicker {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #38BDF8;
    margin-bottom: 0.2rem;
  }

  .console-title {
    font-size: 1.65rem;
    font-weight: 700;
    color: #E6EDF3;
    margin: 0;
    line-height: 1.25;
  }

  .console-subtitle {
    font-size: 0.9rem;
    color: #94A3B8;
    margin: 0.3rem 0 0 0;
  }

  .badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin: 0.4rem 0 0.85rem 0;
  }

  .badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.22rem 0.65rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    line-height: 1.3;
    border: 1px solid transparent;
  }

  .badge-label {
    opacity: 0.78;
    font-weight: 500;
  }

  .badge-level {
    background: rgba(56, 189, 248, 0.14);
    color: #7DD3FC;
    border-color: rgba(56, 189, 248, 0.32);
  }

  .badge-style {
    background: rgba(245, 158, 11, 0.14);
    color: #FCD34D;
    border-color: rgba(245, 158, 11, 0.32);
  }

  .badge-channel {
    background: rgba(167, 139, 250, 0.14);
    color: #C4B5FD;
    border-color: rgba(167, 139, 250, 0.32);
  }

  .badge-cache-hit {
    background: rgba(16, 185, 129, 0.16);
    color: #34D399;
    border-color: rgba(16, 185, 129, 0.42);
  }

  .badge-api-new {
    background: rgba(56, 189, 248, 0.16);
    color: #38BDF8;
    border-color: rgba(56, 189, 248, 0.42);
  }

  .badge-v1 {
    background: rgba(245, 158, 11, 0.14);
    color: #FBBF24;
    border-color: rgba(245, 158, 11, 0.35);
  }

  .badge-v2 {
    background: rgba(16, 185, 129, 0.14);
    color: #34D399;
    border-color: rgba(16, 185, 129, 0.35);
  }

  .status-banner {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    padding: 0.65rem 0.9rem;
    border-radius: 8px;
    margin-bottom: 0.65rem;
    font-size: 0.84rem;
  }

  .status-banner-cache {
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.32);
    color: #D1FAE5;
  }

  .status-banner-api {
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.32);
    color: #E0F2FE;
  }

  .meta-bar {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.75rem;
    color: #94A3B8;
    background: rgba(14, 17, 23, 0.65);
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 6px;
    padding: 0.45rem 0.7rem;
    margin-bottom: 0.75rem;
    word-break: break-all;
  }

  .version-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 0.55rem;
    margin-bottom: 0.75rem;
    border-bottom: 1px solid rgba(148, 163, 184, 0.18);
  }

  .version-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #E6EDF3;
  }

  .empty-state {
    text-align: center;
    padding: 2.6rem 1.4rem;
    color: #94A3B8;
  }

  .empty-state-title {
    font-size: 1.08rem;
    font-weight: 600;
    color: #E6EDF3;
    margin-bottom: 0.45rem;
  }

  .sidebar-kv {
    font-size: 0.82rem;
    color: #CBD5E1;
    margin: 0.28rem 0;
  }

  .sidebar-kv code {
    color: #7DD3FC;
    background: rgba(56, 189, 248, 0.1);
    padding: 0.1rem 0.35rem;
    border-radius: 4px;
  }

  .artifact-item {
    font-size: 0.8rem;
    margin: 0.35rem 0;
    color: #94A3B8;
    line-height: 1.35;
  }

  .artifact-item strong {
    color: #E6EDF3;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.78rem;
  }

  .ticket-preview {
    font-size: 0.83rem;
    color: #CBD5E1;
    background: rgba(14, 17, 23, 0.65);
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 6px;
    padding: 0.6rem 0.75rem;
    margin-top: 0.25rem;
    line-height: 1.45;
  }
</style>
"""


def load_profiles_data(path: Path = DEFAULT_PROFILES_PATH) -> list[CustomerProfile]:
    """Carrega os perfis de cliente para a interface web."""
    return load_profiles(path)


def load_tickets_data(path: Path = DEFAULT_TICKETS_PATH) -> list[TicketItem]:
    """Carrega os tickets pré-cadastrados para a interface web."""
    return load_tickets(path)


def format_output_text(output: dict[str, Any] | str) -> str:
    """Converte a saída do modelo em texto legível preservando indentação ASCII."""
    if isinstance(output, str):
        return output
    return json.dumps(output, indent=2, ensure_ascii=False)


def run_generation(
    profile: CustomerProfile,
    ticket: str,
    content_type: ContentType,
    prompt_version: str = "v2",
    force_refresh: bool = False,
    llm_client: BaseLLMClient | None = None,
    storage: JsonHistoryStorage | None = None,
) -> tuple[PipelineExecutionResult, str]:
    """Executa o pipeline para uma combinação e retorna (resultado, prompt_montado)."""
    prompt_text = get_prompt_for_content_type(
        content_type=content_type,
        profile=profile,
        ticket=ticket,
        version=prompt_version,
    )
    result = execute_pipeline(
        profile=profile,
        ticket=ticket,
        content_type=content_type,
        prompt_version=prompt_version,
        force_refresh=force_refresh,
        llm_client=llm_client,
        storage=storage,
    )
    return result, prompt_text


def run_comparison(
    profile: CustomerProfile,
    ticket: str,
    content_type: ContentType,
    force_refresh: bool = False,
    llm_client: BaseLLMClient | None = None,
    storage: JsonHistoryStorage | None = None,
) -> dict[str, tuple[PipelineExecutionResult, str]]:
    """Executa v1 e v2 sob exatamente as mesmas configurações de cliente e armazenamento."""
    shared_storage = storage or JsonHistoryStorage()
    shared_client = llm_client or GeminiClient()

    results: dict[str, tuple[PipelineExecutionResult, str]] = {}
    for version in ("v1", "v2"):
        results[version] = run_generation(
            profile=profile,
            ticket=ticket,
            content_type=content_type,
            prompt_version=version,
            force_refresh=force_refresh,
            llm_client=shared_client,
            storage=shared_storage,
        )
    return results


def save_rubric_rating(
    cache_key: str,
    clarity_score: int,
    clarity_justification: str,
    customer_fit_score: int,
    customer_fit_justification: str,
    actionability_score: int,
    actionability_justification: str,
    storage: JsonHistoryStorage | None = None,
) -> ExecutionRecord:
    """Valida via Pydantic e persiste a rubrica de 3 critérios no histórico JSON."""
    evaluation = build_rubric_evaluation(
        clarity_score=clarity_score,
        clarity_justification=clarity_justification,
        customer_fit_score=customer_fit_score,
        customer_fit_justification=customer_fit_justification,
        actionability_score=actionability_score,
        actionability_justification=actionability_justification,
    )
    return evaluate_execution(
        cache_key=cache_key,
        evaluation=evaluation,
        storage=storage,
    )


def summarize_history(records: list[ExecutionRecord]) -> dict[str, int]:
    """Calcula métricas resumidas do histórico de execuções."""
    return {
        "total": len(records),
        "evaluated": sum(1 for r in records if r.evaluation is not None),
        "v1": sum(1 for r in records if r.prompt_version == "v1"),
        "v2": sum(1 for r in records if r.prompt_version == "v2"),
    }


def filter_history_records(
    records: list[ExecutionRecord],
    customer_id: str | None = None,
    content_type: str | None = None,
    prompt_version: str | None = None,
    only_evaluated: bool = False,
) -> list[ExecutionRecord]:
    """Filtra registros do histórico para auditoria e comparação."""
    filtered: list[ExecutionRecord] = []
    for record in records:
        if customer_id and record.customer_id != customer_id:
            continue
        if content_type and record.content_type != content_type:
            continue
        if prompt_version and record.prompt_version != prompt_version:
            continue
        if only_evaluated and record.evaluation is None:
            continue
        filtered.append(record)
    return filtered


def export_history_json(records: list[ExecutionRecord]) -> str:
    """Serializa uma lista de ExecutionRecord em JSON formatado UTF-8."""
    return json.dumps(
        [record.model_dump(mode="json") for record in records],
        indent=2,
        ensure_ascii=False,
    )


def _compute_rubric_group_stats(group: list[ExecutionRecord]) -> dict[str, int | float]:
    """Calcula contagem e médias arredondadas em 2 casas decimais para um grupo avaliado."""
    if not group:
        return {
            "count": 0,
            "clarity_avg": 0.0,
            "customer_fit_avg": 0.0,
            "actionability_avg": 0.0,
            "total_avg": 0.0,
        }

    clarity_scores = [r.evaluation.clarity.score for r in group if r.evaluation is not None]
    fit_scores = [r.evaluation.customer_fit.score for r in group if r.evaluation is not None]
    act_scores = [r.evaluation.actionability.score for r in group if r.evaluation is not None]
    totals = [c + f + a for c, f, a in zip(clarity_scores, fit_scores, act_scores)]

    return {
        "count": len(group),
        "clarity_avg": round(fmean(clarity_scores), 2),
        "customer_fit_avg": round(fmean(fit_scores), 2),
        "actionability_avg": round(fmean(act_scores), 2),
        "total_avg": round(fmean(totals), 2),
    }


def aggregate_rubric_metrics(
    records: list[ExecutionRecord],
    content_type: str | None = None,
) -> dict[str, Any]:
    """Agrega médias da rubrica por versão, por (versão, tipo) e calcula deltas v2 vs v1."""
    evaluated = [
        r
        for r in records
        if r.evaluation is not None
        and (content_type is None or r.content_type == content_type)
    ]

    grouped_by_version: dict[str, list[ExecutionRecord]] = defaultdict(list)
    grouped_cross: dict[tuple[str, str], list[ExecutionRecord]] = defaultdict(list)

    for rec in evaluated:
        grouped_by_version[rec.prompt_version].append(rec)
        grouped_cross[(rec.prompt_version, rec.content_type)].append(rec)

    ordered_versions = ["v1", "v2"] + sorted(
        ver for ver in grouped_by_version if ver not in ("v1", "v2")
    )
    by_version: dict[str, dict[str, int | float]] = {
        ver: _compute_rubric_group_stats(grouped_by_version.get(ver, []))
        for ver in ordered_versions
    }

    by_version_and_type: list[dict[str, Any]] = [
        {
            "prompt_version": version,
            "content_type": ctype,
            **_compute_rubric_group_stats(group_records),
        }
        for (version, ctype), group_records in sorted(grouped_cross.items())
    ]

    v1_stats = by_version["v1"]
    v2_stats = by_version["v2"]
    if v1_stats["count"] > 0 and v2_stats["count"] > 0:
        deltas_v2_vs_v1: dict[str, float] | None = {
            "clarity_avg": round(
                float(v2_stats["clarity_avg"]) - float(v1_stats["clarity_avg"]), 2
            ),
            "customer_fit_avg": round(
                float(v2_stats["customer_fit_avg"]) - float(v1_stats["customer_fit_avg"]), 2
            ),
            "actionability_avg": round(
                float(v2_stats["actionability_avg"]) - float(v1_stats["actionability_avg"]), 2
            ),
            "total_avg": round(
                float(v2_stats["total_avg"]) - float(v1_stats["total_avg"]), 2
            ),
        }
    else:
        deltas_v2_vs_v1 = None

    return {
        "evaluated_count": len(evaluated),
        "content_type_filter": content_type,
        "by_version": by_version,
        "by_version_and_type": by_version_and_type,
        "deltas_v2_vs_v1": deltas_v2_vs_v1,
    }


def extract_ticket_identifier(ticket_text: str) -> str:
    """Extrai código [TKT-XXX] do texto do ticket ou retorna 'Personalizado'."""
    match = _TICKET_ID_PATTERN.search(ticket_text)
    if match is not None:
        return match.group(1)
    return "Personalizado"


def build_ticket_analytics_rows(
    records: list[ExecutionRecord],
    profiles: list[CustomerProfile],
) -> list[dict[str, Any]]:
    """Cruza ExecutionRecord com CustomerProfile para análise interativa de tickets."""
    profile_by_id = {profile.customer_id: profile for profile in profiles}
    rows: list[dict[str, Any]] = []

    for rec in records:
        profile = profile_by_id.get(rec.customer_id)
        ev = rec.evaluation

        if ev is not None:
            clarity_score: int | None = ev.clarity.score
            customer_fit_score: int | None = ev.customer_fit.score
            actionability_score: int | None = ev.actionability.score
            total_score: int | None = (
                ev.clarity.score + ev.customer_fit.score + ev.actionability.score
            )
            is_evaluated = True
        else:
            clarity_score = None
            customer_fit_score = None
            actionability_score = None
            total_score = None
            is_evaluated = False

        rows.append(
            {
                "cache_key": rec.cache_key,
                "timestamp": rec.timestamp,
                "ticket_id": extract_ticket_identifier(rec.ticket),
                "ticket_preview": rec.ticket[:65],
                "ticket_full": rec.ticket,
                "customer_id": rec.customer_id,
                "customer_name": profile.name if profile is not None else "desconhecido",
                "technical_level": (
                    profile.technical_level if profile is not None else "desconhecido"
                ),
                "communication_style": (
                    profile.communication_style if profile is not None else "desconhecido"
                ),
                "channel": profile.channel if profile is not None else "desconhecido",
                "content_type": rec.content_type,
                "prompt_version": rec.prompt_version,
                "provider": rec.provider,
                "model": rec.model,
                "temperature": rec.temperature,
                "is_evaluated": is_evaluated,
                "clarity_score": clarity_score,
                "customer_fit_score": customer_fit_score,
                "actionability_score": actionability_score,
                "total_score": total_score,
            }
        )

    return rows


def summarize_by_ticket(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Agrupa linhas analíticas por ticket_id com totais e médias v1 vs v2."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["ticket_id"])].append(row)

    summary: list[dict[str, Any]] = []
    for ticket_id in sorted(grouped.keys()):
        group = grouped[ticket_id]
        evaluated_rows = [
            r for r in group if r["is_evaluated"] and r["total_score"] is not None
        ]
        v1_totals = [
            int(r["total_score"])
            for r in evaluated_rows
            if r["prompt_version"] == "v1"
        ]
        v2_totals = [
            int(r["total_score"])
            for r in evaluated_rows
            if r["prompt_version"] == "v2"
        ]

        summary.append(
            {
                "ticket_id": ticket_id,
                "total_executions": len(group),
                "evaluated_count": len(evaluated_rows),
                "avg_total_score": (
                    round(fmean(int(r["total_score"]) for r in evaluated_rows), 2)
                    if evaluated_rows
                    else None
                ),
                "v1_avg_total": round(fmean(v1_totals), 2) if v1_totals else None,
                "v2_avg_total": round(fmean(v2_totals), 2) if v2_totals else None,
            }
        )

    return summary


def _export_analytics_csv(rows: list[dict[str, Any]]) -> str:
    """Serializa as linhas analíticas cruzadas em CSV UTF-8 para download."""
    fieldnames = [
        "timestamp",
        "ticket_id",
        "customer_id",
        "customer_name",
        "technical_level",
        "communication_style",
        "channel",
        "content_type",
        "prompt_version",
        "provider",
        "model",
        "temperature",
        "is_evaluated",
        "clarity_score",
        "customer_fit_score",
        "actionability_score",
        "total_score",
        "cache_key",
        "ticket_preview",
        "ticket_full",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _inject_custom_css(st: Any) -> None:
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


def _render_badge_pills_raw(
    st: Any,
    technical_level: str,
    communication_style: str,
    channel: str,
) -> None:
    level = html.escape(technical_level)
    style = html.escape(communication_style)
    chan = html.escape(channel)
    st.markdown(
        (
            '<div class="badge-row">'
            f'<span class="badge-pill badge-level"><span class="badge-label">Nível</span> {level}</span>'
            f'<span class="badge-pill badge-style"><span class="badge-label">Estilo</span> {style}</span>'
            f'<span class="badge-pill badge-channel"><span class="badge-label">Canal</span> {chan}</span>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_profile_badges(st: Any, profile: CustomerProfile) -> None:
    _render_badge_pills_raw(
        st,
        profile.technical_level,
        profile.communication_style,
        profile.channel,
    )


def _render_ticket_selector(
    st: Any,
    tickets: list[TicketItem],
    key_prefix: str,
) -> str:
    with st.container(border=True):
        source = st.radio(
            "Origem do Ticket",
            options=["Ticket pré-cadastrado", "Relato personalizado"],
            horizontal=True,
            key=f"{key_prefix}_ticket_source",
        )
        if source == "Ticket pré-cadastrado":
            selected_ticket = st.selectbox(
                "Ticket em data/tickets.json",
                options=tickets,
                format_func=lambda t: f"{t.ticket_id} — {t.title}",
                key=f"{key_prefix}_ticket_select",
            )
            st.markdown(
                f'<div class="ticket-preview">{html.escape(selected_ticket.description)}</div>',
                unsafe_allow_html=True,
            )
            return (
                f"[{selected_ticket.ticket_id}] {selected_ticket.title}: "
                f"{selected_ticket.description}"
            )

        return st.text_area(
            "Descreva o problema técnico do cliente",
            height=130,
            placeholder="Ex.: Erro 504 ao processar webhook de pagamento durante pico de vendas...",
            key=f"{key_prefix}_ticket_custom",
        ).strip()


def _render_status_banner(st: Any, result: PipelineExecutionResult) -> None:
    record = result.record
    path_str = html.escape(str(result.storage_path))
    model_str = html.escape(f"{record.provider}/{record.model}")

    if result.from_cache:
        st.markdown(
            (
                '<div class="status-banner status-banner-cache">'
                '<span class="badge-pill badge-cache-hit">CACHE HIT</span>'
                f'<span>Resposta reutilizada de <code>{path_str}</code> sem nova chamada à API.</span>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            (
                '<div class="status-banner status-banner-api">'
                '<span class="badge-pill badge-api-new">NOVA EXECUÇÃO API</span>'
                f'<span>Gerado via <code>{model_str}</code> e salvo em <code>{path_str}</code>.</span>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    meta_line = (
        f"versão={html.escape(record.prompt_version)} | "
        f"tipo={html.escape(record.content_type)} | "
        f"temp={record.temperature} | "
        f"utc={html.escape(record.timestamp)}<br/>"
        f"cache_key={html.escape(record.cache_key)}"
    )
    st.markdown(f'<div class="meta-bar">{meta_line}</div>', unsafe_allow_html=True)


def _render_output_and_rubric(
    st: Any,
    result: PipelineExecutionResult,
    prompt_text: str,
    storage: JsonHistoryStorage,
    form_key: str,
    on_evaluated: Any | None = None,
) -> None:
    record = result.record
    _render_status_banner(st, result)

    with st.container(border=True):
        output_text = format_output_text(record.output)
        if record.content_type == "flow_summary":
            st.code(output_text, language="text")
        elif isinstance(record.output, dict):
            st.json(record.output)
        else:
            st.markdown(output_text)

    with st.expander("Inspecionar prompt exato enviado ao modelo"):
        st.code(prompt_text, language="text")

    with st.form(key=form_key):
        st.markdown("#### Rubrica Obrigatória (1 a 5)")
        existing = record.evaluation

        c_col_s, c_col_j = st.columns([1, 2.2])
        with c_col_s:
            c_score = st.slider(
                "Clareza (clarity)",
                min_value=1,
                max_value=5,
                value=existing.clarity.score if existing else 4,
                key=f"{form_key}_clarity_score",
            )
        with c_col_j:
            c_just = st.text_input(
                "Justificativa — Clareza",
                value=existing.clarity.justification if existing else "",
                placeholder="A resposta é clara, organizada e sem ambiguidade?",
                key=f"{form_key}_clarity_just",
            )

        f_col_s, f_col_j = st.columns([1, 2.2])
        with f_col_s:
            f_score = st.slider(
                "Adequação (customer_fit)",
                min_value=1,
                max_value=5,
                value=existing.customer_fit.score if existing else 4,
                key=f"{form_key}_fit_score",
            )
        with f_col_j:
            f_just = st.text_input(
                "Justificativa — Adequação ao Cliente",
                value=existing.customer_fit.justification if existing else "",
                placeholder="Vocabulário, profundidade e formato combinam com perfil e canal?",
                key=f"{form_key}_fit_just",
            )

        a_col_s, a_col_j = st.columns([1, 2.2])
        with a_col_s:
            a_score = st.slider(
                "Acionabilidade (actionability)",
                min_value=1,
                max_value=5,
                value=existing.actionability.score if existing else 4,
                key=f"{form_key}_act_score",
            )
        with a_col_j:
            a_just = st.text_input(
                "Justificativa — Acionabilidade",
                value=existing.actionability.justification if existing else "",
                placeholder="O cliente ou atendente sabe exatamente o que fazer a seguir?",
                key=f"{form_key}_act_just",
            )

        submitted = st.form_submit_button(
            "Salvar Avaliação da Rubrica",
            use_container_width=True,
        )
        if submitted:
            try:
                updated = save_rubric_rating(
                    cache_key=record.cache_key,
                    clarity_score=c_score,
                    clarity_justification=c_just,
                    customer_fit_score=f_score,
                    customer_fit_justification=f_just,
                    actionability_score=a_score,
                    actionability_justification=a_just,
                    storage=storage,
                )
                if callable(on_evaluated):
                    on_evaluated(updated)
                total = (
                    updated.evaluation.clarity.score
                    + updated.evaluation.customer_fit.score
                    + updated.evaluation.actionability.score
                )
                st.success(
                    f"Rubrica salva em `{storage.file_path}` — Nota: **{total}/15** "
                    f"(Clareza: {updated.evaluation.clarity.score}/5, "
                    f"Adequação: {updated.evaluation.customer_fit.score}/5, "
                    f"Acionabilidade: {updated.evaluation.actionability.score}/5)"
                )
            except (ValidationError, ValueError, KeyError) as exc:
                st.error(f"Não foi possível salvar a rubrica: {exc}")


def _render_ticket_dashboard(
    st: Any,
    all_records: list[ExecutionRecord],
    profiles: list[CustomerProfile],
    storage: JsonHistoryStorage,
) -> None:
    """Renderiza o Dashboard Interativo de Análise de Tickets e Atendimento na Aba 3."""
    import altair as alt
    import pandas as pd

    if not all_records:
        with st.container(border=True):
            st.info(
                f"Nenhuma execução registrada ainda em `{storage.file_path}`. "
                "Gere atendimentos na Aba 1 ou comparações na Aba 2 para alimentar o dashboard."
            )
        return

    all_rows = build_ticket_analytics_rows(all_records, profiles)
    paired_all = list(zip(all_records, all_rows))

    # A. BARRA SUPERIOR DE FILTROS GLOBAIS + EXPORTAÇÃO
    with st.container(border=True):
        st.markdown("#### Filtros Globais e Exportação")
        ticket_options = ["Todos"] + sorted({r["ticket_id"] for r in all_rows})
        level_options = ["Todos"] + list(_TECHNICAL_LEVELS)
        extra_levels = sorted(
            {
                r["technical_level"]
                for r in all_rows
                if r["technical_level"] not in _TECHNICAL_LEVELS
            }
        )
        level_options.extend(extra_levels)

        channel_options = ["Todos"] + list(_CHANNELS)
        extra_channels = sorted(
            {r["channel"] for r in all_rows if r["channel"] not in _CHANNELS}
        )
        channel_options.extend(extra_channels)

        type_options = ["Todos"] + list(CONTENT_TYPES)
        version_options = ["Todas"] + list(AVAILABLE_VERSIONS)

        fc1, fc2, fc3, fc4, fc5 = st.columns(5)
        sel_ticket = fc1.selectbox("Ticket", ticket_options, key="dash_filter_ticket")
        sel_level = fc2.selectbox("Nível Técnico", level_options, key="dash_filter_level")
        sel_channel = fc3.selectbox("Canal", channel_options, key="dash_filter_channel")
        sel_type = fc4.selectbox("Tipo de Artefato", type_options, key="dash_filter_type")
        sel_version = fc5.selectbox("Versão", version_options, key="dash_filter_version")

        filtered_pairs = [
            (rec, row)
            for rec, row in paired_all
            if (sel_ticket == "Todos" or row["ticket_id"] == sel_ticket)
            and (sel_level == "Todos" or row["technical_level"] == sel_level)
            and (sel_channel == "Todos" or row["channel"] == sel_channel)
            and (sel_type == "Todos" or row["content_type"] == sel_type)
            and (sel_version == "Todas" or row["prompt_version"] == sel_version)
        ]

        ac1, ac2, ac3 = st.columns([1.4, 1.0, 1.0])
        with ac1:
            only_evaluated = st.checkbox(
                "Somente execuções avaliadas",
                value=False,
                key="dash_filter_only_eval",
            )
        if only_evaluated:
            filtered_pairs = [(rec, row) for rec, row in filtered_pairs if row["is_evaluated"]]

        filtered_records = [rec for rec, _ in filtered_pairs]
        filtered_rows = [row for _, row in filtered_pairs]

        with ac2:
            st.download_button(
                label=f"Baixar JSON Filtrado ({len(filtered_records)})",
                data=export_history_json(filtered_records),
                file_name="history_filtered.json",
                mime="application/json",
                use_container_width=True,
                key="dash_dl_json",
            )
        with ac3:
            st.download_button(
                label=f"Baixar CSV Analítico ({len(filtered_rows)})",
                data=_export_analytics_csv(filtered_rows),
                file_name="tickets_analytics.csv",
                mime="text/csv",
                use_container_width=True,
                key="dash_dl_csv",
            )

    if not filtered_rows:
        with st.container(border=True):
            st.info("Nenhuma execução corresponde aos filtros selecionados.")
        return

    # B. LINHA DE KPIs DINÂMICOS (REAGEM AOS FILTROS)
    evaluated_rows = [r for r in filtered_rows if r["is_evaluated"] and r["total_score"] is not None]
    analytics = aggregate_rubric_metrics(filtered_records)
    deltas = analytics["deltas_v2_vs_v1"]
    ticket_summaries = summarize_by_ticket(filtered_rows)

    if evaluated_rows:
        overall_avg = round(fmean(int(r["total_score"]) for r in evaluated_rows), 2)
        dim_avgs = {
            "Clareza": round(fmean(int(r["clarity_score"]) for r in evaluated_rows), 2),
            "Adequação": round(fmean(int(r["customer_fit_score"]) for r in evaluated_rows), 2),
            "Acionabilidade": round(
                fmean(int(r["actionability_score"]) for r in evaluated_rows), 2
            ),
        }
        best_dim_name, best_dim_score = max(dim_avgs.items(), key=lambda item: item[1])
        evaluated_tickets = [
            t for t in ticket_summaries if t["avg_total_score"] is not None
        ]
        critical_ticket = (
            min(evaluated_tickets, key=lambda t: float(t["avg_total_score"]))
            if evaluated_tickets
            else None
        )
    else:
        overall_avg = None
        best_dim_name, best_dim_score = None, None
        critical_ticket = None

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        with st.container(border=True):
            st.metric("Execuções no Recorte", len(filtered_rows))
            st.caption(f"{len(evaluated_rows)} avaliada(s) com rubrica")
    with k2:
        with st.container(border=True):
            st.metric(
                "Nota Média Geral (/15)",
                f"{overall_avg:.2f}" if overall_avg is not None else "—",
                delta=(
                    f"{deltas['total_avg']:+.2f} (v2 vs v1)"
                    if deltas is not None
                    else None
                ),
            )
            if deltas is None:
                st.caption("Avalie v1 e v2 no recorte para ver o delta")
    with k3:
        with st.container(border=True):
            st.metric("Melhor Dimensão", best_dim_name if best_dim_name else "—")
            st.caption(
                f"Média: {best_dim_score:.2f} / 5.00"
                if best_dim_score is not None
                else "Sem avaliações no recorte"
            )
    with k4:
        with st.container(border=True):
            st.metric(
                "Ticket Mais Crítico",
                critical_ticket["ticket_id"] if critical_ticket else "—",
            )
            st.caption(
                f"Média: {critical_ticket['avg_total_score']:.2f} / 15"
                if critical_ticket is not None
                else "Sem tickets avaliados"
            )

    # C. VISUALIZAÇÕES INTERATIVAS EM GRID (2 x 2)
    palette_map = {"v1": "#818CF8", "v2": "#38BDF8"}
    fallback_colors = ("#10B981", "#F59E0B", "#A78BFA")

    row1_left, row1_right = st.columns(2, gap="medium")

    # C1. Desempenho por Ticket: v1 vs v2 (Barras horizontais agrupadas)
    with row1_left:
        with st.container(border=True):
            st.markdown("##### Desempenho por Ticket (`v1` vs `v2`)")
            if not evaluated_rows:
                st.info("Avalie execuções com a rubrica para visualizar o desempenho por ticket.")
            else:
                ticket_ver_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
                for row in evaluated_rows:
                    ticket_ver_groups[(row["ticket_id"], row["prompt_version"])].append(row)

                t_chart_rows: list[dict[str, Any]] = []
                for (tid, ver), grp in sorted(ticket_ver_groups.items()):
                    t_chart_rows.append(
                        {
                            "ticket_id": tid,
                            "version": ver,
                            "avg_total": round(fmean(int(r["total_score"]) for r in grp), 2),
                            "count": len(grp),
                            "clarity_avg": round(fmean(int(r["clarity_score"]) for r in grp), 2),
                            "fit_avg": round(
                                fmean(int(r["customer_fit_score"]) for r in grp), 2
                            ),
                            "act_avg": round(
                                fmean(int(r["actionability_score"]) for r in grp), 2
                            ),
                        }
                    )

                active_t_versions = sorted({r["version"] for r in t_chart_rows})
                t_colors = [
                    palette_map.get(v, fallback_colors[i % len(fallback_colors)])
                    for i, v in enumerate(active_t_versions)
                ]

                ticket_chart = (
                    alt.Chart(alt.Data(values=t_chart_rows))
                    .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
                    .encode(
                        y=alt.Y(
                            "ticket_id:N",
                            title=None,
                            axis=alt.Axis(
                                labelColor="#E6EDF3",
                                labelFontSize=11,
                                domainColor="#334155",
                                tickColor="#334155",
                            ),
                        ),
                        yOffset=alt.YOffset("version:N", sort=active_t_versions),
                        x=alt.X(
                            "avg_total:Q",
                            scale=alt.Scale(domain=[0, 15]),
                            title="Nota Média (/15)",
                            axis=alt.Axis(
                                values=[0, 3, 6, 9, 12, 15],
                                labelColor="#E6EDF3",
                                titleColor="#94A3B8",
                                gridColor="rgba(148, 163, 184, 0.16)",
                                domainColor="#334155",
                            ),
                        ),
                        color=alt.Color(
                            "version:N",
                            scale=alt.Scale(domain=active_t_versions, range=t_colors),
                            legend=alt.Legend(
                                title="Versão",
                                orient="top",
                                labelColor="#E6EDF3",
                                titleColor="#94A3B8",
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip("ticket_id:N", title="Ticket"),
                            alt.Tooltip("version:N", title="Versão"),
                            alt.Tooltip("avg_total:Q", title="Média Total (/15)", format=".2f"),
                            alt.Tooltip("count:Q", title="Avaliações"),
                            alt.Tooltip("clarity_avg:Q", title="Clareza (/5)", format=".2f"),
                            alt.Tooltip("fit_avg:Q", title="Adequação (/5)", format=".2f"),
                            alt.Tooltip("act_avg:Q", title="Acionabilidade (/5)", format=".2f"),
                        ],
                    )
                    .properties(height=270)
                    .configure(background="transparent")
                    .configure_view(stroke=None)
                )
                st.altair_chart(ticket_chart, use_container_width=True)

    # C2. Heatmap: Nível Técnico × Canal
    with row1_right:
        with st.container(border=True):
            hm_hdr_l, hm_hdr_r = st.columns([1.4, 1])
            with hm_hdr_l:
                st.markdown("##### Heatmap: Nível Técnico × Canal")
            with hm_hdr_r:
                hm_metric = st.radio(
                    "Métrica",
                    options=["Total (/15)", "Adequação (/5)"],
                    horizontal=True,
                    label_visibility="collapsed",
                    key="dash_heatmap_metric",
                )

            if not evaluated_rows:
                st.info("Avalie execuções com a rubrica para habilitar a matriz Nível × Canal.")
            else:
                hm_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
                for row in evaluated_rows:
                    hm_groups[(row["technical_level"], row["channel"])].append(row)

                use_total = hm_metric == "Total (/15)"
                hm_domain = [0, 15] if use_total else [0, 5]
                hm_rows: list[dict[str, Any]] = []
                for (lvl, ch), grp in sorted(hm_groups.items()):
                    val = round(
                        fmean(
                            int(r["total_score"] if use_total else r["customer_fit_score"])
                            for r in grp
                        ),
                        2,
                    )
                    hm_rows.append(
                        {
                            "technical_level": lvl,
                            "channel": ch,
                            "metric_val": val,
                            "label": f"{val:.1f}",
                            "count": len(grp),
                        }
                    )

                hm_base = alt.Chart(alt.Data(values=hm_rows)).encode(
                    x=alt.X(
                        "channel:N",
                        title="Canal de Atendimento",
                        axis=alt.Axis(
                            labelAngle=0,
                            labelColor="#E6EDF3",
                            titleColor="#94A3B8",
                            domainColor="#334155",
                        ),
                    ),
                    y=alt.Y(
                        "technical_level:N",
                        title="Nível Técnico",
                        axis=alt.Axis(
                            labelColor="#E6EDF3",
                            titleColor="#94A3B8",
                            domainColor="#334155",
                        ),
                    ),
                )
                rects = hm_base.mark_rect(cornerRadius=6, stroke="#0E1117", strokeWidth=2).encode(
                    color=alt.Color(
                        "metric_val:Q",
                        scale=alt.Scale(
                            domain=hm_domain,
                            range=["#1E293B", "#0284C7", "#10B981"],
                        ),
                        legend=alt.Legend(
                            title=hm_metric,
                            labelColor="#E6EDF3",
                            titleColor="#94A3B8",
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("technical_level:N", title="Nível Técnico"),
                        alt.Tooltip("channel:N", title="Canal"),
                        alt.Tooltip("metric_val:Q", title=hm_metric, format=".2f"),
                        alt.Tooltip("count:Q", title="Avaliações"),
                    ],
                )
                labels = hm_base.mark_text(
                    fontSize=13,
                    fontWeight="bold",
                    color="#E6EDF3",
                ).encode(text="label:N")

                heatmap_chart = (
                    (rects + labels)
                    .properties(height=270)
                    .configure(background="transparent")
                    .configure_view(stroke=None)
                )
                st.altair_chart(heatmap_chart, use_container_width=True)

    row2_left, row2_right = st.columns(2, gap="medium")

    # C3. Comparativo por Critério da Rubrica (v1 vs v2)
    with row2_left:
        with st.container(border=True):
            st.markdown("##### Comparativo por Critério da Rubrica (`v1` vs `v2`)")
            if analytics["evaluated_count"] == 0:
                st.info("Avalie execuções para comparar Clareza, Adequação e Acionabilidade.")
            else:
                by_version = analytics["by_version"]
                criterion_specs = (
                    ("Clareza", "clarity_avg"),
                    ("Adequação", "customer_fit_avg"),
                    ("Acionabilidade", "actionability_avg"),
                )
                active_versions = [v for v, s in by_version.items() if s["count"] > 0]
                crit_rows = [
                    {
                        "version": ver,
                        "criterion": label,
                        "score": by_version[ver][key],
                        "count": by_version[ver]["count"],
                    }
                    for ver in active_versions
                    for label, key in criterion_specs
                ]
                crit_colors = [
                    palette_map.get(v, fallback_colors[i % len(fallback_colors)])
                    for i, v in enumerate(active_versions)
                ]

                crit_chart = (
                    alt.Chart(alt.Data(values=crit_rows))
                    .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, size=30)
                    .encode(
                        x=alt.X(
                            "criterion:N",
                            sort=["Clareza", "Adequação", "Acionabilidade"],
                            title=None,
                            axis=alt.Axis(
                                labelAngle=0,
                                labelColor="#E6EDF3",
                                labelFontSize=11,
                                domainColor="#334155",
                            ),
                        ),
                        xOffset=alt.XOffset("version:N", sort=active_versions),
                        y=alt.Y(
                            "score:Q",
                            scale=alt.Scale(domain=[0, 5]),
                            title="Nota Média (0 a 5)",
                            axis=alt.Axis(
                                values=[0, 1, 2, 3, 4, 5],
                                labelColor="#E6EDF3",
                                titleColor="#94A3B8",
                                gridColor="rgba(148, 163, 184, 0.16)",
                                domainColor="#334155",
                            ),
                        ),
                        color=alt.Color(
                            "version:N",
                            scale=alt.Scale(domain=active_versions, range=crit_colors),
                            legend=alt.Legend(
                                title="Versão",
                                orient="top",
                                labelColor="#E6EDF3",
                                titleColor="#94A3B8",
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip("version:N", title="Versão"),
                            alt.Tooltip("criterion:N", title="Critério"),
                            alt.Tooltip("score:Q", title="Média (/5)", format=".2f"),
                            alt.Tooltip("count:Q", title="Amostras"),
                        ],
                    )
                    .properties(height=270)
                    .configure(background="transparent")
                    .configure_view(stroke=None)
                )
                st.altair_chart(crit_chart, use_container_width=True)

    # C4. Distribuição por Tipo de Artefato (content_type)
    with row2_right:
        with st.container(border=True):
            st.markdown("##### Distribuição por Tipo de Artefato (`content_type`)")
            ct_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in filtered_rows:
                ct_groups[row["content_type"]].append(row)

            ct_rows: list[dict[str, Any]] = []
            for ct in CONTENT_TYPES:
                grp = ct_groups.get(ct, [])
                if not grp:
                    continue
                ev_grp = [r for r in grp if r["is_evaluated"] and r["total_score"] is not None]
                avg_t = (
                    round(fmean(int(r["total_score"]) for r in ev_grp), 2)
                    if ev_grp
                    else None
                )
                ct_rows.append(
                    {
                        "content_type": ct,
                        "executions": len(grp),
                        "evaluated": len(ev_grp),
                        "avg_total": avg_t if avg_t is not None else 0.0,
                        "avg_label": f"{avg_t:.1f}/15" if avg_t is not None else "S/ nota",
                    }
                )

            ct_base = alt.Chart(alt.Data(values=ct_rows)).encode(
                x=alt.X(
                    "content_type:N",
                    sort=list(CONTENT_TYPES),
                    title=None,
                    axis=alt.Axis(
                        labelAngle=0,
                        labelColor="#E6EDF3",
                        labelFontSize=11,
                        domainColor="#334155",
                    ),
                ),
                y=alt.Y(
                    "executions:Q",
                    title="Volume de Execuções",
                    axis=alt.Axis(
                        tickMinStep=1,
                        labelColor="#E6EDF3",
                        titleColor="#94A3B8",
                        gridColor="rgba(148, 163, 184, 0.16)",
                        domainColor="#334155",
                    ),
                ),
            )
            ct_bars = ct_base.mark_bar(
                cornerRadiusTopLeft=5,
                cornerRadiusTopRight=5,
                size=42,
            ).encode(
                color=alt.Color(
                    "avg_total:Q",
                    scale=alt.Scale(
                        domain=[0, 15],
                        range=["#475569", "#38BDF8", "#10B981"],
                    ),
                    legend=alt.Legend(
                        title="Nota Média (/15)",
                        orient="top",
                        labelColor="#E6EDF3",
                        titleColor="#94A3B8",
                    ),
                ),
                tooltip=[
                    alt.Tooltip("content_type:N", title="Artefato"),
                    alt.Tooltip("executions:Q", title="Execuções"),
                    alt.Tooltip("evaluated:Q", title="Avaliadas"),
                    alt.Tooltip("avg_label:N", title="Nota Média"),
                ],
            )
            ct_labels = ct_base.mark_text(
                dy=-8,
                fontSize=11,
                fontWeight="bold",
                color="#E6EDF3",
            ).encode(text="avg_label:N")

            ct_chart = (
                (ct_bars + ct_labels)
                .properties(height=270)
                .configure(background="transparent")
                .configure_view(stroke=None)
            )
            st.altair_chart(ct_chart, use_container_width=True)

    # D. EXPLORADOR INTERATIVO DE TICKETS COM DRILL-DOWN (on_select="rerun")
    with st.container(border=True):
        st.markdown("#### Explorador Interativo de Tickets")
        st.caption(
            "Clique em qualquer linha da tabela abaixo para abrir o painel inspetor completo "
            "da execução selecionada."
        )

        ordered_pairs = list(reversed(filtered_pairs))
        display_df = pd.DataFrame(
            [
                {
                    "Ticket": row["ticket_id"],
                    "Cliente": row["customer_name"],
                    "Nível": row["technical_level"],
                    "Canal": row["channel"],
                    "Artefato": row["content_type"],
                    "Versão": row["prompt_version"],
                    "Clareza": row["clarity_score"] if row["clarity_score"] is not None else "—",
                    "Adequação": (
                        row["customer_fit_score"]
                        if row["customer_fit_score"] is not None
                        else "—"
                    ),
                    "Acionabilidade": (
                        row["actionability_score"]
                        if row["actionability_score"] is not None
                        else "—"
                    ),
                    "Total (/15)": (
                        row["total_score"] if row["total_score"] is not None else "—"
                    ),
                    "Data": row["timestamp"][:19].replace("T", " "),
                }
                for _, row in ordered_pairs
            ]
        )

        selection_event = st.dataframe(
            display_df,
            on_select="rerun",
            selection_mode="single-row",
            use_container_width=True,
            hide_index=True,
            key="ticket_explorer_table",
        )

        selected_indices = (
            selection_event.selection.rows
            if selection_event is not None and hasattr(selection_event, "selection")
            else []
        )
        selected_idx = selected_indices[0] if selected_indices else 0
        if selected_idx >= len(ordered_pairs):
            selected_idx = 0

        sel_rec, sel_row = ordered_pairs[selected_idx]

    # PAINEL INSPETOR DO TICKET SELECIONADO
    with st.container(border=True):
        badge_ver_cls = "badge-v2" if sel_row["prompt_version"] == "v2" else "badge-v1"
        st.markdown(
            (
                '<div class="version-header">'
                f'<span class="version-title">Inspetor de Execução · '
                f'{html.escape(sel_row["ticket_id"])} ({html.escape(sel_row["content_type"])})</span>'
                f'<span class="badge-pill {badge_ver_cls}">'
                f'{html.escape(sel_row["prompt_version"])}</span>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        col_left, col_right = st.columns([1, 1.3], gap="large")
        with col_left:
            st.markdown(
                f"**Cliente:** {sel_row['customer_name']} (`{sel_row['customer_id']}`)"
            )
            _render_badge_pills_raw(
                st,
                sel_row["technical_level"],
                sel_row["communication_style"],
                sel_row["channel"],
            )

            meta_html = (
                f"modelo={html.escape(sel_rec.provider)}/{html.escape(sel_rec.model)} | "
                f"temp={sel_rec.temperature} | "
                f"utc={html.escape(sel_rec.timestamp)}<br/>"
                f"cache_key={html.escape(sel_rec.cache_key)}"
            )
            st.markdown(f'<div class="meta-bar">{meta_html}</div>', unsafe_allow_html=True)

            st.markdown("**Enunciado Completo do Ticket**")
            st.markdown(
                f'<div class="ticket-preview">{html.escape(sel_rec.ticket)}</div>',
                unsafe_allow_html=True,
            )

            st.write("")
            st.markdown("**Avaliação da Rubrica**")
            if sel_rec.evaluation is not None:
                ev = sel_rec.evaluation
                rc1, rc2, rc3 = st.columns(3)
                with rc1:
                    with st.container(border=True):
                        st.metric("Clareza", f"{ev.clarity.score}/5")
                        st.caption(ev.clarity.justification)
                with rc2:
                    with st.container(border=True):
                        st.metric("Adequação", f"{ev.customer_fit.score}/5")
                        st.caption(ev.customer_fit.justification)
                with rc3:
                    with st.container(border=True):
                        st.metric("Acionabilidade", f"{ev.actionability.score}/5")
                        st.caption(ev.actionability.justification)
            else:
                st.info("Esta execução ainda não foi avaliada com a rubrica.")

        with col_right:
            st.markdown("**Resposta Gerada pela IA**")
            with st.container(border=True):
                out_text = format_output_text(sel_rec.output)
                if sel_rec.content_type == "flow_summary":
                    st.code(out_text, language="text")
                elif isinstance(sel_rec.output, dict):
                    st.json(sel_rec.output)
                else:
                    st.markdown(out_text)


def _render_sidebar(
    st: Any,
    storage: JsonHistoryStorage,
    profiles: list[CustomerProfile],
    tickets: list[TicketItem],
) -> None:
    with st.sidebar:
        st.markdown("### Console do Pipeline")
        with st.container(border=True):
            st.markdown("**Ambiente e Dados**")
            st.markdown(
                (
                    f'<div class="sidebar-kv">Histórico JSON: <code>{html.escape(str(storage.file_path))}</code></div>'
                    f'<div class="sidebar-kv">Perfis carregados: <code>{len(profiles)}</code></div>'
                    f'<div class="sidebar-kv">Tickets carregados: <code>{len(tickets)}</code></div>'
                    f'<div class="sidebar-kv">Versões de prompt: <code>{html.escape(", ".join(AVAILABLE_VERSIONS))}</code></div>'
                ),
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            st.markdown("**Guia dos 4 Artefatos**")
            items_html = "".join(
                f'<div class="artifact-item"><strong>{ct}</strong><br/>{html.escape(desc)}</div>'
                for ct, desc in _ARTIFACT_GUIDE
            )
            st.markdown(items_html, unsafe_allow_html=True)


def render_app() -> None:
    """Renderiza a aplicação Streamlit com 3 abas em tema escuro técnico."""
    import streamlit as st

    st.set_page_config(
        page_title="Assistente de Suporte Técnico IA",
        page_icon="🤖",
        layout="wide",
    )
    _inject_custom_css(st)

    st.markdown(
        (
            '<div class="console-header">'
            '<div class="console-kicker">Console Técnico · Engenharia de Prompt &amp; Pipeline IA</div>'
            '<h1 class="console-title">Assistente de Suporte Técnico com IA</h1>'
            '<p class="console-subtitle">'
            "Adaptação dinâmica por perfil e canal, comparação controlada v1 vs v2, "
            "cache determinístico SHA-256 e auditoria por rubrica."
            "</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    storage = JsonHistoryStorage(DEFAULT_HISTORY_PATH)
    try:
        profiles = load_profiles_data()
        tickets = load_tickets_data()
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        st.error(f"Erro ao carregar dados iniciais: {exc}")
        return

    _render_sidebar(st, storage, profiles, tickets)

    tab_generate, tab_compare, tab_history = st.tabs(
        [
            "Gerar Atendimento",
            "Comparação Controlada (v1 vs v2)",
            "Histórico e Auditoria",
        ]
    )

    # ABA 1 — GERAR ATENDIMENTO (SPLIT WORKSPACE 2 COLUNAS)
    with tab_generate:
        col_config, col_output = st.columns([1, 1.35], gap="large")

        with col_config:
            with st.container(border=True):
                st.markdown("#### Configuração do Atendimento")
                selected_profile = st.selectbox(
                    "Perfil do Cliente",
                    options=profiles,
                    format_func=lambda p: (
                        f"{p.customer_id} — {p.name} ({p.technical_level} / {p.channel})"
                    ),
                    key="gen_profile",
                )
                _render_profile_badges(st, selected_profile)

                ticket_text = _render_ticket_selector(st, tickets, key_prefix="gen")

                col_type, col_ver = st.columns([1.8, 1])
                with col_type:
                    selected_type: ContentType = st.selectbox(
                        "Tipo de Conteúdo",
                        options=list(CONTENT_TYPES),
                        format_func=lambda ct: f"{ct} — {CONTENT_TYPE_LABELS[ct]}",
                        key="gen_content_type",
                    )
                with col_ver:
                    selected_version = st.selectbox(
                        "Versão do Prompt",
                        options=list(reversed(AVAILABLE_VERSIONS)),
                        key="gen_prompt_version",
                    )

                force_refresh = st.checkbox(
                    "Forçar nova chamada à API (ignorar cache)",
                    value=False,
                    key="gen_force_refresh",
                )

                if st.button(
                    "Gerar Resposta",
                    type="primary",
                    key="btn_generate",
                    use_container_width=True,
                ):
                    if not ticket_text:
                        st.error("Informe ou selecione um ticket antes de gerar a resposta.")
                    else:
                        try:
                            with st.spinner("Executando pipeline de geração..."):
                                result, prompt_text = run_generation(
                                    profile=selected_profile,
                                    ticket=ticket_text,
                                    content_type=selected_type,
                                    prompt_version=selected_version,
                                    force_refresh=force_refresh,
                                    storage=storage,
                                )
                            st.session_state["last_generation"] = (result, prompt_text)
                        except (LLMServiceError, ValueError, ValidationError) as exc:
                            st.error(str(exc))

        with col_output:
            if "last_generation" not in st.session_state:
                with st.container(border=True):
                    st.markdown(
                        (
                            '<div class="empty-state">'
                            '<div class="empty-state-title">Pronto para gerar o artefato</div>'
                            "<div>Selecione o perfil do cliente, o ticket, o tipo de conteúdo e a versão "
                            "do prompt no painel à esquerda e clique em <strong>Gerar Resposta</strong> "
                            "para visualizar a saída, inspecionar o prompt e aplicar a rubrica.</div>"
                            "</div>"
                        ),
                        unsafe_allow_html=True,
                    )
            else:
                gen_result, gen_prompt = st.session_state["last_generation"]

                def _update_gen_state(updated_record: ExecutionRecord) -> None:
                    st.session_state["last_generation"] = (
                        PipelineExecutionResult(
                            record=updated_record,
                            from_cache=gen_result.from_cache,
                            storage_path=gen_result.storage_path,
                        ),
                        gen_prompt,
                    )

                _render_output_and_rubric(
                    st=st,
                    result=gen_result,
                    prompt_text=gen_prompt,
                    storage=storage,
                    form_key=f"rubric_gen_{gen_result.record.cache_key[:12]}",
                    on_evaluated=_update_gen_state,
                )

    # ABA 2 — COMPARAÇÃO CONTROLADA (v1 vs v2)
    with tab_compare:
        with st.container(border=True):
            st.markdown("#### Comparação Lado a Lado (`v1` Baseline vs `v2` Otimizada)")
            st.caption(
                "Executa exatamente o mesmo cliente, ticket, provedor, modelo e temperatura, "
                "variando exclusivamente a versão do prompt."
            )

            top_left, top_right = st.columns([1.15, 1.35], gap="large")
            with top_left:
                cmp_profile = st.selectbox(
                    "Perfil do Cliente",
                    options=profiles,
                    format_func=lambda p: (
                        f"{p.customer_id} — {p.name} ({p.technical_level} / {p.channel})"
                    ),
                    key="cmp_profile",
                )
                _render_profile_badges(st, cmp_profile)

                cmp_type: ContentType = st.selectbox(
                    "Tipo de Conteúdo",
                    options=list(CONTENT_TYPES),
                    format_func=lambda ct: f"{ct} — {CONTENT_TYPE_LABELS[ct]}",
                    key="cmp_content_type",
                )
                cmp_force_refresh = st.checkbox(
                    "Forçar nova chamada à API (ambas versões)",
                    value=False,
                    key="cmp_force_refresh",
                )
            with top_right:
                cmp_ticket_text = _render_ticket_selector(st, tickets, key_prefix="cmp")

            if st.button(
                "Comparar v1 vs v2",
                type="primary",
                key="btn_compare",
                use_container_width=True,
            ):
                if not cmp_ticket_text:
                    st.error("Informe ou selecione um ticket antes de iniciar a comparação.")
                else:
                    try:
                        with st.spinner("Executando v1 e v2 sob as mesmas configurações..."):
                            comparison = run_comparison(
                                profile=cmp_profile,
                                ticket=cmp_ticket_text,
                                content_type=cmp_type,
                                force_refresh=cmp_force_refresh,
                                storage=storage,
                            )
                        st.session_state["last_comparison"] = comparison
                    except (LLMServiceError, ValueError, ValidationError) as exc:
                        st.error(str(exc))

        if "last_comparison" in st.session_state:
            comparison_data = st.session_state["last_comparison"]
            col_v1, col_v2 = st.columns(2, gap="large")

            with col_v1:
                with st.container(border=True):
                    st.markdown(
                        (
                            '<div class="version-header">'
                            '<span class="version-title">Versão v1</span>'
                            '<span class="badge-pill badge-v1">Baseline Minimalista</span>'
                            "</div>"
                        ),
                        unsafe_allow_html=True,
                    )
                    res_v1, prompt_v1 = comparison_data["v1"]

                    def _update_v1_state(updated_record: ExecutionRecord) -> None:
                        comparison_data["v1"] = (
                            PipelineExecutionResult(
                                record=updated_record,
                                from_cache=res_v1.from_cache,
                                storage_path=res_v1.storage_path,
                            ),
                            prompt_v1,
                        )
                        st.session_state["last_comparison"] = comparison_data

                    _render_output_and_rubric(
                        st=st,
                        result=res_v1,
                        prompt_text=prompt_v1,
                        storage=storage,
                        form_key=f"rubric_cmp_v1_{res_v1.record.cache_key[:12]}",
                        on_evaluated=_update_v1_state,
                    )

            with col_v2:
                with st.container(border=True):
                    st.markdown(
                        (
                            '<div class="version-header">'
                            '<span class="version-title">Versão v2</span>'
                            '<span class="badge-pill badge-v2">Otimizada · Calibrada</span>'
                            "</div>"
                        ),
                        unsafe_allow_html=True,
                    )
                    res_v2, prompt_v2 = comparison_data["v2"]

                    def _update_v2_state(updated_record: ExecutionRecord) -> None:
                        comparison_data["v2"] = (
                            PipelineExecutionResult(
                                record=updated_record,
                                from_cache=res_v2.from_cache,
                                storage_path=res_v2.storage_path,
                            ),
                            prompt_v2,
                        )
                        st.session_state["last_comparison"] = comparison_data

                    _render_output_and_rubric(
                        st=st,
                        result=res_v2,
                        prompt_text=prompt_v2,
                        storage=storage,
                        form_key=f"rubric_cmp_v2_{res_v2.record.cache_key[:12]}",
                        on_evaluated=_update_v2_state,
                    )

    # ABA 3 — HISTÓRICO E AUDITORIA (DASHBOARD INTERATIVO DE TICKETS)
    with tab_history:
        try:
            all_records = storage.load_all()
        except (ValueError, ValidationError) as exc:
            st.error(f"Erro ao carregar histórico JSON: {exc}")
            all_records = []

        _render_ticket_dashboard(st, all_records, profiles, storage)


if __name__ == "__main__":
    render_app()
