from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from typing import Any

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
    GeminiClient,
    JsonHistoryStorage,
    LLMServiceError,
)

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


def _inject_custom_css(st: Any) -> None:
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


def _render_profile_badges(st: Any, profile: CustomerProfile) -> None:
    level = html.escape(profile.technical_level)
    style = html.escape(profile.communication_style)
    channel = html.escape(profile.channel)
    st.markdown(
        (
            '<div class="badge-row">'
            f'<span class="badge-pill badge-level"><span class="badge-label">Nível</span> {level}</span>'
            f'<span class="badge-pill badge-style"><span class="badge-label">Estilo</span> {style}</span>'
            f'<span class="badge-pill badge-channel"><span class="badge-label">Canal</span> {channel}</span>'
            "</div>"
        ),
        unsafe_allow_html=True,
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

    # ABA 3 — HISTÓRICO E AUDITORIA
    with tab_history:
        try:
            all_records = storage.load_all()
        except (ValueError, ValidationError) as exc:
            st.error(f"Erro ao carregar histórico JSON: {exc}")
            all_records = []

        metrics = summarize_history(all_records)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            with st.container(border=True):
                st.metric("Total de Execuções", metrics["total"])
        with m2:
            with st.container(border=True):
                st.metric("Avaliadas com Rubrica", metrics["evaluated"])
        with m3:
            with st.container(border=True):
                st.metric("Execuções v1", metrics["v1"])
        with m4:
            with st.container(border=True):
                st.metric("Execuções v2", metrics["v2"])

        if not all_records:
            with st.container(border=True):
                st.info(f"Nenhuma execução registrada ainda em `{storage.file_path}`.")
        else:
            with st.container(border=True):
                f1, f2, f3, f4, f5 = st.columns([1.1, 1.1, 0.9, 1.0, 1.15])
                customer_options = ["Todos"] + sorted({r.customer_id for r in all_records})
                type_options = ["Todos"] + list(CONTENT_TYPES)
                version_options = ["Todas"] + list(AVAILABLE_VERSIONS)

                selected_cust_filter = f1.selectbox("Cliente", customer_options)
                selected_type_filter = f2.selectbox("Tipo", type_options)
                selected_ver_filter = f3.selectbox("Versão", version_options)
                with f4:
                    st.write("")
                    st.write("")
                    only_eval = st.checkbox("Somente avaliadas", value=False)

                filtered_records = filter_history_records(
                    all_records,
                    customer_id=(
                        None if selected_cust_filter == "Todos" else selected_cust_filter
                    ),
                    content_type=(
                        None if selected_type_filter == "Todos" else selected_type_filter
                    ),
                    prompt_version=(
                        None if selected_ver_filter == "Todas" else selected_ver_filter
                    ),
                    only_evaluated=only_eval,
                )

                with f5:
                    st.write("")
                    st.download_button(
                        label=f"Baixar JSON ({len(filtered_records)})",
                        data=export_history_json(filtered_records),
                        file_name="history_export.json",
                        mime="application/json",
                        use_container_width=True,
                    )

            for idx, rec in enumerate(reversed(filtered_records), start=1):
                if rec.evaluation is not None:
                    total_score = (
                        rec.evaluation.clarity.score
                        + rec.evaluation.customer_fit.score
                        + rec.evaluation.actionability.score
                    )
                    eval_tag = f"Nota: {total_score}/15"
                else:
                    eval_tag = "Sem rubrica"

                header = (
                    f"#{idx} | {rec.timestamp[:19]} UTC | {rec.customer_id} | "
                    f"{rec.content_type} ({rec.prompt_version}) | {eval_tag}"
                )
                with st.expander(header):
                    st.caption(
                        f"Modelo: `{rec.provider}/{rec.model}` (temp={rec.temperature}) | "
                        f"Cache Key: `{rec.cache_key}`"
                    )
                    st.markdown(f"**Ticket:** {rec.ticket}")
                    out_text = format_output_text(rec.output)
                    if rec.content_type == "flow_summary":
                        st.code(out_text, language="text")
                    else:
                        st.markdown(out_text)

                    if rec.evaluation is not None:
                        ev = rec.evaluation
                        st.markdown("**Rubrica Registrada:**")
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


if __name__ == "__main__":
    render_app()
