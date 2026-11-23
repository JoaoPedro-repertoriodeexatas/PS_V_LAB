from __future__ import annotations

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


def _render_profile_badges(st: Any, profile: CustomerProfile) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Nível Técnico", profile.technical_level)
    col2.metric("Estilo de Comunicação", profile.communication_style)
    col3.metric("Canal", profile.channel)


def _render_ticket_selector(
    st: Any,
    tickets: list[TicketItem],
    key_prefix: str,
) -> str:
    source = st.radio(
        "Origem do Ticket",
        options=["Ticket pré-cadastrado", "Relato personalizado"],
        horizontal=True,
        key=f"{key_prefix}_ticket_source",
    )
    if source == "Ticket pré-cadastrado":
        selected_ticket = st.selectbox(
            "Selecione um ticket de data/tickets.json",
            options=tickets,
            format_func=lambda t: f"{t.ticket_id} — {t.title}",
            key=f"{key_prefix}_ticket_select",
        )
        st.caption(selected_ticket.description)
        return f"[{selected_ticket.ticket_id}] {selected_ticket.title}: {selected_ticket.description}"

    return st.text_area(
        "Descreva o problema técnico do cliente",
        height=120,
        placeholder="Ex.: Erro 504 ao processar webhook de pagamento durante pico de vendas...",
        key=f"{key_prefix}_ticket_custom",
    ).strip()


def _render_output_and_rubric(
    st: Any,
    result: PipelineExecutionResult,
    prompt_text: str,
    storage: JsonHistoryStorage,
    form_key: str,
) -> None:
    record = result.record
    if result.from_cache:
        st.info(
            f"CACHE HIT — Resposta reutilizada de `{result.storage_path}` sem nova chamada à API."
        )
    else:
        st.success(
            f"NOVA EXECUÇÃO API — Resposta gerada via `{record.provider}/{record.model}` "
            f"e salva em `{result.storage_path}`."
        )

    st.caption(
        f"Versão: `{record.prompt_version}` | Tipo: `{record.content_type}` | "
        f"Temperatura: `{record.temperature}` | Timestamp: `{record.timestamp}`"
    )
    st.caption(f"Cache Key: `{record.cache_key}`")

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
        st.markdown("#### Avaliar com a Rubrica Obrigatória (1 a 5)")
        existing = record.evaluation

        c_score = st.slider(
            "Clareza (clarity)",
            min_value=1,
            max_value=5,
            value=existing.clarity.score if existing else 4,
            key=f"{form_key}_clarity_score",
        )
        c_just = st.text_input(
            "Justificativa — Clareza",
            value=existing.clarity.justification if existing else "",
            placeholder="A resposta é clara, organizada e sem ambiguidade?",
            key=f"{form_key}_clarity_just",
        )

        f_score = st.slider(
            "Adequação ao Nível do Cliente (customer_fit)",
            min_value=1,
            max_value=5,
            value=existing.customer_fit.score if existing else 4,
            key=f"{form_key}_fit_score",
        )
        f_just = st.text_input(
            "Justificativa — Adequação ao Cliente",
            value=existing.customer_fit.justification if existing else "",
            placeholder="Vocabulário, profundidade e formato combinam com o perfil e canal?",
            key=f"{form_key}_fit_just",
        )

        a_score = st.slider(
            "Acionabilidade (actionability)",
            min_value=1,
            max_value=5,
            value=existing.actionability.score if existing else 4,
            key=f"{form_key}_act_score",
        )
        a_just = st.text_input(
            "Justificativa — Acionabilidade",
            value=existing.actionability.justification if existing else "",
            placeholder="O cliente ou atendente sabe exatamente o que fazer a seguir?",
            key=f"{form_key}_act_just",
        )

        submitted = st.form_submit_button("Salvar Avaliação da Rubrica")
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
                st.success(
                    f"Rubrica salva em `{storage.file_path}`! "
                    f"(Clareza: {updated.evaluation.clarity.score}/5, "
                    f"Adequação: {updated.evaluation.customer_fit.score}/5, "
                    f"Acionabilidade: {updated.evaluation.actionability.score}/5)"
                )
            except (ValidationError, ValueError, KeyError) as exc:
                st.error(f"Não foi possível salvar a rubrica: {exc}")


def render_app() -> None:
    """Renderiza a aplicação Streamlit com 3 abas."""
    import streamlit as st

    st.set_page_config(
        page_title="Assistente de Suporte Técnico IA",
        page_icon="🤖",
        layout="wide",
    )

    st.title("Assistente de Suporte Técnico com IA e Engenharia de Prompt")
    st.caption(
        "Pipeline modular com adaptação por perfil/canal, versionamento controlado (v1 vs v2), "
        "cache determinístico e avaliação por rubrica."
    )

    storage = JsonHistoryStorage(DEFAULT_HISTORY_PATH)
    try:
        profiles = load_profiles_data()
        tickets = load_tickets_data()
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        st.error(f"Erro ao carregar dados iniciais: {exc}")
        return

    with st.sidebar:
        st.header("Painel do Pipeline")
        st.markdown(f"**Histórico JSON:** `{storage.file_path}`")
        st.markdown(f"**Perfis carregados:** `{len(profiles)}`")
        st.markdown(f"**Tickets carregados:** `{len(tickets)}`")
        st.markdown(f"**Versões de prompt:** `{', '.join(AVAILABLE_VERSIONS)}`")

    tab_generate, tab_compare, tab_history = st.tabs(
        [
            "Gerar Atendimento",
            "Comparação Controlada (v1 vs v2)",
            "Histórico e Auditoria",
        ]
    )

    # ABA 1 — GERAR ATENDIMENTO
    with tab_generate:
        st.subheader("Configuração do Atendimento")
        selected_profile = st.selectbox(
            "Perfil do Cliente",
            options=profiles,
            format_func=lambda p: f"{p.customer_id} — {p.name} ({p.technical_level} / {p.channel})",
            key="gen_profile",
        )
        _render_profile_badges(st, selected_profile)

        ticket_text = _render_ticket_selector(st, tickets, key_prefix="gen")

        col_type, col_ver, col_refresh = st.columns([2, 1, 1])
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
        with col_refresh:
            st.write("")
            force_refresh = st.checkbox(
                "Forçar nova chamada à API (ignorar cache)",
                value=False,
                key="gen_force_refresh",
            )

        if st.button("Gerar Resposta", type="primary", key="btn_generate"):
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

        if "last_generation" in st.session_state:
            st.divider()
            gen_result, gen_prompt = st.session_state["last_generation"]
            _render_output_and_rubric(
                st=st,
                result=gen_result,
                prompt_text=gen_prompt,
                storage=storage,
                form_key=f"rubric_gen_{gen_result.record.cache_key[:12]}",
            )

    # ABA 2 — COMPARAÇÃO CONTROLADA (v1 vs v2)
    with tab_compare:
        st.subheader("Comparação Lado a Lado (v1 Baseline vs v2 Otimizada)")
        st.caption(
            "Executa exatamente o mesmo cliente, ticket, provedor, modelo e temperatura "
            "alterando apenas a versão do prompt."
        )

        cmp_profile = st.selectbox(
            "Perfil do Cliente",
            options=profiles,
            format_func=lambda p: f"{p.customer_id} — {p.name} ({p.technical_level} / {p.channel})",
            key="cmp_profile",
        )
        _render_profile_badges(st, cmp_profile)

        cmp_ticket_text = _render_ticket_selector(st, tickets, key_prefix="cmp")

        col_ct, col_rf = st.columns([2, 1])
        with col_ct:
            cmp_type: ContentType = st.selectbox(
                "Tipo de Conteúdo",
                options=list(CONTENT_TYPES),
                format_func=lambda ct: f"{ct} — {CONTENT_TYPE_LABELS[ct]}",
                key="cmp_content_type",
            )
        with col_rf:
            st.write("")
            cmp_force_refresh = st.checkbox(
                "Forçar nova chamada à API (ambas versões)",
                value=False,
                key="cmp_force_refresh",
            )

        if st.button("Comparar v1 vs v2", type="primary", key="btn_compare"):
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
            st.divider()
            comparison_data = st.session_state["last_comparison"]
            col_v1, col_v2 = st.columns(2)

            with col_v1:
                st.markdown("### Versão `v1` (Baseline)")
                res_v1, prompt_v1 = comparison_data["v1"]
                _render_output_and_rubric(
                    st=st,
                    result=res_v1,
                    prompt_text=prompt_v1,
                    storage=storage,
                    form_key=f"rubric_cmp_v1_{res_v1.record.cache_key[:12]}",
                )

            with col_v2:
                st.markdown("### Versão `v2` (Otimizada)")
                res_v2, prompt_v2 = comparison_data["v2"]
                _render_output_and_rubric(
                    st=st,
                    result=res_v2,
                    prompt_text=prompt_v2,
                    storage=storage,
                    form_key=f"rubric_cmp_v2_{res_v2.record.cache_key[:12]}",
                )

    # ABA 3 — HISTÓRICO E AUDITORIA
    with tab_history:
        st.subheader(f"Histórico de Execuções (`{storage.file_path}`)")
        try:
            all_records = storage.load_all()
        except (ValueError, ValidationError) as exc:
            st.error(f"Erro ao carregar histórico JSON: {exc}")
            all_records = []

        metrics = summarize_history(all_records)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total de Execuções", metrics["total"])
        m2.metric("Avaliadas com Rubrica", metrics["evaluated"])
        m3.metric("Execuções v1", metrics["v1"])
        m4.metric("Execuções v2", metrics["v2"])

        if not all_records:
            st.info("Nenhuma execução registrada ainda em data/history.json.")
        else:
            f1, f2, f3, f4 = st.columns(4)
            customer_options = ["Todos"] + sorted({r.customer_id for r in all_records})
            type_options = ["Todos"] + list(CONTENT_TYPES)
            version_options = ["Todas"] + list(AVAILABLE_VERSIONS)

            selected_cust_filter = f1.selectbox("Filtrar por Cliente", customer_options)
            selected_type_filter = f2.selectbox("Filtrar por Tipo", type_options)
            selected_ver_filter = f3.selectbox("Filtrar por Versão", version_options)
            only_eval = f4.checkbox("Somente avaliadas", value=False)

            filtered_records = filter_history_records(
                all_records,
                customer_id=None if selected_cust_filter == "Todos" else selected_cust_filter,
                content_type=None if selected_type_filter == "Todos" else selected_type_filter,
                prompt_version=None if selected_ver_filter == "Todas" else selected_ver_filter,
                only_evaluated=only_eval,
            )

            st.download_button(
                label=f"Baixar JSON ({len(filtered_records)} registro(s))",
                data=export_history_json(filtered_records),
                file_name="history_export.json",
                mime="application/json",
            )

            for idx, rec in enumerate(reversed(filtered_records), start=1):
                eval_tag = "Avaliada" if rec.evaluation is not None else "Sem rubrica"
                header = (
                    f"#{idx} | {rec.timestamp} | {rec.customer_id} | "
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
                        st.write(
                            {
                                "clarity": ev.clarity.model_dump(),
                                "customer_fit": ev.customer_fit.model_dump(),
                                "actionability": ev.actionability.model_dump(),
                            }
                        )


if __name__ == "__main__":
    render_app()
