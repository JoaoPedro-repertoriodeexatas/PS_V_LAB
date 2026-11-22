from __future__ import annotations

from src.models import ContentType

VERSION = "v1"
DESCRIPTION = "Baseline minimalista sem calibração profunda por canal ou nível técnico."

PERSONA = (
    "Você é um assistente de suporte técnico responsável por analisar tickets de clientes "
    "e produzir respostas úteis com base nas informações fornecidas."
)

GENERAL_CONSTRAINTS: tuple[str, ...] = (
    "Baseie sua resposta apenas nos fatos informados no ticket.",
    "Não invente logs, erros ou dados que não foram mencionados.",
)

TEMPLATES: dict[ContentType, dict[str, str]] = {
    "diagnosis": {
        "task_instruction": (
            "Analise o problema relatado no ticket e produza um diagnóstico técnico "
            "organizado em hipóteses, evidências disponíveis, dados faltantes e conclusão provável."
        ),
        "output_format": (
            "Organize a resposta em texto estruturado com as seções: "
            "Hipóteses, Evidências Disponíveis, Dados Faltantes e Conclusão Provável."
        ),
    },
    "resolution": {
        "task_instruction": (
            "Elabore um passo a passo de resolução para ajudar a resolver o problema descrito no ticket."
        ),
        "output_format": (
            "Apresente uma lista numerada com os passos recomendados para resolução."
        ),
    },
    "clarification": {
        "task_instruction": (
            "Liste perguntas de esclarecimento necessárias para coletar informações "
            "faltantes e avançar no atendimento do ticket."
        ),
        "output_format": (
            "Apresente uma lista simples de perguntas objetivas ao cliente."
        ),
    },
    "flow_summary": {
        "task_instruction": (
            "Resuma o fluxo de triagem e decisão para o atendimento deste ticket."
        ),
        "output_format": (
            "Apresente um fluxograma ASCII simples ou uma descrição estruturada das decisões."
        ),
    },
}
