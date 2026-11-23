from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

TechnicalLevel = Literal["leigo", "intermediário", "avançado"]
CommunicationStyle = Literal["direto", "detalhado", "didático"]
Channel = Literal["chat", "e-mail", "telefone"]
ContentType = Literal["diagnosis", "resolution", "clarification", "flow_summary"]


def _utc_now_iso() -> str:
    """Retorna o timestamp atual em formato ISO 8601 UTC."""
    return datetime.now(timezone.utc).isoformat()


class CustomerProfile(BaseModel):
    """Perfil do cliente utilizado para adaptar linguagem, profundidade e canal."""

    model_config = ConfigDict(str_strip_whitespace=True)

    customer_id: str = Field(..., min_length=1, description="Identificador único do cliente.")
    name: str = Field(..., min_length=1, description="Nome do cliente.")
    technical_level: TechnicalLevel = Field(..., description="Nível de conhecimento técnico.")
    communication_style: CommunicationStyle = Field(..., description="Estilo preferido de comunicação.")
    channel: Channel = Field(..., description="Canal de atendimento utilizado.")


class TicketItem(BaseModel):
    """Ticket de suporte técnico de entrada para análise do assistente."""

    model_config = ConfigDict(str_strip_whitespace=True)

    ticket_id: str = Field(..., min_length=1, description="Identificador único do ticket.")
    title: str = Field(..., min_length=1, description="Título resumido do incidente.")
    description: str = Field(..., min_length=1, description="Descrição detalhada do problema.")


class CriterionScore(BaseModel):
    """Nota e justificativa para um critério individual da rubrica."""

    model_config = ConfigDict(str_strip_whitespace=True)

    score: int = Field(..., ge=1, le=5, description="Nota inteira de 1 (insuficiente) a 5 (excelente).")
    justification: str = Field(
        ...,
        min_length=5,
        description="Justificativa curta explicando a nota atribuída.",
    )


class RubricEvaluation(BaseModel):
    """Rubrica fixa obrigatória com 3 critérios de avaliação comparativa."""

    clarity: CriterionScore = Field(..., description="Clareza, organização e ausência de ambiguidade.")
    customer_fit: CriterionScore = Field(
        ...,
        description="Adequação do vocabulário e profundidade ao perfil do cliente.",
    )
    actionability: CriterionScore = Field(
        ...,
        description="Capacidade de orientar próximos passos claros para cliente ou atendente.",
    )


class ExecutionRecord(BaseModel):
    """Registro persistido de uma execução do pipeline de geração de conteúdo."""

    timestamp: str = Field(
        default_factory=_utc_now_iso,
        description="Data e hora da execução em ISO 8601 UTC.",
    )
    cache_key: str = Field(..., min_length=1, description="Chave determinística para reutilização de cache.")
    customer_id: str = Field(..., min_length=1, description="ID do perfil de cliente utilizado.")
    ticket: str = Field(..., min_length=1, description="Texto ou identificador do ticket analisado.")
    prompt_version: str = Field(..., min_length=1, description="Versão do prompt utilizada na execução.")
    content_type: ContentType = Field(..., description="Tipo de conteúdo gerado.")
    provider: str = Field(..., min_length=1, description="Provedor do modelo de linguagem.")
    model: str = Field(..., min_length=1, description="Identificador do modelo utilizado.")
    temperature: float = Field(..., ge=0.0, le=2.0, description="Temperatura configurada na chamada ao LLM.")
    output: dict[str, Any] | str = Field(..., description="Saída estruturada ou textual gerada pelo modelo.")
    latency_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Tempo de geração da resposta via API em milissegundos.",
    )
    evaluation: RubricEvaluation | None = Field(
        default=None,
        description="Avaliação opcional baseada na rubrica fixa de 3 critérios.",
    )

    @field_validator("latency_ms", mode="after")
    @classmethod
    def _round_latency_ms(cls, value: float | None) -> float | None:
        if value is None:
            return None
        return round(float(value), 2)
