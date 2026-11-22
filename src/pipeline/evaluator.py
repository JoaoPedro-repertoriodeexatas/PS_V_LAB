from __future__ import annotations

from src.models import CriterionScore, ExecutionRecord, RubricEvaluation
from src.services.storage import JsonHistoryStorage


def build_rubric_evaluation(
    clarity_score: int,
    clarity_justification: str,
    customer_fit_score: int,
    customer_fit_justification: str,
    actionability_score: int,
    actionability_justification: str,
) -> RubricEvaluation:
    """Constrói e valida via Pydantic uma avaliação completa da rubrica de 3 critérios."""
    return RubricEvaluation(
        clarity=CriterionScore(
            score=clarity_score,
            justification=clarity_justification,
        ),
        customer_fit=CriterionScore(
            score=customer_fit_score,
            justification=customer_fit_justification,
        ),
        actionability=CriterionScore(
            score=actionability_score,
            justification=actionability_justification,
        ),
    )


def evaluate_execution(
    cache_key: str,
    evaluation: RubricEvaluation,
    storage: JsonHistoryStorage | None = None,
) -> ExecutionRecord:
    """Persiste a avaliação de rubrica na execução correspondente do histórico JSON."""
    resolved_storage = storage or JsonHistoryStorage()
    return resolved_storage.attach_evaluation(cache_key=cache_key, evaluation=evaluation)
