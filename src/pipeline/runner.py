from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

from src.models import ContentType, CustomerProfile, ExecutionRecord
from src.pipeline.cache import find_cached_execution, generate_cache_key
from src.prompts import get_prompt_for_content_type
from src.services.llm_client import BaseLLMClient, GeminiClient
from src.services.storage import JsonHistoryStorage


@dataclass(frozen=True)
class PipelineExecutionResult:
    """Resultado consolidado de uma execução do pipeline de suporte."""

    record: ExecutionRecord
    from_cache: bool
    storage_path: Path
    elapsed_ms: float = 0.0


def execute_pipeline(
    profile: CustomerProfile,
    ticket: str,
    content_type: ContentType,
    prompt_version: str = "v2",
    force_refresh: bool = False,
    llm_client: BaseLLMClient | None = None,
    storage: JsonHistoryStorage | None = None,
) -> PipelineExecutionResult:
    """Orquestra cache, montagem de prompt, geração via LLM e persistência em JSON."""
    resolved_storage = storage or JsonHistoryStorage()
    resolved_client = llm_client or GeminiClient()

    cache_key = generate_cache_key(
        customer_id=profile.customer_id,
        ticket=ticket,
        content_type=content_type,
        prompt_version=prompt_version,
        provider=resolved_client.provider,
        model=resolved_client.model,
        temperature=resolved_client.temperature,
    )

    if not force_refresh:
        cache_t0 = time.perf_counter()
        existing_records = resolved_storage.load_all()
        cached_record = find_cached_execution(existing_records, cache_key)
        if cached_record is not None:
            cache_elapsed_ms = round((time.perf_counter() - cache_t0) * 1000, 2)
            return PipelineExecutionResult(
                record=cached_record,
                from_cache=True,
                storage_path=resolved_storage.file_path,
                elapsed_ms=cache_elapsed_ms,
            )

    prompt = get_prompt_for_content_type(
        content_type=content_type,
        profile=profile,
        ticket=ticket,
        version=prompt_version,
    )
    api_t0 = time.perf_counter()
    generated_output = resolved_client.generate(prompt)
    api_latency_ms = round((time.perf_counter() - api_t0) * 1000, 2)

    record = ExecutionRecord(
        cache_key=cache_key,
        customer_id=profile.customer_id,
        ticket=ticket.strip(),
        prompt_version=prompt_version.strip().lower(),
        content_type=content_type,
        provider=resolved_client.provider,
        model=resolved_client.model,
        temperature=resolved_client.temperature,
        output=generated_output,
        latency_ms=api_latency_ms,
    )
    saved_path = resolved_storage.save(record)

    return PipelineExecutionResult(
        record=record,
        from_cache=False,
        storage_path=saved_path,
        elapsed_ms=api_latency_ms,
    )
