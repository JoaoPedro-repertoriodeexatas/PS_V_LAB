from __future__ import annotations

import hashlib
import json

from src.models import ContentType, ExecutionRecord


def generate_cache_key(
    customer_id: str,
    ticket: str,
    content_type: ContentType,
    prompt_version: str,
    provider: str,
    model: str,
    temperature: float,
) -> str:
    """Gera um hash SHA-256 determinístico a partir dos parâmetros canônicos da execução."""
    canonical_payload = {
        "customer_id": customer_id.strip(),
        "ticket": ticket.strip(),
        "content_type": content_type.strip(),
        "prompt_version": prompt_version.strip().lower(),
        "provider": provider.strip().lower(),
        "model": model.strip(),
        "temperature": round(float(temperature), 4),
    }
    canonical_json = json.dumps(
        canonical_payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def find_cached_execution(
    records: list[ExecutionRecord],
    cache_key: str,
) -> ExecutionRecord | None:
    """Retorna o registro equivalente mais recente no histórico para o cache_key informado."""
    for record in reversed(records):
        if record.cache_key == cache_key:
            return record
    return None
