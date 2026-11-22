from __future__ import annotations

import json
from pathlib import Path

from src.models import ExecutionRecord, RubricEvaluation

DEFAULT_HISTORY_PATH = Path("data/history.json")


class JsonHistoryStorage:
    """Persistência segura e validada do histórico de execuções em arquivo JSON."""

    def __init__(self, file_path: Path | str = DEFAULT_HISTORY_PATH) -> None:
        self.file_path = Path(file_path)

    def load_all(self) -> list[ExecutionRecord]:
        """Carrega e valida todos os registros do histórico JSON."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            return []

        raw_content = self.file_path.read_text(encoding="utf-8")
        if not raw_content.strip():
            return []

        parsed = json.loads(raw_content)
        if not isinstance(parsed, list):
            raise ValueError(
                f"O arquivo de histórico '{self.file_path}' deve conter uma lista JSON."
            )

        return [ExecutionRecord.model_validate(item) for item in parsed]

    def _write_all(self, records: list[ExecutionRecord]) -> Path:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(
            [record.model_dump(mode="json") for record in records],
            indent=2,
            ensure_ascii=False,
        )
        temp_path = self.file_path.with_suffix(f"{self.file_path.suffix}.tmp")
        temp_path.write_text(f"{serialized}\n", encoding="utf-8")
        temp_path.replace(self.file_path)
        return self.file_path

    def save(self, record: ExecutionRecord) -> Path:
        """Adiciona um novo ExecutionRecord ao histórico e grava o arquivo JSON."""
        records = self.load_all()
        records.append(record)
        return self._write_all(records)

    def attach_evaluation(
        self,
        cache_key: str,
        evaluation: RubricEvaluation,
    ) -> ExecutionRecord:
        """Anexa uma RubricEvaluation ao registro mais recente com o cache_key informado."""
        records = self.load_all()
        for idx in range(len(records) - 1, -1, -1):
            if records[idx].cache_key == cache_key:
                updated_record = records[idx].model_copy(update={"evaluation": evaluation})
                records[idx] = updated_record
                self._write_all(records)
                return updated_record

        raise KeyError(
            f"Nenhum registro encontrado em '{self.file_path}' para cache_key='{cache_key}'."
        )
