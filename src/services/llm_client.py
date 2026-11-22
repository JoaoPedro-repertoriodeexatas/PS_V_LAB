from __future__ import annotations

import os
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from google import genai
from google.genai import types

PLACEHOLDER_API_KEY = "sua_chave_api_aqui"
DEFAULT_PROVIDER = "google"
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.2


class LLMServiceError(RuntimeError):
    """Exceção encapsulada para falhas de comunicação ou geração na API do LLM."""


class BaseLLMClient(ABC):
    """Interface base desacoplada para clientes de modelos de linguagem."""

    @property
    @abstractmethod
    def provider(self) -> str:
        """Nome do provedor de LLM (ex.: 'google')."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Identificador do modelo utilizado (ex.: 'gemini-2.5-flash')."""

    @property
    @abstractmethod
    def temperature(self) -> float:
        """Temperatura configurada para geração."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Envia o prompt ao modelo e retorna a resposta textual gerada."""


class GeminiClient(BaseLLMClient):
    """Cliente oficial para a API Gemini utilizando o SDK google-genai."""

    def __init__(
        self,
        api_key: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        load_env: bool = True,
    ) -> None:
        if load_env:
            load_dotenv(override=False)

        resolved_api_key = (
            api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")
        ).strip()
        if not resolved_api_key or resolved_api_key.lower() == PLACEHOLDER_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY ausente ou com valor placeholder. "
                "Copie '.env.example' para '.env' e configure uma chave válida da API Gemini."
            )

        self._provider = (
            provider if provider is not None else os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)
        ).strip() or DEFAULT_PROVIDER
        self._model = (
            model if model is not None else os.getenv("LLM_MODEL", DEFAULT_MODEL)
        ).strip() or DEFAULT_MODEL

        if temperature is not None:
            resolved_temperature = float(temperature)
        else:
            raw_temp = os.getenv("LLM_TEMPERATURE", str(DEFAULT_TEMPERATURE)).strip()
            try:
                resolved_temperature = float(raw_temp)
            except ValueError as exc:
                raise ValueError(
                    f"Valor inválido para LLM_TEMPERATURE: '{raw_temp}'. Informe um número entre 0.0 e 2.0."
                ) from exc

        if not (0.0 <= resolved_temperature <= 2.0):
            raise ValueError(
                f"Temperatura fora do intervalo permitido (0.0 a 2.0): {resolved_temperature}."
            )

        self._temperature = resolved_temperature
        self._client = genai.Client(api_key=resolved_api_key)

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
        if not prompt or not prompt.strip():
            raise ValueError("O prompt enviado ao LLM não pode estar vazio.")

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=self._temperature),
            )
        except Exception as exc:
            raise LLMServiceError(
                f"Falha ao consultar a API Gemini ({self._provider}/{self._model}): {exc}"
            ) from exc

        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise LLMServiceError(
                f"A API Gemini ({self._provider}/{self._model}) retornou uma resposta vazia."
            )
        return text
