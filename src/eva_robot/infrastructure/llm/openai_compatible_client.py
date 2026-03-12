from __future__ import annotations

import time
from typing import Any

import requests

from ...shared.observability import StructuredLogger
from ...shared.openai_compatible import (
    extract_openai_model_ids,
    normalize_openai_models_url,
    normalize_openai_responses_url,
    resolve_openai_model,
)


def _extract_text_content(payload: dict[str, Any]) -> str:
    output = payload.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue

            content = item.get("content")
            if not isinstance(content, list):
                continue

            for block in content:
                if not isinstance(block, dict):
                    continue
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "".join(parts).strip()

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message", {})
            if isinstance(message, dict):
                content = message.get("content", "")
                if isinstance(content, str):
                    return content.strip()

    return str(payload.get("output_text", "")).strip()


class OpenAiCompatibleLlmClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: int = 30,
        retries: int = 3,
        logger: StructuredLogger | None = None,
    ) -> None:
        self._url = normalize_openai_responses_url(base_url)
        self._models_url = normalize_openai_models_url(base_url)
        self._api_key = api_key
        self._requested_model = model
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._retries = retries
        self._logger = logger or StructuredLogger()
        self._model_resolution_attempted = False

    def generate(self, prompt: str, user_input: str) -> str:
        self._resolve_model_alias()

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "input": [
                {
                    "role": "user",
                    "content": f"{prompt.strip()}\nUser: {user_input}",
                }
            ],
        }

        for i in range(self._retries):
            started_at = time.perf_counter()
            try:
                response = requests.post(
                    self._url,
                    headers=headers,
                    json=payload,
                    timeout=self._timeout_seconds,
                )
                response.raise_for_status()
                duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
                text = _extract_text_content(response.json())
                if text:
                    return text

                self._logger.warning(
                    "llm.empty_response",
                    provider="openai_compatible",
                    model=self._model,
                    attempt=i + 1,
                    duration_ms=duration_ms,
                )
            except Exception as exc:
                duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
                self._logger.warning(
                    "llm.request_failed",
                    provider="openai_compatible",
                    model=self._model,
                    attempt=i + 1,
                    retries=self._retries,
                    duration_ms=duration_ms,
                    error=str(exc),
                )

        return ""

    def _resolve_model_alias(self) -> None:
        if self._model_resolution_attempted:
            return

        self._model_resolution_attempted = True

        try:
            response = requests.get(
                self._models_url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=min(self._timeout_seconds, 15),
            )
            response.raise_for_status()
            advertised_models = extract_openai_model_ids(response.json())
            resolved_model = resolve_openai_model(
                self._requested_model, advertised_models
            )
            if resolved_model and resolved_model != self._requested_model:
                self._model = resolved_model
                self._logger.warning(
                    "llm.model_resolved",
                    provider="openai_compatible",
                    requested_model=self._requested_model,
                    resolved_model=resolved_model,
                )
            elif advertised_models and self._requested_model not in advertised_models:
                self._logger.warning(
                    "llm.model_unlisted",
                    provider="openai_compatible",
                    requested_model=self._requested_model,
                    available_models=advertised_models[:5],
                )
        except Exception as exc:
            self._logger.warning(
                "llm.model_resolution_failed",
                provider="openai_compatible",
                requested_model=self._requested_model,
                error=str(exc),
            )
