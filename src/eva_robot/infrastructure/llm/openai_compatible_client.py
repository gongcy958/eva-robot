from __future__ import annotations

import time
from typing import Any

import requests

from ...shared.observability import StructuredLogger


def _normalize_responses_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/responses"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/responses"
    return f"{normalized}/v1/responses"


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
        self._url = _normalize_responses_url(base_url)
        self._api_key = api_key
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._retries = retries
        self._logger = logger or StructuredLogger()

    def generate(self, prompt: str, user_input: str) -> str:
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
