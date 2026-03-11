from __future__ import annotations

import time

import requests

from ...shared.observability import StructuredLogger


class OllamaLlmClient:
    def __init__(
        self,
        url: str,
        model: str,
        timeout_seconds: int = 30,
        retries: int = 3,
        logger: StructuredLogger | None = None,
    ) -> None:
        self._url = url
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._retries = retries
        self._logger = logger or StructuredLogger()

    def generate(self, prompt: str, user_input: str) -> str:
        for i in range(self._retries):
            started_at = time.perf_counter()
            try:
                response = requests.post(
                    self._url,
                    json={
                        "model": self._model,
                        "prompt": f"{prompt}\nUser: {user_input}",
                        "stream": False,
                    },
                    timeout=self._timeout_seconds,
                )
                response.raise_for_status()
                duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
                text = response.json().get("response", "").strip()
                if text:
                    return text

                self._logger.warning(
                    "llm.empty_response",
                    attempt=i + 1,
                    duration_ms=duration_ms,
                )
            except Exception as exc:
                duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
                self._logger.warning(
                    "llm.request_failed",
                    attempt=i + 1,
                    retries=self._retries,
                    duration_ms=duration_ms,
                    error=str(exc),
                )

        return ""
