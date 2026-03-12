from __future__ import annotations

from ...application.services.ports import LlmService
from ...shared.observability import StructuredLogger


class FailoverLlmClient:
    def __init__(
        self,
        primary: LlmService,
        fallback: LlmService,
        primary_provider: str,
        fallback_provider: str,
        logger: StructuredLogger | None = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._primary_provider = primary_provider
        self._fallback_provider = fallback_provider
        self._fallback_active = False
        self._logger = logger or StructuredLogger()

    def generate(self, prompt: str, user_input: str) -> str:
        if self._fallback_active:
            return self._fallback.generate(prompt, user_input)

        primary_response = self._primary.generate(prompt, user_input).strip()
        if primary_response:
            return primary_response

        self._fallback_active = True
        self._logger.warning(
            "llm.failover_activated",
            from_provider=self._primary_provider,
            to_provider=self._fallback_provider,
        )

        fallback_response = self._fallback.generate(prompt, user_input).strip()
        if fallback_response:
            return fallback_response

        self._logger.error(
            "llm.failover_failed",
            from_provider=self._primary_provider,
            to_provider=self._fallback_provider,
        )
        return ""
