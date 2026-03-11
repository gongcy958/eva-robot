from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import requests

from .config import AppConfig
from .observability import StructuredLogger


Level = Literal["info", "warning", "error"]


@dataclass(frozen=True)
class PreflightFinding:
    level: Level
    code: str
    message: str


def _normalize_openai_models_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/models"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/models"
    return f"{normalized}/v1/models"


def _normalize_openai_responses_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/responses"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/responses"
    return f"{normalized}/v1/responses"


def _normalize_ollama_tags_url(generate_url: str) -> str:
    normalized = generate_url.rstrip("/")
    if normalized.endswith("/api/generate"):
        return f"{normalized[:-len('/api/generate')]}/api/tags"
    if normalized.endswith("/api"):
        return f"{normalized}/tags"
    return f"{normalized}/api/tags"


class StartupPreflight:
    def __init__(self, config: AppConfig, logger: StructuredLogger | None = None) -> None:
        self._config = config
        self._logger = logger or StructuredLogger()

    def run(self) -> None:
        if self._config.skip_startup_checks:
            self._emit(
                [
                    PreflightFinding(
                        "warning",
                        "startup.skip",
                        "Startup checks skipped by configuration.",
                    )
                ]
            )
            return

        findings: list[PreflightFinding] = []
        findings.extend(self._check_whisper_model())

        provider = self._config.resolved_llm_provider()
        if provider == "openai_compatible":
            findings.extend(self._check_openai_compatible())
        elif provider == "ollama":
            findings.extend(self._check_ollama())
        else:
            findings.append(
                PreflightFinding(
                    "error",
                    "llm.provider.unsupported",
                    f"Unsupported LLM provider: {self._config.llm_provider}",
                )
            )

        self._emit(findings)

        if any(finding.level == "error" for finding in findings):
            raise RuntimeError(
                "Startup preflight failed. Fix the errors above or set "
                "SKIP_STARTUP_CHECKS=true to bypass temporarily."
            )

    def _emit(self, findings: list[PreflightFinding]) -> None:
        if not findings:
            findings = [
                PreflightFinding("info", "startup.ok", "All startup checks passed.")
            ]

        for finding in findings:
            print(f"[Preflight][{finding.level.upper()}] {finding.message}")
            log_fn = getattr(self._logger, finding.level)
            log_fn("preflight.check", code=finding.code, message=finding.message)

    def _check_whisper_model(self) -> list[PreflightFinding]:
        model_path = self._config.whisper_model_path.strip()
        path = Path(model_path).expanduser()
        looks_like_local_path = model_path.startswith(("~", "/", ".")) or "/" in model_path

        if looks_like_local_path:
            if path.exists():
                return [
                    PreflightFinding(
                        "info",
                        "whisper.path.ok",
                        f"Whisper model path is available: {path}",
                    )
                ]
            return [
                PreflightFinding(
                    "error",
                    "whisper.path.missing",
                    f"Whisper model path does not exist: {path}",
                )
            ]

        return [
            PreflightFinding(
                "info",
                "whisper.path.named_model",
                f"Whisper model uses provider name: {model_path}",
            )
        ]

    def _check_openai_compatible(self) -> list[PreflightFinding]:
        if not self._config.llm_api_key:
            return [
                PreflightFinding(
                    "error",
                    "llm.api_key.missing",
                    "LLM_API_KEY is missing for openai_compatible provider.",
                )
            ]

        findings: list[PreflightFinding] = []
        headers = {"Authorization": f"Bearer {self._config.llm_api_key}"}
        timeout_seconds = min(self._config.llm_timeout_seconds, 15)

        try:
            models_response = requests.get(
                _normalize_openai_models_url(self._config.llm_base_url),
                headers=headers,
                timeout=timeout_seconds,
            )
            models_response.raise_for_status()
            findings.append(
                PreflightFinding(
                    "info",
                    "llm.models.ok",
                    "OpenAI-compatible endpoint and API key look valid.",
                )
            )
        except Exception as exc:
            return [
                PreflightFinding(
                    "error",
                    "llm.models.failed",
                    f"Failed to reach OpenAI-compatible models endpoint: {exc}",
                )
            ]

        if not self._config.llm_preflight_probe:
            findings.append(
                PreflightFinding(
                    "warning",
                    "llm.probe.skipped",
                    "Model probe skipped because LLM_PREFLIGHT_PROBE=false.",
                )
            )
            return findings

        try:
            response = requests.post(
                _normalize_openai_responses_url(self._config.llm_base_url),
                headers={**headers, "Content-Type": "application/json"},
                json={
                    "model": self._config.resolved_llm_model(),
                    "input": [{"role": "user", "content": "ping"}],
                    "max_output_tokens": 1,
                },
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            findings.append(
                PreflightFinding(
                    "info",
                    "llm.probe.ok",
                    f"Remote model probe succeeded: {self._config.resolved_llm_model()}",
                )
            )
        except Exception as exc:
            findings.append(
                PreflightFinding(
                    "error",
                    "llm.probe.failed",
                    f"Remote model probe failed for {self._config.resolved_llm_model()}: {exc}",
                )
            )

        return findings

    def _check_ollama(self) -> list[PreflightFinding]:
        tags_url = _normalize_ollama_tags_url(self._config.ollama_url)
        timeout_seconds = min(self._config.llm_timeout_seconds, 15)

        try:
            response = requests.get(tags_url, timeout=timeout_seconds)
            response.raise_for_status()
        except Exception as exc:
            return [
                PreflightFinding(
                    "error",
                    "ollama.tags.failed",
                    f"Failed to reach Ollama at {tags_url}: {exc}",
                )
            ]

        findings = [
            PreflightFinding(
                "info",
                "ollama.tags.ok",
                f"Ollama endpoint is reachable: {tags_url}",
            )
        ]

        model_names = {
            model.get("name", "")
            for model in response.json().get("models", [])
            if isinstance(model, dict)
        }
        if self._config.ollama_model in model_names:
            findings.append(
                PreflightFinding(
                    "info",
                    "ollama.model.ok",
                    f"Ollama model is available: {self._config.ollama_model}",
                )
            )
        else:
            findings.append(
                PreflightFinding(
                    "error",
                    "ollama.model.missing",
                    f"Ollama model not found locally: {self._config.ollama_model}",
                )
            )

        return findings
