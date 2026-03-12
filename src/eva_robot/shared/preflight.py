from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import requests

from .config import AppConfig
from .observability import StructuredLogger
from .openai_compatible import (
    extract_openai_model_ids,
    normalize_openai_models_url,
    normalize_openai_responses_url,
    resolve_openai_model,
)


Level = Literal["info", "warning", "error"]


@dataclass(frozen=True)
class PreflightFinding:
    level: Level
    code: str
    message: str


@dataclass(frozen=True)
class StartupPreflightResult:
    effective_llm_provider: str
    used_fallback: bool = False


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

    def run(self) -> StartupPreflightResult:
        provider = self._config.resolved_llm_provider()
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
            return StartupPreflightResult(effective_llm_provider=provider)

        findings: list[PreflightFinding] = []
        blocking_findings: list[PreflightFinding] = []

        whisper_findings = self._check_whisper_model()
        findings.extend(whisper_findings)
        blocking_findings.extend(
            finding for finding in whisper_findings if finding.level == "error"
        )

        used_fallback = False
        effective_provider = provider
        if provider == "openai_compatible":
            openai_findings = self._check_openai_compatible()
            if self._has_errors(openai_findings):
                ollama_findings = self._check_ollama()
                if self._has_errors(ollama_findings):
                    findings.extend(openai_findings)
                    findings.extend(ollama_findings)
                    blocking_findings.extend(
                        finding
                        for finding in openai_findings + ollama_findings
                        if finding.level == "error"
                    )
                else:
                    findings.extend(self._downgrade_errors(openai_findings))
                    findings.extend(ollama_findings)
                    findings.append(
                        PreflightFinding(
                            "warning",
                            "llm.fallback.to_ollama",
                            "OpenAI-compatible provider is unavailable; "
                            f"falling back to local Ollama model: {self._config.ollama_model}",
                        )
                    )
                    used_fallback = True
                    effective_provider = "ollama"
            else:
                findings.extend(openai_findings)
        elif provider == "ollama":
            ollama_findings = self._check_ollama()
            findings.extend(ollama_findings)
            blocking_findings.extend(
                finding for finding in ollama_findings if finding.level == "error"
            )
        else:
            unsupported = PreflightFinding(
                "error",
                "llm.provider.unsupported",
                f"Unsupported LLM provider: {self._config.llm_provider}",
            )
            findings.append(unsupported)
            blocking_findings.append(unsupported)

        self._emit(findings)

        if blocking_findings:
            raise RuntimeError(
                "Startup preflight failed. Fix the errors above or set "
                "SKIP_STARTUP_CHECKS=true to bypass temporarily."
            )

        return StartupPreflightResult(
            effective_llm_provider=effective_provider,
            used_fallback=used_fallback,
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

    @staticmethod
    def _has_errors(findings: list[PreflightFinding]) -> bool:
        return any(finding.level == "error" for finding in findings)

    @staticmethod
    def _downgrade_errors(findings: list[PreflightFinding]) -> list[PreflightFinding]:
        downgraded: list[PreflightFinding] = []
        for finding in findings:
            if finding.level == "error":
                downgraded.append(
                    PreflightFinding(
                        "warning",
                        finding.code,
                        finding.message,
                    )
                )
            else:
                downgraded.append(finding)
        return downgraded

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
        requested_model = self._config.resolved_llm_model()
        probe_model = requested_model
        advertised_models: list[str] = []

        try:
            models_response = requests.get(
                normalize_openai_models_url(self._config.llm_base_url),
                headers=headers,
                timeout=timeout_seconds,
            )
            models_response.raise_for_status()
            advertised_models = extract_openai_model_ids(models_response.json())
            findings.append(
                PreflightFinding(
                    "info",
                    "llm.models.ok",
                    "OpenAI-compatible endpoint and API key look valid.",
                )
            )
            resolved_model = resolve_openai_model(requested_model, advertised_models)
            if resolved_model and resolved_model != requested_model:
                probe_model = resolved_model
                findings.append(
                    PreflightFinding(
                        "warning",
                        "llm.model.resolved",
                        "Configured model "
                        f"{requested_model} is not advertised by the provider; "
                        f"using {resolved_model} instead.",
                    )
                )
            elif advertised_models and requested_model not in advertised_models:
                findings.append(
                    PreflightFinding(
                        "warning",
                        "llm.model.unlisted",
                        "Configured model "
                        f"{requested_model} is not advertised by the provider.",
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
                normalize_openai_responses_url(self._config.llm_base_url),
                headers={**headers, "Content-Type": "application/json"},
                json={
                    "model": probe_model,
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
                    "Remote model probe succeeded: "
                    f"{probe_model}"
                    + (
                        f" (requested {requested_model})"
                        if probe_model != requested_model
                        else ""
                    ),
                )
            )
        except Exception as exc:
            findings.append(
                PreflightFinding(
                    "error",
                    "llm.probe.failed",
                    "Remote model probe failed for "
                    f"{probe_model}"
                    + (
                        f" (requested {requested_model})"
                        if probe_model != requested_model
                        else ""
                    )
                    + f": {exc}",
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
