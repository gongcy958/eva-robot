from __future__ import annotations

import re
from typing import Any, Iterable


_DATE_SUFFIX_PATTERN = re.compile(r"-(\d{4}-\d{2}-\d{2})$")


def normalize_openai_models_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/models"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/models"
    return f"{normalized}/v1/models"


def normalize_openai_responses_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/responses"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/responses"
    return f"{normalized}/v1/responses"


def extract_openai_model_ids(payload: dict[str, Any]) -> list[str]:
    data = payload.get("data")
    if not isinstance(data, list):
        return []

    model_ids: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if isinstance(model_id, str) and model_id.strip():
            model_ids.append(model_id.strip())

    return model_ids


def resolve_openai_model(
    requested_model: str, available_models: Iterable[str]
) -> str | None:
    requested = requested_model.strip()
    advertised = [model.strip() for model in available_models if model and model.strip()]
    if not requested or not advertised:
        return None

    if requested in advertised:
        return requested

    direct_matches = [
        model
        for model in advertised
        if _is_general_model(model) and model.startswith(f"{requested}-")
    ]
    if direct_matches:
        return max(direct_matches, key=_general_model_sort_key)

    if not _looks_like_general_model(requested):
        return None

    requested_family = _general_model_family(requested)
    family_matches = [
        model
        for model in advertised
        if _is_general_model(model)
        and _general_model_family(model) == requested_family
    ]
    if family_matches:
        return max(family_matches, key=_general_model_sort_key)

    return None


def _looks_like_general_model(model: str) -> bool:
    normalized = _strip_date_suffix(model.strip().lower())
    return normalized.startswith("gpt-") and "codex" not in normalized


def _is_general_model(model: str) -> bool:
    return _looks_like_general_model(model)


def _general_model_family(model: str) -> str:
    normalized = _strip_date_suffix(model.strip().lower())
    if "." in normalized:
        return normalized.split(".", 1)[0]
    return normalized


def _general_model_sort_key(model: str) -> tuple[tuple[int, ...], str, str]:
    normalized = _strip_date_suffix(model.strip().lower())
    version = _extract_version_tuple(normalized)
    date_suffix = _extract_date_suffix(model) or ""
    return version, date_suffix, normalized


def _extract_version_tuple(model: str) -> tuple[int, ...]:
    if not model.startswith("gpt-"):
        return ()

    suffix = model[len("gpt-") :]
    version_label = suffix.split("-", 1)[0]
    version_parts: list[int] = []
    for part in version_label.split("."):
        if not part.isdigit():
            return ()
        version_parts.append(int(part))
    return tuple(version_parts)


def _strip_date_suffix(model: str) -> str:
    match = _DATE_SUFFIX_PATTERN.search(model)
    if not match:
        return model
    return model[: match.start()]


def _extract_date_suffix(model: str) -> str | None:
    match = _DATE_SUFFIX_PATTERN.search(model.strip())
    if not match:
        return None
    return match.group(1)
