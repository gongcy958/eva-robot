from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class AudioTuningRecommendation:
    ambient_p50: float
    ambient_p80: float
    ambient_p95: float
    speech_p25: float
    speech_p50: float
    speech_p90: float
    speech_to_noise_ratio: float
    room_variability_ratio: float
    silence_threshold: float
    ambient_noise_seconds: float
    speech_start_threshold_multiplier: float
    speech_end_threshold_multiplier: float
    speech_start_chunks: int
    preroll_duration_seconds: float
    followup_cooldown_seconds: float
    notes: tuple[str, ...]


def recommend_voice_frontend_config(
    ambient_levels: Sequence[float],
    speech_levels: Sequence[float],
) -> AudioTuningRecommendation:
    ambient = _coerce_levels(ambient_levels)
    speech = _coerce_levels(speech_levels)
    if ambient.size == 0:
        raise ValueError("ambient_levels must not be empty")
    if speech.size == 0:
        raise ValueError("speech_levels must not be empty")

    ambient_p50 = _percentile(ambient, 50)
    ambient_p80 = _percentile(ambient, 80)
    ambient_p95 = _percentile(ambient, 95)
    speech_p25 = _percentile(speech, 25)
    speech_p50 = _percentile(speech, 50)
    speech_p90 = _percentile(speech, 90)

    speech_to_noise_ratio = speech_p50 / max(ambient_p95, 1e-4)
    room_variability_ratio = ambient_p95 / max(ambient_p50, 1e-4)

    silence_floor = max(0.008, ambient_p95 * 1.12)
    silence_ceiling = max(0.012, speech_p25 * 0.42)
    if silence_floor <= silence_ceiling:
        silence_threshold = (silence_floor + silence_ceiling) / 2
    else:
        silence_threshold = max(0.008, min(silence_floor, speech_p25 * 0.5))
    silence_threshold = round(_clamp(silence_threshold, 0.008, 0.08), 4)

    ambient_noise_seconds = 0.6 if room_variability_ratio >= 1.8 else 0.4

    start_target = max(silence_threshold * 1.1, speech_p25 * 0.62)
    start_multiplier = start_target / max(ambient_p80, 1e-4)
    speech_start_threshold_multiplier = round(
        _clamp(start_multiplier, 1.6, 3.2),
        2,
    )

    end_target = max(silence_threshold, speech_p25 * 0.46)
    end_multiplier = end_target / max(ambient_p80, 1e-4)
    speech_end_threshold_multiplier = round(
        _clamp(
            min(end_multiplier, speech_start_threshold_multiplier - 0.25),
            1.2,
            max(1.2, speech_start_threshold_multiplier - 0.2),
        ),
        2,
    )

    if speech_to_noise_ratio < 2.4 or room_variability_ratio >= 2.2:
        speech_start_chunks = 4
    elif speech_to_noise_ratio >= 5.5 and room_variability_ratio <= 1.4:
        speech_start_chunks = 2
    else:
        speech_start_chunks = 3

    preroll_duration_seconds = 0.4 if speech_start_chunks >= 4 else 0.3
    followup_cooldown_seconds = 0.8 if speech_to_noise_ratio < 2.2 else 0.6

    notes: list[str] = []
    if speech_to_noise_ratio < 2.0:
        notes.append("Speech is close to room noise; try moving the mic closer or lowering speaker volume.")
    if ambient_p95 > 0.03:
        notes.append("Ambient noise is fairly high; keep the dynamic threshold settings enabled.")
    if room_variability_ratio >= 2.5:
        notes.append("Background noise is bursty; use a longer ambient calibration window and more speech-start chunks.")
    if speech_p50 < 0.05:
        notes.append("Speech energy is low; consider raising microphone input gain a little.")
    if not notes:
        notes.append("Speech and room noise look well separated; the recommended settings should be a stable starting point.")

    return AudioTuningRecommendation(
        ambient_p50=round(ambient_p50, 4),
        ambient_p80=round(ambient_p80, 4),
        ambient_p95=round(ambient_p95, 4),
        speech_p25=round(speech_p25, 4),
        speech_p50=round(speech_p50, 4),
        speech_p90=round(speech_p90, 4),
        speech_to_noise_ratio=round(speech_to_noise_ratio, 2),
        room_variability_ratio=round(room_variability_ratio, 2),
        silence_threshold=silence_threshold,
        ambient_noise_seconds=ambient_noise_seconds,
        speech_start_threshold_multiplier=speech_start_threshold_multiplier,
        speech_end_threshold_multiplier=speech_end_threshold_multiplier,
        speech_start_chunks=speech_start_chunks,
        preroll_duration_seconds=preroll_duration_seconds,
        followup_cooldown_seconds=followup_cooldown_seconds,
        notes=tuple(notes),
    )


def format_recommended_env(
    recommendation: AudioTuningRecommendation,
) -> dict[str, str]:
    return {
        "SILENCE_THRESHOLD": f"{recommendation.silence_threshold:.4f}",
        "AMBIENT_NOISE_SECONDS": f"{recommendation.ambient_noise_seconds:.1f}",
        "SPEECH_START_THRESHOLD_MULTIPLIER": (
            f"{recommendation.speech_start_threshold_multiplier:.2f}"
        ),
        "SPEECH_END_THRESHOLD_MULTIPLIER": (
            f"{recommendation.speech_end_threshold_multiplier:.2f}"
        ),
        "SPEECH_START_CHUNKS": str(recommendation.speech_start_chunks),
        "PREROLL_DURATION_SECONDS": (
            f"{recommendation.preroll_duration_seconds:.1f}"
        ),
        "FOLLOWUP_COOLDOWN_SECONDS": (
            f"{recommendation.followup_cooldown_seconds:.1f}"
        ),
    }


def _coerce_levels(levels: Sequence[float]) -> np.ndarray:
    coerced = np.asarray([max(0.0, float(level)) for level in levels], dtype=np.float32)
    return coerced[np.isfinite(coerced)]


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile))


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
