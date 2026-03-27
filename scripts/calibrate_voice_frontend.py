#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import sounddevice as sd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eva_robot.interfaces.voice.microphone import MicrophoneRecorder
from src.eva_robot.shared.audio_tuning import (
    AudioTuningRecommendation,
    format_recommended_env,
    recommend_voice_frontend_config,
)


PROMPT_TEXT = (
    "Eva, please translate this sentence into English. "
    "Then explain the key phrase one more time."
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture room noise and speaking volume, then recommend Eva voice settings."
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-seconds", type=float, default=0.1)
    parser.add_argument("--ambient-seconds", type=float, default=3.0)
    parser.add_argument("--speech-seconds", type=float, default=6.0)
    parser.add_argument(
        "--env-file",
        default=".env.local",
        help="Env file to update when --write-env is used.",
    )
    parser.add_argument(
        "--write-env",
        action="store_true",
        help="Write the recommended voice settings into the env file.",
    )
    args = parser.parse_args()

    print("=== Eva voice frontend calibration ===")
    print("Step 1/2: stay quiet so I can measure ambient noise.")
    ambient_levels = capture_levels(
        duration_seconds=args.ambient_seconds,
        sample_rate=args.sample_rate,
        chunk_seconds=args.chunk_seconds,
    )

    print("\nStep 2/2: read the sample sentence in your normal voice.")
    print(f"Sample: {PROMPT_TEXT}")
    speech_levels = capture_levels(
        duration_seconds=args.speech_seconds,
        sample_rate=args.sample_rate,
        chunk_seconds=args.chunk_seconds,
    )

    recommendation = recommend_voice_frontend_config(
        ambient_levels=ambient_levels,
        speech_levels=speech_levels,
    )

    print_recommendation(recommendation)
    env_values = format_recommended_env(recommendation)
    print("\nRecommended env overrides:")
    for key, value in env_values.items():
        print(f"{key}={value}")

    if args.write_env:
        env_path = PROJECT_ROOT / args.env_file
        update_env_file(env_path, env_values)
        print(f"\nUpdated {env_path}")
    else:
        print("\nTip: rerun with --write-env to persist these values into .env.local.")

    return 0


def capture_levels(
    *,
    duration_seconds: float,
    sample_rate: int,
    chunk_seconds: float,
) -> list[float]:
    frames_per_chunk = max(1, int(sample_rate * chunk_seconds))
    total_chunks = max(1, int(duration_seconds / chunk_seconds))
    levels: list[float] = []

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=frames_per_chunk,
    ) as stream:
        for index in range(total_chunks):
            data, _overflow = stream.read(frames_per_chunk)
            chunk = data[:, 0].copy()
            levels.append(MicrophoneRecorder._chunk_level(chunk))
            if (index + 1) % max(1, total_chunks // 6) == 0 or index + 1 == total_chunks:
                progress = round(((index + 1) / total_chunks) * 100)
                print(f"  ... {progress}%")

    return levels


def print_recommendation(recommendation: AudioTuningRecommendation) -> None:
    print("\nMetrics:")
    print(
        "  ambient p50/p80/p95 = "
        f"{recommendation.ambient_p50:.4f} / "
        f"{recommendation.ambient_p80:.4f} / "
        f"{recommendation.ambient_p95:.4f}"
    )
    print(
        "  speech  p25/p50/p90 = "
        f"{recommendation.speech_p25:.4f} / "
        f"{recommendation.speech_p50:.4f} / "
        f"{recommendation.speech_p90:.4f}"
    )
    print(
        "  speech/noise ratio = "
        f"{recommendation.speech_to_noise_ratio:.2f}"
    )
    print(
        "  room variability    = "
        f"{recommendation.room_variability_ratio:.2f}"
    )

    print("\nNotes:")
    for note in recommendation.notes:
        print(f"  - {note}")


def update_env_file(path: Path, values: dict[str, str]) -> None:
    lines: list[str] = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()

    updated: list[str] = []
    remaining = dict(values)
    for line in lines:
        stripped = line.strip()
        replaced = False
        for key, value in values.items():
            if stripped.startswith(f"{key}=") or stripped.startswith(f"export {key}="):
                updated.append(f"{key}={value}")
                remaining.pop(key, None)
                replaced = True
                break
        if not replaced:
            updated.append(line)

    for key, value in remaining.items():
        updated.append(f"{key}={value}")

    path.write_text("\n".join(updated).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    time.sleep(0.2)
    raise SystemExit(main())
