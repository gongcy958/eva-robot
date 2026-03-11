#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


VALID_PROFILES = {"default", "high_quality"}
TARGET_KEY = "LLM_PROFILE"
MODEL_KEY = "LLM_MODEL"


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in VALID_PROFILES:
        print("Usage: python3 scripts/set_llm_profile.py [default|high_quality]")
        return 1

    profile = sys.argv[1]
    env_path = Path(__file__).resolve().parents[1] / ".env.local"
    if not env_path.exists():
        print(f"Missing {env_path}. Create it from .env.example first.")
        return 1

    lines = env_path.read_text(encoding="utf-8").splitlines()
    updated: list[str] = []
    profile_updated = False
    model_updated = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{TARGET_KEY}=") or stripped.startswith(f"export {TARGET_KEY}="):
            updated.append(f"{TARGET_KEY}={profile}")
            profile_updated = True
            continue
        if stripped.startswith(f"{MODEL_KEY}=") or stripped.startswith(f"export {MODEL_KEY}="):
            updated.append(f"{MODEL_KEY}=")
            model_updated = True
            continue
        updated.append(line)

    if not profile_updated:
        updated.append(f"{TARGET_KEY}={profile}")
    if not model_updated:
        updated.append(f"{MODEL_KEY}=")

    env_path.write_text("\n".join(updated).rstrip() + "\n", encoding="utf-8")
    model = "gpt-5.2" if profile == "high_quality" else "gpt-5.1"
    print(f"Set {TARGET_KEY}={profile} ({model}) in {env_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
