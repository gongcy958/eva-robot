import os
import platform

import numpy as np
import requests
import sounddevice as sd
from faster_whisper import WhisperModel

from intent_router import route_intent
from prompts import PROMPTS


WHISPER_MODEL_PATH = os.getenv(
    "WHISPER_MODEL_PATH", "/Users/mine/.cache/faster-whisper/small"
)
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
RECORD_SECONDS = int(os.getenv("RECORD_SECONDS", "3"))


model = WhisperModel(
    WHISPER_MODEL_PATH,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE,
)


def call_local_llm(prompt: str, user_input: str, retries: int = 3) -> str:
    for i in range(retries):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": f"{prompt}\nUser: {user_input}",
                    "stream": False,
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except Exception as exc:
            print(f"[LLM] call failed, retry {i + 1}/{retries}: {exc}")
    return "Sorry, I can't respond right now."


def speak_text(text: str) -> None:
    if platform.system() == "Darwin":
        safe_text = text.replace('"', "'")
        code = os.system(f'say "{safe_text}"')
        if code != 0:
            print("[TTS] macOS say failed, response printed only.")
    else:
        print("[TTS] Non-macOS detected, speech playback skipped.")


def process_once() -> None:
    print(f"\nRecording {RECORD_SECONDS} seconds. Speak clearly...")
    try:
        audio = sd.rec(
            int(RECORD_SECONDS * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
    except Exception as exc:
        print(f"[Audio] recording error: {exc}")
        return

    audio = audio.flatten()
    print("Raw audio shape:", audio.shape, "Max amplitude:", np.max(np.abs(audio)))

    try:
        segments, _info = model.transcribe(audio, language="en", vad_filter=False)
        text = " ".join(seg.text for seg in segments).strip()
    except Exception as exc:
        print(f"[ASR] whisper transcription error: {exc}")
        return

    print("Detected text:", repr(text))
    if not text:
        print("Didn't catch anything.")
        return

    intent = route_intent(text)
    prompt = PROMPTS.get(intent, "Greet the user politely.")
    response = call_local_llm(prompt, text)
    print(f"[{intent}] AI:", response)

    try:
        speak_text(response)
    except Exception as exc:
        print(f"[TTS] playback error: {exc}")


def main() -> None:
    print("=== Family English Robot Stable MVP ===")
    print("Speak English, AI will respond. Press Ctrl+C to quit.")

    while True:
        try:
            process_once()
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as exc:
            print("[Main] unexpected error:", exc)


if __name__ == "__main__":
    main()
