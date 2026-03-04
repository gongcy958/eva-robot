from __future__ import annotations

import time

import numpy as np

from ..services.ports import (
    AsrService,
    AudioInputService,
    IntentRoutingService,
    LlmService,
    TtsService,
)
from ...domain.intents import PROMPTS


class RunVoiceTurnUseCase:
    def __init__(
        self,
        recorder: AudioInputService,
        asr: AsrService,
        router: IntentRoutingService,
        llm: LlmService,
        tts: TtsService,
        record_seconds: int,
    ) -> None:
        self._recorder = recorder
        self._asr = asr
        self._router = router
        self._llm = llm
        self._tts = tts
        self._record_seconds = record_seconds

    def listen_once(self) -> str | None:
        print(
            f"\nListening... (max {self._record_seconds}s, "
            "auto-stop after you stop speaking)"
        )
        try:
            audio = self._recorder.record()
        except Exception as exc:
            print(f"[Audio] recording error: {exc}")
            time.sleep(1)
            return None

        print("Raw audio shape:", getattr(audio, "shape", "n/a"), "Max amplitude:", np.max(np.abs(audio)))

        try:
            text = self._asr.transcribe(audio)
        except Exception as exc:
            print(f"[ASR] whisper transcription error: {exc}")
            return None

        print("Detected text:", repr(text))
        if not text:
            print("Didn't catch anything.")
            return None

        return text

    def handle_text(self, text: str) -> None:
        text = text.strip()
        if not text:
            return

        intent = self._router.route(text)
        prompt = PROMPTS.get(intent, "Greet the user politely.")
        response = self._llm.generate(prompt, text)
        print(f"[{intent}] AI:", response)

        try:
            self._tts.speak(response)
        except Exception as exc:
            print(f"[TTS] playback error: {exc}")

    def run_once(self) -> None:
        text = self.listen_once()
        if not text:
            return
        self.handle_text(text)
