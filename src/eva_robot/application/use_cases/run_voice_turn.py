from __future__ import annotations

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

    def run_once(self) -> None:
        print(f"\nRecording {self._record_seconds} seconds. Speak clearly...")
        try:
            audio = self._recorder.record()
        except Exception as exc:
            print(f"[Audio] recording error: {exc}")
            return

        print("Raw audio shape:", getattr(audio, "shape", "n/a"), "Max amplitude:", np.max(np.abs(audio)))

        try:
            text = self._asr.transcribe(audio)
        except Exception as exc:
            print(f"[ASR] whisper transcription error: {exc}")
            return

        print("Detected text:", repr(text))
        if not text:
            print("Didn't catch anything.")
            return

        intent = self._router.route(text)
        prompt = PROMPTS.get(intent, "Greet the user politely.")
        response = self._llm.generate(prompt, text)
        print(f"[{intent}] AI:", response)

        try:
            self._tts.speak(response)
        except Exception as exc:
            print(f"[TTS] playback error: {exc}")
