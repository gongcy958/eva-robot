from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time

import numpy as np

from ..services.ports import (
    AsrService,
    AudioInputService,
    IntentRoutingService,
    LlmService,
    TtsService,
)
from ...domain.intents import Intent, PROMPTS
from ...shared.observability import StructuredLogger


@dataclass(frozen=True)
class ConversationTurn:
    user: str
    assistant: str


FALLBACK_RESPONSES: dict[Intent, str] = {
    "small_talk": "Sorry, I had trouble replying just now. Could you say that again?",
    "word_explain": "Sorry, I couldn't explain that word just now. Please repeat the word or phrase.",
    "sentence_fix": "Sorry, I couldn't fix that sentence just now. Please say the sentence again.",
    "grammar_question": "Sorry, I couldn't answer that grammar question just now. Please ask me once more.",
    "ask_in_english": "Sorry, I couldn't answer that right now. Please try asking again.",
}


class RunVoiceTurnUseCase:
    def __init__(
        self,
        recorder: AudioInputService,
        asr: AsrService,
        router: IntentRoutingService,
        llm: LlmService,
        tts: TtsService,
        record_seconds: int,
        conversation_memory_turns: int = 3,
        asr_retries: int = 2,
        logger: StructuredLogger | None = None,
    ) -> None:
        self._recorder = recorder
        self._asr = asr
        self._router = router
        self._llm = llm
        self._tts = tts
        self._record_seconds = record_seconds
        self._conversation_memory = deque(
            maxlen=max(0, conversation_memory_turns)
        )
        self._asr_retries = max(1, asr_retries)
        self._logger = logger or StructuredLogger()

    def _build_prompt(self, base_prompt: str) -> str:
        prompt_parts = [base_prompt.strip()]
        if not self._conversation_memory:
            return "\n\n".join(prompt_parts)

        prompt_parts.append(
            "Recent conversation context. Use it only if it helps answer naturally."
        )
        for turn in self._conversation_memory:
            prompt_parts.append(f"User: {turn.user}")
            prompt_parts.append(f"Assistant: {turn.assistant}")
        return "\n".join(prompt_parts)

    def _fallback_response(self, intent: Intent) -> str:
        return FALLBACK_RESPONSES.get(
            intent,
            "Sorry, I had trouble responding just now. Please try again.",
        )

    def _remember_turn(self, user_text: str, assistant_text: str) -> None:
        if self._conversation_memory.maxlen == 0:
            return
        self._conversation_memory.append(
            ConversationTurn(user=user_text, assistant=assistant_text)
        )

    def _transcribe(self, audio: np.ndarray) -> str | None:
        last_error: Exception | None = None
        for attempt in range(1, self._asr_retries + 1):
            started_at = time.perf_counter()
            try:
                text = self._asr.transcribe(audio)
                duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
                self._logger.info(
                    "asr.transcribe_completed",
                    attempt=attempt,
                    duration_ms=duration_ms,
                    transcript=text,
                )
                return text
            except Exception as exc:
                duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
                last_error = exc
                self._logger.warning(
                    "asr.transcribe_failed",
                    attempt=attempt,
                    retries=self._asr_retries,
                    duration_ms=duration_ms,
                    error=str(exc),
                )

        if last_error is not None:
            print(
                f"[ASR] whisper transcription error after "
                f"{self._asr_retries} attempts: {last_error}"
            )
        return None

    def listen_once(self) -> str | None:
        print(
            f"\nListening... (max {self._record_seconds}s, "
            "auto-stop after you stop speaking)"
        )
        started_at = time.perf_counter()
        try:
            audio = self._recorder.record()
        except Exception as exc:
            print(f"[Audio] recording error: {exc}")
            self._logger.error("audio.record_failed", error=str(exc))
            time.sleep(1)
            return None

        if audio is None:
            print("[Audio] no audio captured.")
            self._logger.info("audio.record_empty")
            return None

        try:
            audio_array = np.asarray(audio, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            print(f"[Audio] unsupported recording format: {exc}")
            self._logger.error("audio.record_invalid_format", error=str(exc))
            return None

        if audio_array.size == 0:
            print("[Audio] no audio captured.")
            self._logger.info("audio.record_empty")
            return None

        record_duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        max_amplitude = float(np.max(np.abs(audio_array)))
        print(
            "Raw audio shape:",
            getattr(audio_array, "shape", "n/a"),
            "Max amplitude:",
            max_amplitude,
        )
        self._logger.info(
            "audio.record_completed",
            duration_ms=record_duration_ms,
            sample_count=int(audio_array.size),
            max_amplitude=round(max_amplitude, 6),
        )

        text = self._transcribe(audio_array)
        if text is None:
            return None

        print("Detected text:", repr(text))
        if not text:
            print("Didn't catch anything.")
            self._logger.info("asr.transcribe_empty")
            return None

        return text

    def handle_text(self, text: str) -> None:
        text = text.strip()
        if not text:
            return

        intent = self._router.route(text)
        self._logger.info(
            "intent.routed",
            intent=intent,
            user_text=text,
            memory_turns=len(self._conversation_memory),
        )

        base_prompt = PROMPTS.get(intent, "Greet the user politely.")
        prompt = self._build_prompt(base_prompt)
        llm_started_at = time.perf_counter()
        response = self._llm.generate(prompt, text).strip()
        llm_duration_ms = round((time.perf_counter() - llm_started_at) * 1000, 2)
        used_fallback = False
        if not response:
            response = self._fallback_response(intent)
            used_fallback = True

        self._logger.info(
            "llm.generate_completed",
            intent=intent,
            duration_ms=llm_duration_ms,
            used_fallback=used_fallback,
            response_length=len(response),
        )
        print(f"[{intent}] AI:", response)
        self._remember_turn(text, response)

        tts_started_at = time.perf_counter()
        try:
            self._tts.speak(response)
            tts_duration_ms = round((time.perf_counter() - tts_started_at) * 1000, 2)
            self._logger.info(
                "tts.speak_completed",
                duration_ms=tts_duration_ms,
                text_length=len(response),
            )
        except Exception as exc:
            print(f"[TTS] playback error: {exc}")
            tts_duration_ms = round((time.perf_counter() - tts_started_at) * 1000, 2)
            self._logger.error(
                "tts.speak_failed",
                duration_ms=tts_duration_ms,
                error=str(exc),
            )

    def run_once(self) -> None:
        text = self.listen_once()
        if not text:
            return
        self.handle_text(text)
