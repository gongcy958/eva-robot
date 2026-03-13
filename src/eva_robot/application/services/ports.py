from dataclasses import dataclass
from typing import Protocol

from ...domain.intents import Intent


@dataclass(frozen=True)
class AsrTranscription:
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    language: str | None = None
    language_probability: float | None = None
    segment_count: int = 0


class AsrService(Protocol):
    def transcribe(self, audio: object) -> str:
        ...

    def transcribe_with_details(self, audio: object) -> AsrTranscription:
        ...


class LlmService(Protocol):
    def generate(self, prompt: str, user_input: str) -> str:
        ...


class TtsService(Protocol):
    def speak(self, text: str) -> None:
        ...


class AudioInputService(Protocol):
    def record(
        self,
        wait_timeout_seconds: float | None = None,
        max_record_seconds: float | None = None,
    ) -> object:
        ...


class IntentRoutingService(Protocol):
    def route(self, text: str) -> Intent:
        ...
