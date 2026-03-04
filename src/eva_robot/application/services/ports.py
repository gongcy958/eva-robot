from typing import Protocol

from ...domain.intents import Intent


class AsrService(Protocol):
    def transcribe(self, audio: object) -> str:
        ...


class LlmService(Protocol):
    def generate(self, prompt: str, user_input: str) -> str:
        ...


class TtsService(Protocol):
    def speak(self, text: str) -> None:
        ...


class AudioInputService(Protocol):
    def record(self) -> object:
        ...


class IntentRoutingService(Protocol):
    def route(self, text: str) -> Intent:
        ...
