#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eva_robot.application.use_cases.run_voice_turn import RunVoiceTurnUseCase
from src.eva_robot.domain.intents import IntentRouter
from src.eva_robot.infrastructure.llm.failover_client import FailoverLlmClient
from src.eva_robot.interfaces.voice import microphone as microphone_module
from src.eva_robot.interfaces.voice.microphone import MicrophoneRecorder
from src.eva_robot.interfaces.voice.runtime import VoiceRuntime
from src.eva_robot.shared.config import AppConfig
from src.eva_robot.shared.preflight import PreflightFinding, StartupPreflight


class DummyRecorder:
    def record(self):
        return None


class DummyAsr:
    def transcribe(self, audio):
        return ""


@dataclass
class DummyTts:
    spoken: list[str]

    def speak(self, text: str) -> None:
        self.spoken.append(text)


@dataclass
class CapturingLlm:
    calls: list[tuple[str, str]]

    def generate(self, prompt: str, user_input: str) -> str:
        self.calls.append((prompt, user_input))
        return "Translation: hello\nTip: keep it simple"


@dataclass
class SequencedLlm:
    responses: list[str]
    calls: list[tuple[str, str]]

    def generate(self, prompt: str, user_input: str) -> str:
        self.calls.append((prompt, user_input))
        if not self.responses:
            return ""
        return self.responses.pop(0)


@dataclass
class ScriptedVoiceTurn:
    heard: list[str]
    spoken: list[str]
    handled: list[str]

    def listen_once(self) -> str | None:
        if not self.heard:
            raise KeyboardInterrupt
        return self.heard.pop(0)

    def speak_feedback(self, text: str) -> None:
        self.spoken.append(text)

    def handle_text(self, text: str) -> None:
        self.handled.append(text)


class FakeInputStream:
    def __init__(self, chunks: list[np.ndarray], **_kwargs) -> None:
        self._chunks = [chunk.astype(np.float32) for chunk in chunks]
        self._index = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self, frames_per_chunk: int) -> tuple[np.ndarray, bool]:
        if self._index >= len(self._chunks):
            data = np.zeros((frames_per_chunk, 1), dtype=np.float32)
            return data, False

        chunk = self._chunks[self._index]
        self._index += 1
        if chunk.size != frames_per_chunk:
            raise AssertionError(
                f"expected chunk of size {frames_per_chunk}, got {chunk.size}"
            )
        return chunk.reshape(frames_per_chunk, 1), False


def assert_equal(actual, expected, message: str) -> None:
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def assert_allclose(actual, expected, message: str, atol: float = 1e-6) -> None:
    if not np.allclose(actual, expected, atol=atol):
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def build_use_case() -> tuple[RunVoiceTurnUseCase, CapturingLlm, DummyTts]:
    llm = CapturingLlm(calls=[])
    tts = DummyTts(spoken=[])
    use_case = RunVoiceTurnUseCase(
        recorder=DummyRecorder(),
        asr=DummyAsr(),
        router=IntentRouter(),
        llm=llm,
        tts=tts,
        record_seconds=3,
    )
    return use_case, llm, tts


def test_intent_router() -> None:
    router = IntentRouter()
    assert_equal(
        router.route("Translate this into English: 我很开心"),
        "translate_text",
        "translate intent",
    )
    assert_equal(
        router.route("慢一点再说一遍"),
        "repeat_slowly",
        "repeat intent",
    )
    assert_equal(
        router.route("What does vivid mean?"),
        "word_explain",
        "word explain intent",
    )


def test_stateful_learning_mode() -> None:
    use_case, llm, tts = build_use_case()
    use_case.handle_text("进入翻译模式")
    use_case.handle_text("我今天很开心")

    assert_equal(len(llm.calls), 1, "llm calls after translation mode")
    assert_equal(llm.calls[0][1], "我今天很开心", "mode forwards raw user text")
    assert_equal(
        tts.spoken[0],
        "好的，已进入翻译模式。",
        "tts wake for translation mode",
    )


def test_stateful_family_scene() -> None:
    use_case, llm, tts = build_use_case()
    use_case.handle_text("进入早餐英语场景")
    use_case.handle_text("What should I say before school?")
    use_case.handle_text("当前是什么模式")

    assert_equal(tts.spoken[0], "好的，已进入早餐英语场景。", "tts scene feedback")
    assert "Family scene: breakfast time." in llm.calls[0][0], "scene prompt should be injected"
    assert_equal(
        tts.spoken[-1],
        "当前是未开启学习模式，早餐英语场景。",
        "status query should report current scene",
    )


def test_follow_up_rewrite() -> None:
    use_case, llm, _tts = build_use_case()
    use_case.handle_text("What does vivid mean?")
    use_case.handle_text("another example")
    use_case.handle_text("again")
    use_case.handle_text("why")

    assert_equal(len(llm.calls), 4, "llm calls for follow-up chain")
    assert "Previous user request" in llm.calls[1][1], "example follow-up should use previous turn"
    assert "Repeat your last reply more slowly" in llm.calls[2][1], "repeat follow-up should reuse prior answer"
    assert "Explain the reason behind your previous reply" in llm.calls[3][1], "why follow-up should explain prior answer"


def test_llm_failover_switches_to_ollama() -> None:
    primary = SequencedLlm(responses=[""], calls=[])
    fallback = SequencedLlm(responses=["from ollama", "still ollama"], calls=[])
    client = FailoverLlmClient(
        primary=primary,
        fallback=fallback,
        primary_provider="openai_compatible",
        fallback_provider="ollama",
    )

    assert_equal(client.generate("prompt 1", "hello"), "from ollama", "first call should fall back")
    assert_equal(client.generate("prompt 2", "world"), "still ollama", "later calls should stay on ollama")
    assert_equal(len(primary.calls), 1, "primary should stop after failover")
    assert_equal(len(fallback.calls), 2, "fallback should serve both calls")


def test_preflight_returns_ollama_when_remote_fails() -> None:
    config = AppConfig(
        whisper_model_path="small",
        llm_provider="openai_compatible",
        llm_api_key="dummy",
        ollama_model="qwen2.5:7b-instruct",
        skip_startup_checks=False,
    )
    preflight = StartupPreflight(config)
    preflight._check_whisper_model = lambda: [
        PreflightFinding("info", "whisper.path.named_model", "ok")
    ]
    preflight._check_openai_compatible = lambda: [
        PreflightFinding("error", "llm.probe.failed", "remote failed")
    ]
    preflight._check_ollama = lambda: [
        PreflightFinding("info", "ollama.tags.ok", "ollama ok"),
        PreflightFinding("info", "ollama.model.ok", "model ok"),
    ]

    result = preflight.run()
    assert_equal(result.effective_llm_provider, "ollama", "preflight provider fallback")
    assert_equal(result.used_fallback, True, "preflight fallback flag")


def test_runtime_wake_then_follow_up() -> None:
    turn = ScriptedVoiceTurn(
        heard=["hello", "how are you"],
        spoken=[],
        handled=[],
    )
    runtime = VoiceRuntime(
        run_voice_turn=turn,
        wake_word="hello",
        wake_ack_message="I'm here.",
        sleep_command="goodbye",
        sleep_ack_message="Going idle.",
        wake_timeout_seconds=60,
    )

    runtime.run()

    assert_equal(turn.spoken[0], "I'm here.", "wake ack should be spoken")
    assert_equal(turn.handled, ["how are you"], "follow-up after wake word should be handled")


def test_runtime_inline_wake_command() -> None:
    turn = ScriptedVoiceTurn(
        heard=["hello translate this"],
        spoken=[],
        handled=[],
    )
    runtime = VoiceRuntime(
        run_voice_turn=turn,
        wake_word="hello",
        wake_ack_message="I'm here.",
        sleep_command="goodbye",
        sleep_ack_message="Going idle.",
        wake_timeout_seconds=60,
    )

    runtime.run()

    assert_equal(turn.spoken[0], "I'm here.", "inline wake should still speak ack")
    assert_equal(
        turn.handled,
        ["translate this"],
        "inline wake command should strip wake word and handle the command",
    )


def test_microphone_waits_for_real_speech() -> None:
    chunks = [
        np.array([0.0], dtype=np.float32),
        np.array([0.6], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([0.6], dtype=np.float32),
        np.array([0.6], dtype=np.float32),
        np.array([0.6], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
    ]
    original_input_stream = microphone_module.sd.InputStream
    microphone_module.sd.InputStream = lambda **kwargs: FakeInputStream(chunks, **kwargs)
    try:
        recorder = MicrophoneRecorder(
            sample_rate=10,
            record_seconds=3,
            min_record_seconds=0.1,
            max_record_seconds=1.0,
            silence_duration_seconds=0.2,
            silence_threshold=0.5,
            no_speech_timeout_seconds=1.0,
            speech_start_chunks=3,
            preroll_duration_seconds=0.3,
        )
        recorded = recorder.record()
    finally:
        microphone_module.sd.InputStream = original_input_stream

    assert_equal(recorded.size, 5, "recorder should ignore single noise spikes before speech")
    assert_allclose(
        recorded.tolist(),
        [0.6, 0.6, 0.6, 0.0, 0.0],
        "recorder should start near actual speech and keep trailing silence",
    )


def main() -> int:
    test_intent_router()
    test_stateful_learning_mode()
    test_stateful_family_scene()
    test_follow_up_rewrite()
    test_llm_failover_switches_to_ollama()
    test_preflight_returns_ollama_when_remote_fails()
    test_runtime_wake_then_follow_up()
    test_runtime_inline_wake_command()
    test_microphone_waits_for_real_speech()
    print("smoke_regression: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
