#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eva_robot.application.use_cases.run_voice_turn import RunVoiceTurnUseCase
from src.eva_robot.domain.intents import IntentRouter
from src.eva_robot.infrastructure.llm.failover_client import FailoverLlmClient
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


def assert_equal(actual, expected, message: str) -> None:
    if actual != expected:
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


def main() -> int:
    test_intent_router()
    test_stateful_learning_mode()
    test_stateful_family_scene()
    test_follow_up_rewrite()
    test_llm_failover_switches_to_ollama()
    test_preflight_returns_ollama_when_remote_fails()
    print("smoke_regression: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
