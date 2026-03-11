from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import re
import time

import numpy as np

from ..services.ports import (
    AsrTranscription,
    AsrService,
    AudioInputService,
    IntentRoutingService,
    LlmService,
    TtsService,
)
from ...domain.intents import Intent, PROMPTS
from ...shared.observability import StructuredLogger

LearningMode = Intent
MODE_DISPLAY_NAMES: dict[LearningMode, str] = {
    "translate_text": "翻译模式",
    "word_explain": "释义模式",
    "sentence_fix": "纠错模式",
    "grammar_question": "语法模式",
    "repeat_slowly": "跟读模式",
    "ask_in_english": "口语模式",
}


@dataclass(frozen=True)
class ConversationTurn:
    user: str
    assistant: str
    intent: Intent


FALLBACK_RESPONSES: dict[Intent, str] = {
    "small_talk": "Sorry, I had trouble replying just now. Could you say that again?",
    "translate_text": "Sorry, I couldn't translate that just now. Please say the sentence again.",
    "word_explain": "Sorry, I couldn't explain that word just now. Please repeat the word or phrase.",
    "sentence_fix": "Sorry, I couldn't fix that sentence just now. Please say the sentence again.",
    "grammar_question": "Sorry, I couldn't answer that grammar question just now. Please ask me once more.",
    "repeat_slowly": "Sorry, I couldn't repeat that clearly just now. Please say it once more.",
    "ask_in_english": "Sorry, I couldn't answer that right now. Please try asking again.",
}


class RunVoiceTurnUseCase:
    _SPOKEN_LABEL_MAP = {
        "translation": "Translation",
        "tip": "Tip",
        "meaning": "Meaning",
        "example": "Example",
        "corrected": "Corrected sentence",
        "why": "Why",
        "rule": "Rule",
        "slow version": "Slow version",
        "chunks": "Chunks",
    }

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
        asr_min_avg_logprob: float = -1.2,
        asr_max_no_speech_prob: float = 0.7,
        asr_low_confidence_message: str = "Sorry, I didn't catch that clearly. Please say it again.",
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
        self._asr_min_avg_logprob = asr_min_avg_logprob
        self._asr_max_no_speech_prob = asr_max_no_speech_prob
        self._asr_low_confidence_message = asr_low_confidence_message
        self._active_learning_mode: LearningMode | None = None
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

    def _remember_turn(self, user_text: str, assistant_text: str, intent: Intent) -> None:
        if self._conversation_memory.maxlen == 0:
            return
        self._conversation_memory.append(
            ConversationTurn(user=user_text, assistant=assistant_text, intent=intent)
        )

    def _resolve_intent(self, text: str) -> Intent:
        intent = self._router.route(text)
        if not self._conversation_memory:
            if intent == "small_talk" and self._active_learning_mode is not None:
                return self._active_learning_mode
            return intent

        normalized = text.lower().strip()
        normalized = re.sub(r"[!?.,，。？！;；:：]+$", "", normalized)
        last_intent = self._conversation_memory[-1].intent

        if normalized in {
            "again",
            "say again",
            "repeat",
            "one more time",
            "speak slowly",
            "slowly please",
            "again please",
            "慢一点",
            "再说一遍",
        }:
            return "repeat_slowly"

        if normalized in {
            "another example",
            "one more example",
            "more examples",
            "give me an example",
            "再来一个例句",
            "举个例子",
        } and last_intent in {
            "translate_text",
            "word_explain",
            "sentence_fix",
            "grammar_question",
            "ask_in_english",
        }:
            return last_intent

        if self._is_reason_followup_request(text):
            return last_intent

        if self._is_meaning_followup_request(text):
            return "word_explain"

        if intent == "small_talk" and self._active_learning_mode is not None:
            return self._active_learning_mode

        return intent

    def _parse_learning_mode_command(self, text: str) -> tuple[str, LearningMode | None] | None:
        normalized = self._normalized_followup_text(text)

        if normalized in {
            "exit mode",
            "quit mode",
            "normal mode",
            "turn off mode",
            "退出模式",
            "关闭模式",
            "普通模式",
            "自由模式",
        }:
            return ("exit", None)

        mode_patterns: list[tuple[LearningMode, tuple[str, ...]]] = [
            (
                "translate_text",
                (
                    "translation mode",
                    "translate mode",
                    "进入翻译模式",
                    "切换到翻译模式",
                    "翻译模式",
                ),
            ),
            (
                "sentence_fix",
                (
                    "correction mode",
                    "fix mode",
                    "sentence fix mode",
                    "进入纠错模式",
                    "切换到纠错模式",
                    "纠错模式",
                    "改错模式",
                ),
            ),
            (
                "word_explain",
                (
                    "explain mode",
                    "word explain mode",
                    "meaning mode",
                    "进入释义模式",
                    "切换到释义模式",
                    "释义模式",
                    "单词解释模式",
                ),
            ),
            (
                "grammar_question",
                (
                    "grammar mode",
                    "进入语法模式",
                    "切换到语法模式",
                    "语法模式",
                ),
            ),
            (
                "repeat_slowly",
                (
                    "repeat mode",
                    "shadowing mode",
                    "slow speaking mode",
                    "进入跟读模式",
                    "切换到跟读模式",
                    "跟读模式",
                ),
            ),
            (
                "ask_in_english",
                (
                    "speaking mode",
                    "english chat mode",
                    "conversation mode",
                    "进入口语模式",
                    "切换到口语模式",
                    "口语模式",
                    "英语对话模式",
                ),
            ),
        ]

        for mode, patterns in mode_patterns:
            if normalized in patterns:
                return ("enter", mode)

        return None

    def _handle_learning_mode_command(self, text: str) -> bool:
        command = self._parse_learning_mode_command(text)
        if command is None:
            return False

        action, mode = command
        if action == "exit":
            previous_mode = self._active_learning_mode
            self._active_learning_mode = None
            message = "好的，已退出学习模式。"
            if previous_mode is None:
                message = "当前没有开启学习模式。"
            print("[Mode]", message)
            self._logger.info(
                "learning_mode.changed",
                action=action,
                previous_mode=previous_mode,
                active_mode=self._active_learning_mode,
            )
            self.speak_feedback(message)
            return True

        previous_mode = self._active_learning_mode
        self._active_learning_mode = mode
        mode_name = MODE_DISPLAY_NAMES.get(mode or "small_talk", "学习模式")
        message = f"好的，已进入{mode_name}。"
        print("[Mode]", message)
        self._logger.info(
            "learning_mode.changed",
            action=action,
            previous_mode=previous_mode,
            active_mode=self._active_learning_mode,
        )
        self.speak_feedback(message)
        return True

    @staticmethod
    def _normalized_followup_text(text: str) -> str:
        normalized = text.lower().strip()
        return re.sub(r"[!?.,，。？！;；:：]+$", "", normalized)

    def _is_generic_repeat_request(self, text: str) -> bool:
        return self._normalized_followup_text(text) in {
            "again",
            "say again",
            "repeat",
            "one more time",
            "speak slowly",
            "slowly please",
            "again please",
            "慢一点",
            "再说一遍",
        }

    def _is_example_followup_request(self, text: str) -> bool:
        return self._normalized_followup_text(text) in {
            "another example",
            "one more example",
            "more examples",
            "give me an example",
            "再来一个例句",
            "举个例子",
        }

    def _is_reason_followup_request(self, text: str) -> bool:
        return self._normalized_followup_text(text) in {
            "why",
            "why is that",
            "why so",
            "为什么",
            "为什么呢",
        }

    def _is_meaning_followup_request(self, text: str) -> bool:
        return self._normalized_followup_text(text) in {
            "what do you mean",
            "what does that mean",
            "mean",
            "什么意思",
            "什么意思呢",
        }

    def _build_effective_user_input(self, intent: Intent, text: str) -> str:
        if not self._conversation_memory:
            return text

        if intent == "repeat_slowly" and self._is_generic_repeat_request(text):
            last_turn = self._conversation_memory[-1]
            return (
                "Repeat your last reply more slowly and clearly. "
                "Keep the meaning simple and easy to follow.\n"
                f"Last reply:\n{last_turn.assistant}"
            )

        if self._is_example_followup_request(text):
            last_turn = self._conversation_memory[-1]
            return (
                "Give one more simple learning example based on the previous exchange.\n"
                f"Previous user request:\n{last_turn.user}\n"
                f"Previous assistant reply:\n{last_turn.assistant}"
            )

        if self._is_reason_followup_request(text):
            last_turn = self._conversation_memory[-1]
            return (
                "Explain the reason behind your previous reply in simple teaching language.\n"
                f"Previous user request:\n{last_turn.user}\n"
                f"Previous assistant reply:\n{last_turn.assistant}"
            )

        if self._is_meaning_followup_request(text):
            last_turn = self._conversation_memory[-1]
            return (
                "Explain the meaning of your previous reply simply. "
                "If useful, include a short Chinese explanation.\n"
                f"Previous reply:\n{last_turn.assistant}"
            )

        return text

    def _build_spoken_response(self, response: str) -> str:
        lines = [line.strip(" -•\t") for line in response.splitlines() if line.strip()]
        if not lines:
            return response.strip()

        spoken_parts: list[str] = []
        for line in lines:
            if ":" in line:
                label, value = line.split(":", 1)
                normalized_label = label.strip().lower()
                spoken_label = self._SPOKEN_LABEL_MAP.get(normalized_label, label.strip())
                spoken_value = value.strip()
                if spoken_value:
                    spoken_parts.append(f"{spoken_label}. {spoken_value}")
                else:
                    spoken_parts.append(spoken_label)
            else:
                spoken_parts.append(line)

        spoken = " ".join(spoken_parts)
        spoken = re.sub(r"\s+", " ", spoken).strip()
        return spoken

    def speak_feedback(self, text: str) -> None:
        feedback = text.strip()
        if not feedback:
            return

        try:
            self._tts.speak(feedback)
            self._logger.info(
                "tts.feedback_completed",
                text_length=len(feedback),
            )
        except Exception as exc:
            self._logger.error(
                "tts.feedback_failed",
                error=str(exc),
            )

    def _transcribe(self, audio: np.ndarray) -> AsrTranscription | None:
        last_error: Exception | None = None
        for attempt in range(1, self._asr_retries + 1):
            started_at = time.perf_counter()
            try:
                if hasattr(self._asr, "transcribe_with_details"):
                    result = self._asr.transcribe_with_details(audio)
                else:
                    result = AsrTranscription(text=self._asr.transcribe(audio))
                duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
                self._logger.info(
                    "asr.transcribe_completed",
                    attempt=attempt,
                    duration_ms=duration_ms,
                    transcript=result.text,
                    avg_logprob=result.avg_logprob,
                    no_speech_prob=result.no_speech_prob,
                    language=result.language,
                    language_probability=result.language_probability,
                )
                return result
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

    def _is_low_confidence_transcription(self, transcription: AsrTranscription) -> bool:
        if not transcription.text.strip():
            return False

        avg_logprob = transcription.avg_logprob
        no_speech_prob = transcription.no_speech_prob

        if avg_logprob is not None and avg_logprob < self._asr_min_avg_logprob:
            return True

        if (
            no_speech_prob is not None
            and no_speech_prob > self._asr_max_no_speech_prob
        ):
            return True

        return False

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

        transcription = self._transcribe(audio_array)
        if transcription is None:
            return None

        if self._is_low_confidence_transcription(transcription):
            self._logger.warning(
                "asr.low_confidence_detected",
                transcript=transcription.text,
                avg_logprob=transcription.avg_logprob,
                no_speech_prob=transcription.no_speech_prob,
            )
            print("[ASR] low confidence detected, asking user to repeat.")
            self.speak_feedback(self._asr_low_confidence_message)
            return None

        text = transcription.text

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

        if self._handle_learning_mode_command(text):
            return

        intent = self._resolve_intent(text)
        self._logger.info(
            "intent.routed",
            intent=intent,
            user_text=text,
            memory_turns=len(self._conversation_memory),
            active_learning_mode=self._active_learning_mode,
        )

        base_prompt = PROMPTS.get(intent, "Greet the user politely.")
        prompt = self._build_prompt(base_prompt)
        effective_user_input = self._build_effective_user_input(intent, text)
        llm_started_at = time.perf_counter()
        response = self._llm.generate(prompt, effective_user_input).strip()
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
            effective_user_input=effective_user_input,
            response_length=len(response),
        )
        print(f"[{intent}] AI:", response)
        self._remember_turn(text, response, intent)

        spoken_response = self._build_spoken_response(response)
        tts_started_at = time.perf_counter()
        try:
            self._tts.speak(spoken_response)
            tts_duration_ms = round((time.perf_counter() - tts_started_at) * 1000, 2)
            self._logger.info(
                "tts.speak_completed",
                duration_ms=tts_duration_ms,
                text_length=len(spoken_response),
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
