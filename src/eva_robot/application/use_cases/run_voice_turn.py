from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from difflib import SequenceMatcher
import re
import time
from typing import Literal

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
FamilyScene = Literal["breakfast", "family_time", "bedtime"]
SCENE_DISPLAY_NAMES: dict[FamilyScene, str] = {
    "breakfast": "早餐英语场景",
    "family_time": "亲子互动英语场景",
    "bedtime": "睡前英语场景",
}
SCENE_PROMPTS: dict[FamilyScene, str] = {
    "breakfast": (
        "Family scene: breakfast time. Prefer short, warm, practical English "
        "for morning routines, greetings, food, school preparation, and simple encouragement."
    ),
    "family_time": (
        "Family scene: parent-child interaction. Prefer playful, supportive, "
        "easy English suitable for family conversation, games, chores, and encouragement."
    ),
    "bedtime": (
        "Family scene: bedtime. Prefer calm, gentle, slower English for bedtime "
        "routines, comfort, reflection, and simple goodnight conversation."
    ),
}
STATUS_QUERY_PATTERNS = {
    "当前是什么模式",
    "现在是什么模式",
    "当前模式",
    "现在什么模式",
    "当前是什么场景",
    "现在是什么场景",
    "当前场景",
    "现在什么场景",
    "what mode are you in",
    "what scene are you in",
}


@dataclass(frozen=True)
class ConversationTurn:
    user: str
    assistant: str
    intent: Intent


@dataclass(frozen=True)
class RecentTtsUtterance:
    text: str
    completed_at: float


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
        asr_second_pass_language: str | None = None,
        asr_second_pass_min_language_probability: float = 0.65,
        asr_second_pass_disable_vad: bool = True,
        echo_filter_window_seconds: float = 3.0,
        echo_filter_min_similarity: float = 0.72,
        echo_filter_min_chars: int = 12,
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
        self._asr_second_pass_language = (
            asr_second_pass_language.strip().lower()
            if asr_second_pass_language
            else None
        )
        self._asr_second_pass_min_language_probability = max(
            0.0,
            min(1.0, asr_second_pass_min_language_probability),
        )
        self._asr_second_pass_disable_vad = asr_second_pass_disable_vad
        self._echo_filter_window_seconds = max(0.0, echo_filter_window_seconds)
        self._echo_filter_min_similarity = max(
            0.0,
            min(1.0, echo_filter_min_similarity),
        )
        self._echo_filter_min_chars = max(0, echo_filter_min_chars)
        self._active_learning_mode: LearningMode | None = None
        self._active_family_scene: FamilyScene | None = None
        self._logger = logger or StructuredLogger()
        self._last_tts_completed_at: float | None = None
        self._recent_tts_utterances: deque[RecentTtsUtterance] = deque(maxlen=4)

    def _build_prompt(self, base_prompt: str) -> str:
        prompt_parts = [base_prompt.strip()]
        if self._active_family_scene is not None:
            prompt_parts.append(SCENE_PROMPTS[self._active_family_scene])
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

        if self._is_natural_followup_request(text):
            return last_intent

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

    def _parse_family_scene_command(self, text: str) -> tuple[str, FamilyScene | None] | None:
        normalized = self._normalized_followup_text(text)

        if normalized in {
            "退出场景",
            "关闭场景",
            "exit scene",
            "leave scene",
            "normal family scene",
        }:
            return ("exit", None)

        patterns: list[tuple[FamilyScene, tuple[str, ...]]] = [
            (
                "breakfast",
                (
                    "进入早餐英语场景",
                    "切换到早餐英语场景",
                    "早餐英语场景",
                    "breakfast scene",
                ),
            ),
            (
                "family_time",
                (
                    "进入亲子互动英语场景",
                    "切换到亲子互动英语场景",
                    "亲子互动英语场景",
                    "家庭互动英语场景",
                    "family scene",
                ),
            ),
            (
                "bedtime",
                (
                    "进入睡前英语场景",
                    "切换到睡前英语场景",
                    "睡前英语场景",
                    "bedtime scene",
                ),
            ),
        ]

        for scene, scene_patterns in patterns:
            if normalized in scene_patterns:
                return ("enter", scene)

        return None

    def _handle_family_scene_command(self, text: str) -> bool:
        command = self._parse_family_scene_command(text)
        if command is None:
            return False

        action, scene = command
        if action == "exit":
            previous_scene = self._active_family_scene
            self._active_family_scene = None
            message = "好的，已退出家庭英语场景。"
            if previous_scene is None:
                message = "当前没有开启家庭英语场景。"
            print("[Scene]", message)
            self._logger.info(
                "family_scene.changed",
                action=action,
                previous_scene=previous_scene,
                active_scene=self._active_family_scene,
            )
            self.speak_feedback(message)
            return True

        previous_scene = self._active_family_scene
        self._active_family_scene = scene
        scene_name = SCENE_DISPLAY_NAMES.get(scene or "family_time", "家庭英语场景")
        message = f"好的，已进入{scene_name}。"
        print("[Scene]", message)
        self._logger.info(
            "family_scene.changed",
            action=action,
            previous_scene=previous_scene,
            active_scene=self._active_family_scene,
        )
        self.speak_feedback(message)
        return True

    def _handle_status_query(self, text: str) -> bool:
        normalized = self._normalized_followup_text(text)
        if normalized not in STATUS_QUERY_PATTERNS:
            return False

        mode_name = (
            MODE_DISPLAY_NAMES[self._active_learning_mode]
            if self._active_learning_mode is not None
            else "未开启学习模式"
        )
        scene_name = (
            SCENE_DISPLAY_NAMES[self._active_family_scene]
            if self._active_family_scene is not None
            else "未开启家庭英语场景"
        )
        message = f"当前是{mode_name}，{scene_name}。"
        print("[Status]", message)
        self._logger.info(
            "assistant.status_reported",
            active_learning_mode=self._active_learning_mode,
            active_family_scene=self._active_family_scene,
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

    def _is_natural_followup_request(self, text: str) -> bool:
        return self._normalized_followup_text(text) in {
            "make it more natural",
            "say it more naturally",
            "more natural please",
            "make it more conversational",
            "更自然一点",
            "更口语化一点",
            "更像口语一点",
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

        if self._is_natural_followup_request(text):
            last_turn = self._conversation_memory[-1]
            return (
                "Rewrite your previous reply in a more natural spoken style. "
                "Keep it concise and learner-friendly.\n"
                f"Previous user request:\n{last_turn.user}\n"
                f"Previous assistant reply:\n{last_turn.assistant}"
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

    @staticmethod
    def _normalize_echo_text(text: str) -> str:
        return re.sub(r"[\W_]+", "", text.lower())

    def _remember_tts_output(self, text: str) -> None:
        completed_at = time.monotonic()
        self._last_tts_completed_at = completed_at
        normalized = self._normalize_echo_text(text)
        if not normalized:
            return
        self._recent_tts_utterances.append(
            RecentTtsUtterance(text=text.strip(), completed_at=completed_at)
        )

    def _match_recent_tts_echo(
        self,
        text: str,
    ) -> tuple[RecentTtsUtterance, float] | None:
        if self._echo_filter_window_seconds <= 0:
            return None

        normalized_text = self._normalize_echo_text(text)
        if not normalized_text:
            return None

        now = time.monotonic()
        for utterance in reversed(self._recent_tts_utterances):
            age_seconds = now - utterance.completed_at
            if age_seconds > self._echo_filter_window_seconds:
                break

            normalized_utterance = self._normalize_echo_text(utterance.text)
            if not normalized_utterance:
                continue

            if normalized_text == normalized_utterance:
                return utterance, 1.0

            if len(normalized_text) < self._echo_filter_min_chars:
                continue

            similarity = SequenceMatcher(
                None,
                normalized_text,
                normalized_utterance,
            ).ratio()
            if similarity >= self._echo_filter_min_similarity:
                return utterance, similarity

            shorter, longer = sorted(
                (normalized_text, normalized_utterance),
                key=len,
            )
            if shorter and shorter in longer:
                containment_ratio = len(shorter) / len(longer)
                if containment_ratio >= self._echo_filter_min_similarity:
                    return utterance, containment_ratio

        return None

    def speak_feedback(self, text: str) -> None:
        feedback = text.strip()
        if not feedback:
            return

        try:
            self._tts.speak(feedback)
            self._remember_tts_output(feedback)
            self._logger.info(
                "tts.feedback_completed",
                text_length=len(feedback),
            )
        except Exception as exc:
            self._logger.error(
                "tts.feedback_failed",
                error=str(exc),
            )

    def seconds_since_last_tts(self) -> float | None:
        if self._last_tts_completed_at is None:
            return None
        return max(0.0, time.monotonic() - self._last_tts_completed_at)

    def _transcribe(self, audio: np.ndarray) -> AsrTranscription | None:
        last_error: Exception | None = None
        for attempt in range(1, self._asr_retries + 1):
            started_at = time.perf_counter()
            try:
                result = self._transcribe_once(audio)
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

    def _transcribe_once(self, audio: np.ndarray) -> AsrTranscription:
        if hasattr(self._asr, "transcribe_with_details"):
            primary = self._asr.transcribe_with_details(audio)
        else:
            primary = AsrTranscription(text=self._asr.transcribe(audio))

        return self._maybe_run_second_pass(audio, primary)

    def _maybe_run_second_pass(
        self,
        audio: np.ndarray,
        primary: AsrTranscription,
    ) -> AsrTranscription:
        if not hasattr(self._asr, "transcribe_with_overrides"):
            return primary

        if not self._should_run_second_pass(primary):
            return primary

        override_language = self._asr_second_pass_language
        override_vad = False if self._asr_second_pass_disable_vad else None

        second_pass_started_at = time.perf_counter()
        secondary = self._asr.transcribe_with_overrides(
            audio,
            language=override_language,
            vad_filter=override_vad,
        )
        self._logger.info(
            "asr.second_pass_completed",
            duration_ms=round((time.perf_counter() - second_pass_started_at) * 1000, 2),
            primary_transcript=primary.text,
            secondary_transcript=secondary.text,
            forced_language=override_language,
            vad_filter=override_vad,
            primary_language=primary.language,
            secondary_language=secondary.language,
            primary_language_probability=primary.language_probability,
            secondary_language_probability=secondary.language_probability,
        )

        selected, selected_stage = self._select_better_transcription(primary, secondary)
        self._logger.info(
            "asr.second_pass_selected",
            selected_stage=selected_stage,
            transcript=selected.text,
        )
        return selected

    def _should_run_second_pass(self, transcription: AsrTranscription) -> bool:
        if self._is_low_confidence_transcription(transcription):
            return True

        if not self._asr_second_pass_disable_vad and not self._asr_second_pass_language:
            return False

        if not self._asr_second_pass_language:
            return False

        language_probability = transcription.language_probability
        if language_probability is None:
            return False

        if language_probability >= self._asr_second_pass_min_language_probability:
            return False

        detected_language = (transcription.language or "").strip().lower()
        return detected_language != self._asr_second_pass_language

    def _select_better_transcription(
        self,
        primary: AsrTranscription,
        secondary: AsrTranscription,
    ) -> tuple[AsrTranscription, str]:
        primary_text = primary.text.strip()
        secondary_text = secondary.text.strip()
        primary_low = self._is_low_confidence_transcription(primary)
        secondary_low = self._is_low_confidence_transcription(secondary)

        if not secondary_text:
            return primary, "primary"

        if not primary_text:
            return secondary, "secondary"

        if primary_low and not secondary_low:
            return secondary, "secondary"

        if secondary_low and not primary_low:
            return primary, "primary"

        primary_score = self._transcription_score(primary)
        secondary_score = self._transcription_score(secondary)
        if secondary_score > primary_score + 0.05:
            return secondary, "secondary"

        if (
            self._asr_second_pass_language
            and (secondary.language or "").strip().lower() == self._asr_second_pass_language
            and (
                primary.language_probability is not None
                and primary.language_probability < self._asr_second_pass_min_language_probability
            )
        ):
            return secondary, "secondary"

        return primary, "primary"

    @staticmethod
    def _transcription_score(transcription: AsrTranscription) -> float:
        score = 0.0

        if transcription.text.strip():
            score += 1.0

        if transcription.avg_logprob is not None:
            score += transcription.avg_logprob

        if transcription.no_speech_prob is not None:
            score -= transcription.no_speech_prob

        if transcription.language_probability is not None:
            score += transcription.language_probability * 0.2

        return score

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

    def listen_once(
        self,
        wait_timeout_seconds: float | None = None,
        max_record_seconds: float | None = None,
    ) -> str | None:
        wait_timeout = (
            round(wait_timeout_seconds, 1)
            if wait_timeout_seconds is not None
            else None
        )
        record_limit = (
            round(max_record_seconds, 1)
            if max_record_seconds is not None
            else self._record_seconds
        )
        if wait_timeout is None:
            print(
                f"\nListening... (record up to {record_limit}s after speech starts)"
            )
        else:
            print(
                "\nListening... "
                f"(wait up to {wait_timeout}s for speech, "
                f"record up to {record_limit}s after speech starts)"
            )
        started_at = time.perf_counter()
        try:
            if hasattr(self._recorder, "record_with_limits"):
                audio = self._recorder.record_with_limits(
                    wait_timeout_seconds=wait_timeout_seconds,
                    max_record_seconds=max_record_seconds,
                )
            else:
                try:
                    audio = self._recorder.record(
                        wait_timeout_seconds=wait_timeout_seconds,
                        max_record_seconds=max_record_seconds,
                    )
                except TypeError:
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

        text = transcription.text
        print("Detected text:", repr(text))
        if not text:
            print("Didn't catch anything.")
            self._logger.info("asr.transcribe_empty")
            return None

        matched_echo = self._match_recent_tts_echo(text)
        if matched_echo is not None:
            utterance, similarity = matched_echo
            print("[Echo] ignored probable self-playback.")
            self._logger.info(
                "asr.self_echo_ignored",
                transcript=text,
                matched_tts=utterance.text,
                similarity=round(similarity, 3),
            )
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

        return text

    def handle_text(self, text: str) -> None:
        text = text.strip()
        if not text:
            return

        if self._handle_family_scene_command(text):
            return

        if self._handle_learning_mode_command(text):
            return

        if self._handle_status_query(text):
            return

        intent = self._resolve_intent(text)
        self._logger.info(
            "intent.routed",
            intent=intent,
            user_text=text,
            memory_turns=len(self._conversation_memory),
            active_learning_mode=self._active_learning_mode,
            active_family_scene=self._active_family_scene,
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
            self._remember_tts_output(spoken_response)
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
