import re
import time

from ...application.use_cases.run_voice_turn import RunVoiceTurnUseCase
from ...shared.observability import StructuredLogger


class VoiceRuntime:
    def __init__(
        self,
        run_voice_turn: RunVoiceTurnUseCase,
        wake_word: str,
        wake_ack_message: str,
        sleep_command: str,
        sleep_ack_message: str,
        wake_timeout_seconds: int,
        logger: StructuredLogger | None = None,
    ) -> None:
        self._run_voice_turn = run_voice_turn
        self._wake_word = wake_word.strip()
        self._wake_ack_message = wake_ack_message.strip()
        self._sleep_command = sleep_command.strip()
        self._sleep_ack_message = sleep_ack_message.strip()
        self._wake_timeout_seconds = wake_timeout_seconds
        self._is_awake = False
        self._last_active_at = 0.0
        self._logger = logger or StructuredLogger()

    @staticmethod
    def _normalize(text: str) -> str:
        return "".join(text.lower().split())

    def _contains(self, text: str, phrase: str) -> bool:
        if not phrase:
            return False
        return self._normalize(phrase) in self._normalize(text)

    def _strip_prefix_phrase(self, text: str, phrase: str) -> str:
        phrase = " ".join(phrase.strip().split())
        if not phrase:
            return text.strip()

        tokens = [re.escape(token) for token in phrase.split()]
        pattern = re.compile(
            r"^\s*" + r"\s*".join(tokens) + r"[\s,，。.!?？:：;；、-]*",
            re.IGNORECASE,
        )
        return pattern.sub("", text, count=1).strip()

    def _strip_leading_phrase_repetitions(self, text: str, phrase: str) -> str:
        stripped = text.strip()
        while True:
            updated = self._strip_prefix_phrase(stripped, phrase)
            if updated == stripped:
                return stripped
            stripped = updated

    def _extract_inline_command(self, text: str, phrase: str) -> str | None:
        stripped = self._strip_leading_phrase_repetitions(text, phrase)
        if stripped and stripped != text.strip():
            return stripped
        return None

    def _announce_followup_listening(self) -> None:
        print("[Wake] still listening for follow-up...")
        self._logger.info(
            "runtime.awaiting_followup",
            wake_timeout_seconds=self._wake_timeout_seconds,
        )

    def _set_awake(self) -> None:
        self._is_awake = True
        self._last_active_at = time.time()
        self._logger.info("runtime.awake")

    def _set_sleeping(self) -> None:
        self._is_awake = False
        self._last_active_at = 0.0
        self._logger.info("runtime.sleeping")

    def _remaining_awake_seconds(self) -> float:
        if not self._is_awake:
            return 0.0
        return max(0.1, self._wake_timeout_seconds - (time.time() - self._last_active_at))

    def run(self) -> None:
        print("=== Family English Robot Stable MVP ===")
        print(
            f"Say '{self._wake_word}' to wake me up, "
            f"say '{self._sleep_command}' to put me to sleep."
        )
        print("Press Ctrl+C to quit.")
        self._logger.info(
            "runtime.started",
            wake_word=self._wake_word,
            sleep_command=self._sleep_command,
            wake_timeout_seconds=self._wake_timeout_seconds,
        )

        while True:
            try:
                if (
                    self._is_awake
                    and time.time() - self._last_active_at > self._wake_timeout_seconds
                ):
                    print("[Wake] timeout reached, entering sleep mode.")
                    self._logger.info("runtime.wake_timeout")
                    self._run_voice_turn.speak_feedback(self._sleep_ack_message)
                    self._set_sleeping()

                wait_timeout_seconds = (
                    self._remaining_awake_seconds() if self._is_awake else None
                )
                text = self._run_voice_turn.listen_once(
                    wait_timeout_seconds=wait_timeout_seconds,
                )
                if not text:
                    continue

                if self._contains(text, self._sleep_command):
                    print("[Wake] sleep command detected.")
                    self._logger.info("runtime.sleep_command_detected", text=text)
                    self._run_voice_turn.speak_feedback(self._sleep_ack_message)
                    self._set_sleeping()
                    continue

                if not self._is_awake:
                    inline_command = self._extract_inline_command(text, self._wake_word)
                    if inline_command is not None:
                        print("[Wake] wake word and command detected.")
                        self._logger.info(
                            "runtime.inline_command_detected",
                            text=text,
                            command_text=inline_command,
                        )
                        self._set_awake()
                        self._run_voice_turn.speak_feedback(self._wake_ack_message)
                        self._run_voice_turn.handle_text(inline_command)
                        self._last_active_at = time.time()
                        self._announce_followup_listening()
                    elif self._contains(text, self._wake_word):
                        print("[Wake] wake word detected. Listening for your request...")
                        self._logger.info("runtime.wake_word_detected", text=text)
                        self._set_awake()
                        self._run_voice_turn.speak_feedback(self._wake_ack_message)
                        self._announce_followup_listening()
                    else:
                        print("[Wake] sleeping, ignored.")
                        self._logger.info("runtime.sleeping_ignored", text=text)
                    continue

                awake_text = text.strip()
                if self._contains(text, self._wake_word):
                    wake_remainder = self._strip_leading_phrase_repetitions(
                        text,
                        self._wake_word,
                    )
                    if not wake_remainder:
                        self._logger.info("runtime.wake_word_while_awake", text=text)
                        self._announce_followup_listening()
                        continue
                    awake_text = wake_remainder

                if not awake_text:
                    self._logger.info("runtime.empty_followup_ignored", text=text)
                    self._announce_followup_listening()
                    continue

                if awake_text != text:
                    self._logger.info(
                        "runtime.wake_prefix_stripped",
                        original_text=text,
                        command_text=awake_text,
                    )

                self._run_voice_turn.handle_text(awake_text)
                self._last_active_at = time.time()
                self._announce_followup_listening()
            except KeyboardInterrupt:
                print("\nExiting...")
                self._logger.info("runtime.stopped")
                break
            except Exception as exc:
                print("[Main] unexpected error:", exc)
                self._logger.error("runtime.unexpected_error", error=str(exc))
