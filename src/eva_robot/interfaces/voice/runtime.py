import time

from ...application.use_cases.run_voice_turn import RunVoiceTurnUseCase
from ...shared.observability import StructuredLogger


class VoiceRuntime:
    def __init__(
        self,
        run_voice_turn: RunVoiceTurnUseCase,
        wake_word: str,
        sleep_command: str,
        wake_timeout_seconds: int,
        logger: StructuredLogger | None = None,
    ) -> None:
        self._run_voice_turn = run_voice_turn
        self._wake_word = wake_word.strip()
        self._sleep_command = sleep_command.strip()
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

    def _set_awake(self) -> None:
        self._is_awake = True
        self._last_active_at = time.time()
        self._logger.info("runtime.awake")

    def _set_sleeping(self) -> None:
        self._is_awake = False
        self._last_active_at = 0.0
        self._logger.info("runtime.sleeping")

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
                    self._set_sleeping()

                text = self._run_voice_turn.listen_once()
                if not text:
                    continue

                if self._contains(text, self._sleep_command):
                    print("[Wake] sleep command detected.")
                    self._logger.info("runtime.sleep_command_detected", text=text)
                    self._set_sleeping()
                    continue

                if not self._is_awake:
                    if self._contains(text, self._wake_word):
                        print("[Wake] wake word detected.")
                        self._logger.info("runtime.wake_word_detected", text=text)
                        self._set_awake()
                    else:
                        print("[Wake] sleeping, ignored.")
                        self._logger.info("runtime.sleeping_ignored", text=text)
                    continue

                self._run_voice_turn.handle_text(text)
                self._last_active_at = time.time()
            except KeyboardInterrupt:
                print("\nExiting...")
                self._logger.info("runtime.stopped")
                break
            except Exception as exc:
                print("[Main] unexpected error:", exc)
                self._logger.error("runtime.unexpected_error", error=str(exc))
