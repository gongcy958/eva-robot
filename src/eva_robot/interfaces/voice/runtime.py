import time

from ...application.use_cases.run_voice_turn import RunVoiceTurnUseCase


class VoiceRuntime:
    def __init__(
        self,
        run_voice_turn: RunVoiceTurnUseCase,
        wake_word: str,
        sleep_command: str,
        wake_timeout_seconds: int,
    ) -> None:
        self._run_voice_turn = run_voice_turn
        self._wake_word = wake_word.strip()
        self._sleep_command = sleep_command.strip()
        self._wake_timeout_seconds = wake_timeout_seconds
        self._is_awake = False
        self._last_active_at = 0.0

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

    def _set_sleeping(self) -> None:
        self._is_awake = False
        self._last_active_at = 0.0

    def run(self) -> None:
        print("=== Family English Robot Stable MVP ===")
        print(
            f"Say '{self._wake_word}' to wake me up, "
            f"say '{self._sleep_command}' to put me to sleep."
        )
        print("Press Ctrl+C to quit.")

        while True:
            try:
                if (
                    self._is_awake
                    and time.time() - self._last_active_at > self._wake_timeout_seconds
                ):
                    print("[Wake] timeout reached, entering sleep mode.")
                    self._set_sleeping()

                text = self._run_voice_turn.listen_once()
                if not text:
                    continue

                if self._contains(text, self._sleep_command):
                    print("[Wake] sleep command detected.")
                    self._set_sleeping()
                    continue

                if not self._is_awake:
                    if self._contains(text, self._wake_word):
                        print("[Wake] wake word detected.")
                        self._set_awake()
                    else:
                        print("[Wake] sleeping, ignored.")
                    continue

                self._run_voice_turn.handle_text(text)
                self._last_active_at = time.time()
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as exc:
                print("[Main] unexpected error:", exc)
