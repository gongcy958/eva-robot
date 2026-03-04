from ...application.use_cases.run_voice_turn import RunVoiceTurnUseCase


class VoiceRuntime:
    def __init__(self, run_voice_turn: RunVoiceTurnUseCase) -> None:
        self._run_voice_turn = run_voice_turn

    def run(self) -> None:
        print("=== Family English Robot Stable MVP ===")
        print("Speak English, AI will respond. Press Ctrl+C to quit.")

        while True:
            try:
                self._run_voice_turn.run_once()
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as exc:
                print("[Main] unexpected error:", exc)
