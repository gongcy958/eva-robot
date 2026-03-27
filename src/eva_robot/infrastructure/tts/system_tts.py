import platform
import subprocess


class SystemTts:
    def __init__(
        self,
        voice: str | None = None,
        rate: int | None = None,
    ) -> None:
        self._voice = (voice or "").strip() or None
        self._rate = rate if rate is None else max(80, min(360, int(rate)))

    def _build_say_command(self, text: str) -> list[str]:
        command = ["say"]
        if self._voice:
            command.extend(["-v", self._voice])
        if self._rate is not None:
            command.extend(["-r", str(self._rate)])
        command.append(text)
        return command

    def speak(self, text: str) -> None:
        if not text.strip():
            return

        if platform.system() == "Darwin":
            try:
                result = subprocess.run(
                    self._build_say_command(text),
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except OSError as exc:
                print(f"[TTS] macOS say unavailable: {exc}")
                return

            if result.returncode != 0:
                error_output = result.stderr.strip()
                if error_output:
                    print(f"[TTS] macOS say failed: {error_output}")
                else:
                    print("[TTS] macOS say failed, response printed only.")
            return

        print("[TTS] Non-macOS detected, speech playback skipped.")
