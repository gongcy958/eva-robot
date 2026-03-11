import platform
import subprocess


class SystemTts:
    def speak(self, text: str) -> None:
        if not text.strip():
            return

        if platform.system() == "Darwin":
            try:
                result = subprocess.run(
                    ["say", text],
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
