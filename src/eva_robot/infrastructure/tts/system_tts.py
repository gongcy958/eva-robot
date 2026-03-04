import os
import platform


class SystemTts:
    def speak(self, text: str) -> None:
        if platform.system() == "Darwin":
            safe_text = text.replace('"', "'")
            code = os.system(f'say "{safe_text}"')
            if code != 0:
                print("[TTS] macOS say failed, response printed only.")
            return

        print("[TTS] Non-macOS detected, speech playback skipped.")
