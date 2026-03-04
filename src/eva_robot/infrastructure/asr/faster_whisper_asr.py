from __future__ import annotations

from faster_whisper import WhisperModel


class FasterWhisperAsr:
    def __init__(self, model_path: str, device: str, compute_type: str) -> None:
        self._model = WhisperModel(
            model_path,
            device=device,
            compute_type=compute_type,
        )

    def transcribe(self, audio: object) -> str:
        segments, _info = self._model.transcribe(audio, language="en", vad_filter=False)
        return " ".join(seg.text for seg in segments).strip()
