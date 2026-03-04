from __future__ import annotations

from faster_whisper import WhisperModel


class FasterWhisperAsr:
    def __init__(
        self,
        model_path: str,
        device: str,
        compute_type: str,
        vad_filter: bool = True,
        beam_size: int = 5,
        temperature: float = 0.0,
    ) -> None:
        self._model = WhisperModel(
            model_path,
            device=device,
            compute_type=compute_type,
        )
        self._vad_filter = vad_filter
        self._beam_size = beam_size
        self._temperature = temperature

    def transcribe(self, audio: object) -> str:
        segments, _info = self._model.transcribe(
            audio,
            language="en",
            vad_filter=self._vad_filter,
            beam_size=self._beam_size,
            temperature=self._temperature,
        )
        return " ".join(seg.text for seg in segments).strip()
