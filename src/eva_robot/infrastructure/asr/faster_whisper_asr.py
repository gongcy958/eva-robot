from __future__ import annotations

from faster_whisper import WhisperModel

from ...application.services.ports import AsrTranscription


class FasterWhisperAsr:
    def __init__(
        self,
        model_path: str,
        device: str,
        compute_type: str,
        language: str | None = None,
        vad_filter: bool = True,
        beam_size: int = 5,
        temperature: float = 0.0,
    ) -> None:
        self._model = WhisperModel(
            model_path,
            device=device,
            compute_type=compute_type,
        )
        self._language = language
        self._vad_filter = vad_filter
        self._beam_size = beam_size
        self._temperature = temperature

    def transcribe_with_details(self, audio: object) -> AsrTranscription:
        segments, _info = self._model.transcribe(
            audio,
            language=self._language,
            vad_filter=self._vad_filter,
            beam_size=self._beam_size,
            temperature=self._temperature,
        )
        segment_list = list(segments)
        text = " ".join(seg.text for seg in segment_list).strip()
        avg_logprob = None
        no_speech_prob = None
        if segment_list:
            avg_logprob = sum(seg.avg_logprob for seg in segment_list) / len(segment_list)
            no_speech_prob = sum(seg.no_speech_prob for seg in segment_list) / len(segment_list)
        return AsrTranscription(
            text=text,
            avg_logprob=avg_logprob,
            no_speech_prob=no_speech_prob,
            language=getattr(_info, "language", None),
            language_probability=getattr(_info, "language_probability", None),
            segment_count=len(segment_list),
        )

    def transcribe(self, audio: object) -> str:
        return self.transcribe_with_details(audio).text
