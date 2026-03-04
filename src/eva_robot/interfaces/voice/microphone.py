import numpy as np
import sounddevice as sd


class MicrophoneRecorder:
    def __init__(self, sample_rate: int, record_seconds: int) -> None:
        self._sample_rate = sample_rate
        self._record_seconds = record_seconds

    def record(self) -> np.ndarray:
        audio = sd.rec(
            int(self._record_seconds * self._sample_rate),
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        return audio.flatten()
