import numpy as np
import sounddevice as sd


class MicrophoneRecorder:
    def __init__(
        self,
        sample_rate: int,
        record_seconds: int,
        min_record_seconds: float = 1.0,
        max_record_seconds: float = 12.0,
        silence_duration_seconds: float = 0.8,
        silence_threshold: float = 0.01,
        no_speech_timeout_seconds: float = 2.0,
    ) -> None:
        self._sample_rate = sample_rate
        # Keep RECORD_SECONDS as a compatibility fallback.
        self._max_record_seconds = max(max_record_seconds, float(record_seconds))
        self._min_record_seconds = min_record_seconds
        self._silence_duration_seconds = silence_duration_seconds
        self._silence_threshold = silence_threshold
        self._no_speech_timeout_seconds = no_speech_timeout_seconds

    def record(self) -> np.ndarray:
        chunk_seconds = 0.1
        frames_per_chunk = int(self._sample_rate * chunk_seconds)
        min_chunks = max(1, int(self._min_record_seconds / chunk_seconds))
        max_chunks = max(min_chunks, int(self._max_record_seconds / chunk_seconds))
        silence_chunks_required = max(
            1, int(self._silence_duration_seconds / chunk_seconds)
        )
        no_speech_chunks = max(1, int(self._no_speech_timeout_seconds / chunk_seconds))

        chunks: list[np.ndarray] = []
        speech_started = False
        silence_chunks = 0

        with sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            blocksize=frames_per_chunk,
        ) as stream:
            for i in range(max_chunks):
                data, _overflow = stream.read(frames_per_chunk)
                chunk = data[:, 0].copy()
                chunks.append(chunk)

                max_amp = float(np.max(np.abs(chunk))) if chunk.size else 0.0
                if max_amp >= self._silence_threshold:
                    speech_started = True
                    silence_chunks = 0
                elif speech_started:
                    silence_chunks += 1

                # If nothing is spoken, return quickly so the main loop can continue.
                if not speech_started and i + 1 >= no_speech_chunks:
                    break

                # Once user has spoken enough, stop after sustained silence.
                if (
                    speech_started
                    and i + 1 >= min_chunks
                    and silence_chunks >= silence_chunks_required
                ):
                    break

        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks)
