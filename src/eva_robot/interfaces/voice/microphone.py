from collections import deque
import math

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
        speech_start_chunks: int = 3,
        preroll_duration_seconds: float = 0.3,
        ambient_noise_seconds: float = 0.4,
        speech_start_threshold_multiplier: float = 2.2,
        speech_end_threshold_multiplier: float = 1.6,
    ) -> None:
        self._sample_rate = sample_rate
        # Keep RECORD_SECONDS as a compatibility fallback.
        self._max_record_seconds = max(max_record_seconds, float(record_seconds))
        self._min_record_seconds = min_record_seconds
        self._silence_duration_seconds = silence_duration_seconds
        self._silence_threshold = silence_threshold
        self._no_speech_timeout_seconds = no_speech_timeout_seconds
        self._speech_start_chunks = max(1, speech_start_chunks)
        self._preroll_duration_seconds = max(0.0, preroll_duration_seconds)
        self._ambient_noise_seconds = max(0.0, ambient_noise_seconds)
        self._speech_start_threshold_multiplier = max(
            1.0, speech_start_threshold_multiplier
        )
        self._speech_end_threshold_multiplier = max(
            1.0,
            min(
                self._speech_start_threshold_multiplier,
                speech_end_threshold_multiplier,
            ),
        )

    @staticmethod
    def _chunk_level(chunk: np.ndarray) -> float:
        if chunk.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(chunk, dtype=np.float32))))

    @staticmethod
    def _ambient_level(levels: deque[float]) -> float:
        if not levels:
            return 0.0
        return float(np.percentile(np.asarray(levels, dtype=np.float32), 80))

    def _resolve_thresholds(self, ambient_level: float) -> tuple[float, float]:
        start_threshold = max(
            self._silence_threshold,
            ambient_level * self._speech_start_threshold_multiplier,
        )
        end_threshold = max(
            self._silence_threshold,
            ambient_level * self._speech_end_threshold_multiplier,
        )
        return start_threshold, min(start_threshold, end_threshold)

    def _bootstrap_ambient_level(
        self,
        level: float,
        ambient_level: float,
    ) -> bool:
        bootstrap_threshold = max(
            self._silence_threshold * 6,
            ambient_level * self._speech_start_threshold_multiplier,
        )
        bootstrap_threshold = min(0.12, bootstrap_threshold)
        return level <= bootstrap_threshold

    def record(
        self,
        wait_timeout_seconds: float | None = None,
        max_record_seconds: float | None = None,
    ) -> np.ndarray:
        chunk_seconds = 0.1
        frames_per_chunk = int(self._sample_rate * chunk_seconds)
        min_chunks = max(1, math.ceil(self._min_record_seconds / chunk_seconds))
        max_chunks = max(
            min_chunks,
            math.ceil((max_record_seconds or self._max_record_seconds) / chunk_seconds),
        )
        silence_chunks_required = max(
            1, math.ceil(self._silence_duration_seconds / chunk_seconds)
        )
        no_speech_chunks = max(
            1,
            math.ceil(
                (wait_timeout_seconds or self._no_speech_timeout_seconds) / chunk_seconds
            ),
        )
        preroll_chunks = max(1, math.ceil(self._preroll_duration_seconds / chunk_seconds))
        ambient_chunks = (
            max(1, math.ceil(self._ambient_noise_seconds / chunk_seconds))
            if self._ambient_noise_seconds > 0
            else 0
        )

        chunks: list[np.ndarray] = []
        pending_chunks: deque[np.ndarray] = deque(maxlen=preroll_chunks)
        ambient_levels: deque[float] = deque(maxlen=ambient_chunks)
        speech_started = False
        silence_chunks = 0
        active_chunks = 0
        speech_end_threshold = self._silence_threshold

        with sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            blocksize=frames_per_chunk,
        ) as stream:
            for i in range(no_speech_chunks + max_chunks):
                data, _overflow = stream.read(frames_per_chunk)
                chunk = data[:, 0].copy()
                level = self._chunk_level(chunk)

                if not speech_started:
                    ambient_level = self._ambient_level(ambient_levels)
                    bootstrapped_ambient = False
                    if ambient_chunks and self._bootstrap_ambient_level(
                        level, ambient_level
                    ):
                        ambient_levels.append(level)
                        ambient_level = self._ambient_level(ambient_levels)
                        bootstrapped_ambient = True
                    speech_start_threshold, speech_end_threshold_candidate = (
                        self._resolve_thresholds(ambient_level)
                    )
                    is_active = level >= speech_start_threshold
                    pending_chunks.append(chunk)
                    if not is_active and ambient_chunks and not bootstrapped_ambient:
                        ambient_levels.append(level)
                    if is_active:
                        active_chunks += 1
                        if active_chunks >= self._speech_start_chunks:
                            speech_started = True
                            speech_end_threshold = speech_end_threshold_candidate
                            chunks.extend(pending_chunks)
                            pending_chunks.clear()
                            silence_chunks = 0
                    else:
                        active_chunks = 0

                    if not speech_started and i + 1 >= no_speech_chunks:
                        break
                    continue

                chunks.append(chunk)
                is_active = level >= speech_end_threshold
                if is_active:
                    silence_chunks = 0
                else:
                    silence_chunks += 1

                speech_chunk_count = len(chunks)

                # Once user has spoken enough, stop after sustained silence.
                if (
                    speech_started
                    and speech_chunk_count >= min_chunks
                    and silence_chunks >= silence_chunks_required
                ):
                    break

                if speech_started and speech_chunk_count >= max_chunks:
                    break

        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks)
