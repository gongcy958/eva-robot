import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AppConfig:
    whisper_model_path: str = os.getenv(
        "WHISPER_MODEL_PATH", "/Users/mine/.cache/faster-whisper/small"
    )
    whisper_device: str = os.getenv("WHISPER_DEVICE", "cpu")
    whisper_compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3:latest")
    sample_rate: int = int(os.getenv("SAMPLE_RATE", "16000"))
    record_seconds: int = int(os.getenv("RECORD_SECONDS", "3"))
    min_record_seconds: float = float(os.getenv("MIN_RECORD_SECONDS", "1.0"))
    max_record_seconds: float = float(os.getenv("MAX_RECORD_SECONDS", "12.0"))
    silence_duration_seconds: float = float(os.getenv("SILENCE_DURATION_SECONDS", "0.8"))
    silence_threshold: float = float(os.getenv("SILENCE_THRESHOLD", "0.01"))
    no_speech_timeout_seconds: float = float(os.getenv("NO_SPEECH_TIMEOUT_SECONDS", "2.0"))
    asr_vad_filter: bool = _env_bool("ASR_VAD_FILTER", True)
    asr_beam_size: int = int(os.getenv("ASR_BEAM_SIZE", "5"))
    asr_temperature: float = float(os.getenv("ASR_TEMPERATURE", "0.0"))
    llm_timeout_seconds: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    llm_retries: int = int(os.getenv("LLM_RETRIES", "3"))
    wake_word: str = os.getenv("WAKE_WORD", "伊娃")
    sleep_command: str = os.getenv("SLEEP_COMMAND", "退下吧")
    wake_timeout_seconds: int = int(os.getenv("WAKE_TIMEOUT_SECONDS", "60"))
