import os
from dataclasses import dataclass
from pathlib import Path


_ORIGINAL_ENV_KEYS = frozenset(os.environ)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]

        if key in _ORIGINAL_ENV_KEYS:
            continue

        os.environ[key] = value


def _load_default_env_files() -> None:
    _load_env_file(_PROJECT_ROOT / ".env")
    _load_env_file(_PROJECT_ROOT / ".env.local")


_load_default_env_files()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional_str(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip()
    if not normalized or normalized.lower() == "auto":
        return None

    return normalized


@dataclass(frozen=True)
class AppConfig:
    whisper_model_path: str = os.getenv(
        "WHISPER_MODEL_PATH", "/Users/mine/.cache/faster-whisper/small"
    )
    whisper_device: str = os.getenv("WHISPER_DEVICE", "cpu")
    whisper_compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    asr_language: str | None = _env_optional_str("ASR_LANGUAGE")
    asr_retries: int = int(os.getenv("ASR_RETRIES", "2"))
    ollama_url: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3:latest")
    sample_rate: int = int(os.getenv("SAMPLE_RATE", "16000"))
    record_seconds: int = int(os.getenv("RECORD_SECONDS", "3"))
    min_record_seconds: float = float(os.getenv("MIN_RECORD_SECONDS", "1.0"))
    max_record_seconds: float = float(os.getenv("MAX_RECORD_SECONDS", "12.0"))
    silence_duration_seconds: float = float(os.getenv("SILENCE_DURATION_SECONDS", "0.8"))
    silence_threshold: float = float(os.getenv("SILENCE_THRESHOLD", "0.01"))
    no_speech_timeout_seconds: float = float(os.getenv("NO_SPEECH_TIMEOUT_SECONDS", "2.0"))
    speech_start_chunks: int = int(os.getenv("SPEECH_START_CHUNKS", "3"))
    preroll_duration_seconds: float = float(os.getenv("PREROLL_DURATION_SECONDS", "0.3"))
    asr_vad_filter: bool = _env_bool("ASR_VAD_FILTER", True)
    asr_beam_size: int = int(os.getenv("ASR_BEAM_SIZE", "5"))
    asr_temperature: float = float(os.getenv("ASR_TEMPERATURE", "0.0"))
    asr_min_avg_logprob: float = float(os.getenv("ASR_MIN_AVG_LOGPROB", "-1.2"))
    asr_max_no_speech_prob: float = float(os.getenv("ASR_MAX_NO_SPEECH_PROB", "0.7"))
    asr_low_confidence_message: str = os.getenv(
        "ASR_LOW_CONFIDENCE_MESSAGE", "抱歉，我没太听清，请再说一遍。"
    )
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai_compatible")
    llm_profile: str = os.getenv("LLM_PROFILE", "default")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://gmncode.cn")
    llm_api_key: str | None = _env_optional_str("LLM_API_KEY")
    llm_model: str | None = _env_optional_str("LLM_MODEL")
    llm_default_model: str = os.getenv("LLM_DEFAULT_MODEL", "gpt-5.1-2025-11-13")
    llm_high_quality_model: str = os.getenv("LLM_HIGH_QUALITY_MODEL", "gpt-5.4")
    llm_timeout_seconds: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    llm_retries: int = int(os.getenv("LLM_RETRIES", "3"))
    llm_preflight_probe: bool = _env_bool("LLM_PREFLIGHT_PROBE", True)
    conversation_memory_turns: int = int(os.getenv("CONVERSATION_MEMORY_TURNS", "3"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file_path: str | None = _env_optional_str("LOG_FILE_PATH", "logs/eva_robot.jsonl")
    skip_startup_checks: bool = _env_bool("SKIP_STARTUP_CHECKS", False)
    wake_word: str = os.getenv("WAKE_WORD", "hello")
    wake_ack_message: str = os.getenv("WAKE_ACK_MESSAGE", "我在。")
    sleep_command: str = os.getenv("SLEEP_COMMAND", "退下吧")
    sleep_ack_message: str = os.getenv("SLEEP_ACK_MESSAGE", "好的，我先待命。")
    wake_timeout_seconds: int = int(os.getenv("WAKE_TIMEOUT_SECONDS", "60"))
    followup_cooldown_seconds: float = float(os.getenv("FOLLOWUP_COOLDOWN_SECONDS", "0.6"))

    def resolved_llm_provider(self) -> str:
        return self.llm_provider.strip().lower()

    def resolved_llm_model(self) -> str:
        if self.llm_model:
            return self.llm_model

        if self.resolved_llm_provider() == "openai_compatible":
            if self.llm_profile.strip().lower() == "high_quality":
                return self.llm_high_quality_model
            return self.llm_default_model

        return self.ollama_model
