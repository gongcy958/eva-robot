import os
from dataclasses import dataclass


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
    llm_timeout_seconds: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    llm_retries: int = int(os.getenv("LLM_RETRIES", "3"))
