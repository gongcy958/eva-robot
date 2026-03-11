from ...application.use_cases.run_voice_turn import RunVoiceTurnUseCase
from ...domain.intents import IntentRouter
from ...infrastructure.asr.faster_whisper_asr import FasterWhisperAsr
from ...infrastructure.llm.openai_compatible_client import OpenAiCompatibleLlmClient
from ...infrastructure.llm.ollama_client import OllamaLlmClient
from ...infrastructure.tts.system_tts import SystemTts
from ..voice.microphone import MicrophoneRecorder
from ..voice.runtime import VoiceRuntime
from ...shared.config import AppConfig
from ...shared.observability import StructuredLogger, configure_logging
from ...shared.preflight import StartupPreflight


def _build_llm_client(config: AppConfig, logger: StructuredLogger):
    provider = config.resolved_llm_provider()
    model = config.resolved_llm_model()

    if provider == "openai_compatible":
        if not config.llm_api_key:
            raise ValueError(
                "LLM_API_KEY is required when LLM_PROVIDER=openai_compatible."
            )

        logger.info(
            "llm.provider_selected",
            provider=provider,
            model=model,
            profile=config.llm_profile,
            base_url=config.llm_base_url,
        )
        return OpenAiCompatibleLlmClient(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
            model=model,
            timeout_seconds=config.llm_timeout_seconds,
            retries=config.llm_retries,
            logger=logger,
        )

    if provider == "ollama":
        logger.info(
            "llm.provider_selected",
            provider=provider,
            model=model,
            profile=config.llm_profile,
            base_url=config.ollama_url,
        )
        return OllamaLlmClient(
            url=config.ollama_url,
            model=model,
            timeout_seconds=config.llm_timeout_seconds,
            retries=config.llm_retries,
            logger=logger,
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER={config.llm_provider!r}. "
        "Use 'openai_compatible' or 'ollama'."
    )


def main() -> None:
    config = AppConfig()
    configure_logging(config.log_level, config.log_file_path)
    logger = StructuredLogger()
    StartupPreflight(config, logger).run()

    recorder = MicrophoneRecorder(
        sample_rate=config.sample_rate,
        record_seconds=config.record_seconds,
        min_record_seconds=config.min_record_seconds,
        max_record_seconds=config.max_record_seconds,
        silence_duration_seconds=config.silence_duration_seconds,
        silence_threshold=config.silence_threshold,
        no_speech_timeout_seconds=config.no_speech_timeout_seconds,
    )
    asr = FasterWhisperAsr(
        model_path=config.whisper_model_path,
        device=config.whisper_device,
        compute_type=config.whisper_compute_type,
        language=config.asr_language,
        vad_filter=config.asr_vad_filter,
        beam_size=config.asr_beam_size,
        temperature=config.asr_temperature,
    )
    router = IntentRouter()
    llm = _build_llm_client(config, logger)
    tts = SystemTts()

    use_case = RunVoiceTurnUseCase(
        recorder=recorder,
        asr=asr,
        router=router,
        llm=llm,
        tts=tts,
        record_seconds=int(config.max_record_seconds),
        conversation_memory_turns=config.conversation_memory_turns,
        asr_retries=config.asr_retries,
        asr_min_avg_logprob=config.asr_min_avg_logprob,
        asr_max_no_speech_prob=config.asr_max_no_speech_prob,
        asr_low_confidence_message=config.asr_low_confidence_message,
        logger=logger,
    )
    runtime = VoiceRuntime(
        run_voice_turn=use_case,
        wake_word=config.wake_word,
        wake_ack_message=config.wake_ack_message,
        sleep_command=config.sleep_command,
        sleep_ack_message=config.sleep_ack_message,
        wake_timeout_seconds=config.wake_timeout_seconds,
        logger=logger,
    )
    runtime.run()


if __name__ == "__main__":
    main()
