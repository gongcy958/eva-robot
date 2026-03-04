from ...application.use_cases.run_voice_turn import RunVoiceTurnUseCase
from ...domain.intents import IntentRouter
from ...infrastructure.asr.faster_whisper_asr import FasterWhisperAsr
from ...infrastructure.llm.ollama_client import OllamaLlmClient
from ...infrastructure.tts.system_tts import SystemTts
from ..voice.microphone import MicrophoneRecorder
from ..voice.runtime import VoiceRuntime
from ...shared.config import AppConfig


def main() -> None:
    config = AppConfig()

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
        vad_filter=config.asr_vad_filter,
        beam_size=config.asr_beam_size,
        temperature=config.asr_temperature,
    )
    router = IntentRouter()
    llm = OllamaLlmClient(
        url=config.ollama_url,
        model=config.ollama_model,
        timeout_seconds=config.llm_timeout_seconds,
        retries=config.llm_retries,
    )
    tts = SystemTts()

    use_case = RunVoiceTurnUseCase(
        recorder=recorder,
        asr=asr,
        router=router,
        llm=llm,
        tts=tts,
        record_seconds=int(config.max_record_seconds),
    )
    runtime = VoiceRuntime(
        run_voice_turn=use_case,
        wake_word=config.wake_word,
        sleep_command=config.sleep_command,
        wake_timeout_seconds=config.wake_timeout_seconds,
    )
    runtime.run()


if __name__ == "__main__":
    main()
