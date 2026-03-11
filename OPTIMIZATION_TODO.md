# Eva Robot Optimization TODO

## 1. Current Pain Points

1. ASR cannot fully capture long utterances; recognition accuracy is unstable.
2. LLM replies are not instructional enough for English learning scenarios.
3. TTS voice is fixed and lacks customization.
4. Missing wake-word + sleep control like voice assistants.

## 2. Overall Optimization Goals

1. Improve speech capture completeness and transcription accuracy.
2. Make responses structured, teachable, and context-aware.
3. Support configurable voices and pluggable TTS providers.
4. Build measurable iteration loop (logs + test set + acceptance metrics).

## 3. Execution Rhythm (Suggested 4-Week Plan)

### Week 1: Fix Input Quality (ASR First)

1. Replace fixed-length recording with silence-based auto-stop recording.
2. Enable and tune VAD (`vad_filter=True`).
3. Tune Whisper inference parameters (`beam_size`, `temperature=0`).
4. Add low-confidence confirmation fallback.

Acceptance:
1. Long utterance truncation rate drops significantly.
2. In noisy environment, obvious misrecognitions reduce.
3. End-to-end interaction feels natural without manual timing.

### Week 2: Improve Teaching Quality (LLM + Intent)

1. Add dedicated intent: `translate_and_explain`.
2. Use intent-specific prompts with strict output schema.
3. Add response template:
   - Translation
   - Better natural expression
   - Short explanation
   - 1 example sentence
4. Add short conversation memory (last 3-5 turns).

Acceptance:
1. Query like "今天天气不错的英语怎么说" always returns translation + explanation.
2. Output format is stable and easy for learners.
3. Multi-turn continuity improves.

### Week 3: Voice Experience Upgrade (TTS)

1. Add `TTS_VOICE` configuration and runtime selection.
2. Keep `SystemTts` and add provider interface for future engines.
3. Add optional providers (Edge TTS / Coqui / cloud provider).
4. Add wake/sleep state machine (wake-word + stop listening command).

Acceptance:
1. Voice can be switched without code changes.
2. TTS provider can be replaced by configuration.
3. Assistant responds only after wake-word and can enter sleep mode by command.

### Week 4: Stabilization and Measurement

1. Add structured logs for each pipeline stage:
   - recording duration
   - ASR duration/result
   - intent
   - LLM latency
   - TTS latency
2. Build a regression dataset (30-50 representative cases).
3. Add regression script and baseline report.

Acceptance:
1. Each change can be measured against baseline.
2. Regressions are detected early.

## 4. Detailed TODO Backlog (Priority Ordered)

## P0 (Do First)

1. Dynamic recording with silence timeout.
2. VAD enabled and tuned.
3. Whisper model/parameter benchmark (`small` vs `medium` if hardware allows).
4. Intent enhancement for translation/explanation scenarios.
5. Structured prompt/output for educational answers.

## P1 (Do Next)

1. Context window memory management.
2. TTS voice selection and provider abstraction.
3. Prompt versioning and A/B evaluation for response quality.
4. Basic failure-recovery strategy (ASR retry, LLM retry with fallback reply).
5. Wake-word and sleep/close command control.

## P2 (Scale Later)

1. Add lightweight semantic router (rule + classifier hybrid).
2. User profile and personalized tutoring difficulty.
3. Session history persistence.
4. Optional web dashboard for logs/metrics.

## 5. Architecture-Oriented Implementation Suggestions

1. Keep all external capabilities behind ports/interfaces:
   - ASR port
   - LLM port
   - TTS port
   - Audio input port
2. Keep business logic in `application/use_cases`.
3. Keep routing and prompt policies in `domain`.
4. Keep provider-specific logic in `infrastructure`.
5. Keep runtime settings in `shared/config` with env overrides.

## 6. Suggested New/Updated Configs

```bash
# ASR
ASR_VAD_ENABLED=true
ASR_BEAM_SIZE=5
ASR_TEMPERATURE=0
SILENCE_TIMEOUT_MS=800
MAX_RECORD_SECONDS=12
MIN_RECORD_SECONDS=1

# LLM
LLM_TIMEOUT_SECONDS=30
LLM_RETRIES=3
LLM_RESPONSE_STYLE=tutor

# TTS
TTS_PROVIDER=system
TTS_VOICE=Alex
TTS_RATE=1.0

# WAKE
WAKE_WORDS="eva,eva robot"
SLEEP_COMMANDS="stop listening,go to sleep,休眠,关闭"
WAKE_TIMEOUT_SECONDS=20
HOTWORD_MODE=prefix
```

## 7. Quality Metrics (Track Weekly)

1. ASR truncation rate (% of utterances cut off).
2. ASR correction rate (% turns needing user repeat).
3. Translation answer completeness (% with all required fields).
4. End-to-end latency (P50/P95).
5. Task success rate on regression set.

## 8. Risks and Mitigations

1. Risk: Better ASR model increases latency.
   - Mitigation: benchmark and choose profile by hardware tier.
2. Risk: Prompt becomes verbose and slow.
   - Mitigation: enforce output schema and token limits.
3. Risk: Too many providers increase maintenance complexity.
   - Mitigation: keep strict provider interface and contract tests.

## 9. Immediate Next Sprint Checklist (Actionable)

1. Implement silence-based recorder in `interfaces/voice/microphone.py`.
2. Expose ASR tuning in `shared/config.py`.
3. Update `infrastructure/asr/faster_whisper_asr.py` with new params.
4. Add `translate_and_explain` in `domain/intents.py`.
5. Add new prompt template and strict output format.
6. Add one smoke regression script under `tests/` or `scripts/`.
7. Add wake/sleep controller and wire it into voice runtime loop.

## 11. Wake/Sleep Feature Design (XiaoAi-like)

1. Introduce runtime states:
   - `SLEEPING`: ignore normal utterances.
   - `AWAKE`: process user requests.
2. Wake flow:
   - Detect wake words (e.g., \"Eva\", \"Eva Robot\").
   - Optional response: \"I'm here.\"
3. Sleep/close flow:
   - Detect sleep commands (e.g., \"go to sleep\", \"关闭\").
   - Switch to `SLEEPING` and confirm.
4. Timeout flow:
   - If no valid command in `WAKE_TIMEOUT_SECONDS`, auto-sleep.
5. Implementation location:
   - State machine in `interfaces/voice/runtime.py`.
   - Phrase matching utility in `domain` (or `application` policy).
   - Config in `shared/config.py`.

Acceptance:
1. Before wake word, normal speech does not trigger LLM.
2. After wake word, request can be handled in one turn.
3. Sleep command reliably disables active handling.
4. Auto-sleep works after timeout.

## 10. Definition of Done (Per Feature)

1. Code merged with config + docs update.
2. At least one measurable metric improved.
3. No regression in core voice interaction flow.
4. Manual test cases pass (normal, noisy, long sentence, translation question).
