# Eva Robot

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)](#roadmap)

An extensible Python robot assistant focused on conversation, command execution, and automation workflows.

Language: English | [简体中文](README.zh-CN.md)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Quick Start](#quick-start)
- [5-Minute Setup](#5-minute-setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Changelog](#changelog)
- [Contact](#contact)
- [License](#license)

## Overview

Eva Robot is designed as a modular assistant that can grow from a simple chat bot into a task-driven automation agent.

## Features

- Conversation and command handling
- Task routing and execution pipeline
- Modular architecture for future plugins
- Structured logging and runtime observability

## Project Structure

```text
eva-robot/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ home_english_robot_stable.py
├─ src/
│  └─ eva_robot/
│     ├─ main.py
│     ├─ interfaces/
│     │  ├─ cli/
│     │  └─ voice/
│     ├─ application/
│     │  ├─ services/
│     │  └─ use_cases/
│     ├─ domain/
│     ├─ infrastructure/
│     │  ├─ asr/
│     │  ├─ llm/
│     │  └─ tts/
│     └─ shared/
```

## Getting Started

### Prerequisites

- Python 3.10 or later
- `pip`

### Installation

```bash
git clone https://github.com/gongcy958/eva-robot.git
cd eva-robot
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

After cloning the repo, the fastest way to get Eva running locally is:

```bash
cp .env.example .env.local
```

Then choose **one** of the following startup modes.

## 5-Minute Setup

If you just want to get Eva running as quickly as possible, use one of the
copy-paste setups below.

### Fastest local setup with Ollama

```bash
cp .env.example .env.local
ollama pull qwen2.5:7b-instruct
```

Put this in `.env.local`:

```bash
LLM_PROVIDER=ollama
OLLAMA_URL=http://127.0.0.1:11434/api/generate
OLLAMA_MODEL=qwen2.5:7b-instruct

WHISPER_MODEL_PATH=small
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
ASR_LANGUAGE=auto
```

Then run:

```bash
python -m src.eva_robot.main
```

### Fastest remote setup

Put this in `.env.local`:

```bash
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://your-openai-compatible-endpoint.example.com/v1
LLM_API_KEY=sk-your_api_key_here
LLM_MODEL=gpt-4o-mini

WHISPER_MODEL_PATH=small
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
ASR_LANGUAGE=auto
```

Then run:

```bash
python -m src.eva_robot.main
```

If you also want local fallback, keep the remote setup above and add:

```bash
OLLAMA_URL=http://127.0.0.1:11434/api/generate
OLLAMA_MODEL=qwen2.5:7b-instruct
```

### Option A: Remote OpenAI-compatible model

Use this when you have a remote provider that exposes OpenAI-compatible
`/models` and `/responses` endpoints.

Edit `.env.local` like this:

```bash
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://your-openai-compatible-endpoint.example.com/v1
LLM_API_KEY=sk-your_api_key_here
LLM_MODEL=gpt-4o-mini

WHISPER_MODEL_PATH=small
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
ASR_LANGUAGE=auto
```

Notes:

- `LLM_BASE_URL` should be the base URL your provider gives you. If your
  provider says the base URL is already `/v1`, keep it as-is.
- `LLM_API_KEY` is required for `LLM_PROVIDER=openai_compatible`.
- `LLM_MODEL` is optional but recommended so startup uses the exact remote
  model you expect.
- `WHISPER_MODEL_PATH` can be a local filesystem path, or a named
  faster-whisper model such as `small`, `medium`, or `large-v3`.

### Option B: Local Ollama model only

Use this when you want everything local except ASR/TTS dependencies.

First make sure Ollama is installed and running, then pull a model:

```bash
ollama pull qwen2.5:7b-instruct
```

Set `.env.local` like this:

```bash
LLM_PROVIDER=ollama
OLLAMA_URL=http://127.0.0.1:11434/api/generate
OLLAMA_MODEL=qwen2.5:7b-instruct

WHISPER_MODEL_PATH=small
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
ASR_LANGUAGE=auto
```

Notes:

- Eva expects the Ollama generate endpoint by default:
  `http://127.0.0.1:11434/api/generate`
- `OLLAMA_MODEL` must exist locally, otherwise startup preflight will fail.

### Option C: Remote primary + local Ollama fallback

This is the most practical setup for day-to-day use: prefer a remote model, but
keep Ollama ready when the remote API is unavailable.

```bash
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://your-openai-compatible-endpoint.example.com/v1
LLM_API_KEY=sk-your_api_key_here
LLM_MODEL=gpt-4o-mini

OLLAMA_URL=http://127.0.0.1:11434/api/generate
OLLAMA_MODEL=qwen2.5:7b-instruct

WHISPER_MODEL_PATH=small
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
ASR_LANGUAGE=auto
```

When `LLM_PROVIDER=openai_compatible`, Eva will automatically fall back to the
local Ollama model for the current session if remote preflight or runtime
requests fail and Ollama is healthy.

### Minimum startup checklist

Before running Eva, make sure all of these are true:

- Python dependencies are installed with `pip install -r requirements.txt`
- `.env.local` exists and has the LLM settings you want
- `WHISPER_MODEL_PATH` points to a valid local model path or a valid named model
- Your terminal/IDE has microphone permission
- If using Ollama, the Ollama app or server is running

### Run Eva

Main package entry:

```bash
python -m src.eva_robot.main
```

Compatibility script:

```bash
python home_english_robot_stable.py
```

## Usage

Run the application:

```bash
python -m src.eva_robot.main
```

### English Robot MVP (Root Script)

This repository also includes a compatibility entry script:

```bash
python home_english_robot_stable.py
```

Prerequisites:

- Ollama is running locally and accessible at `http://127.0.0.1:11434`
- Whisper model files are available locally (default path is shown below)
- Microphone permission is granted to your terminal/IDE

Configuration is loaded from environment variables and from local `.env` /
`.env.local` files if present. Copy `.env.example` to `.env.local` for a safe
local setup:

```bash
cp .env.example .env.local
```

For a quick first run, the **minimum** variables you usually need are:

```bash
# Remote model
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://your-openai-compatible-endpoint.example.com/v1
LLM_API_KEY=sk-your_api_key_here
LLM_MODEL=gpt-4o-mini

# OR local Ollama
LLM_PROVIDER=ollama
OLLAMA_URL=http://127.0.0.1:11434/api/generate
OLLAMA_MODEL=qwen2.5:7b-instruct

# ASR
WHISPER_MODEL_PATH=small
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
ASR_LANGUAGE=auto
```

Full environment variables:

```bash
export LLM_PROVIDER="openai_compatible"
export LLM_PROFILE="default"  # or high_quality
export LLM_DEFAULT_MODEL="gpt-5.1-2025-11-13"
export LLM_HIGH_QUALITY_MODEL="gpt-5.4"
export LLM_BASE_URL="https://gmncode.cn"
export LLM_API_KEY="your_api_key_here"
export LLM_PREFLIGHT_PROBE="true"
export WHISPER_MODEL_PATH="/Users/mine/.cache/faster-whisper/small"
export WHISPER_DEVICE="cpu"
export WHISPER_COMPUTE_TYPE="int8"
export ASR_LANGUAGE="auto"
export ASR_RETRIES="2"
export ASR_MIN_AVG_LOGPROB="-1.2"
export ASR_MAX_NO_SPEECH_PROB="0.7"
export ASR_LOW_CONFIDENCE_MESSAGE="抱歉，我没太听清，请再说一遍。"
export ASR_SECOND_PASS_LANGUAGE="en"
export ASR_SECOND_PASS_MIN_LANGUAGE_PROBABILITY="0.65"
export ASR_SECOND_PASS_DISABLE_VAD="true"
export ECHO_FILTER_WINDOW_SECONDS="3.0"
export ECHO_FILTER_MIN_SIMILARITY="0.72"
export ECHO_FILTER_MIN_CHARS="12"
export LOW_CONFIDENCE_CONFIRMATION_TIMEOUT_SECONDS="12.0"
export OLLAMA_URL="http://127.0.0.1:11434/api/generate"
export OLLAMA_MODEL="qwen2.5:7b-instruct"
export SAMPLE_RATE="16000"
export RECORD_SECONDS="3"
export MIN_RECORD_SECONDS="1.0"
export MAX_RECORD_SECONDS="12.0"
export SILENCE_DURATION_SECONDS="0.8"
export SILENCE_THRESHOLD="0.01"
export NO_SPEECH_TIMEOUT_SECONDS="2.0"
export SPEECH_START_CHUNKS="3"
export PREROLL_DURATION_SECONDS="0.3"
export AMBIENT_NOISE_SECONDS="0.4"
export SPEECH_START_THRESHOLD_MULTIPLIER="2.2"
export SPEECH_END_THRESHOLD_MULTIPLIER="1.6"
export FOLLOWUP_COOLDOWN_SECONDS="0.6"
export ASR_VAD_FILTER="true"
export ASR_BEAM_SIZE="5"
export ASR_TEMPERATURE="0.0"
export CONVERSATION_MEMORY_TURNS="3"
export LOG_LEVEL="INFO"
export LOG_FILE_PATH="logs/eva_robot.jsonl"
export SKIP_STARTUP_CHECKS="false"
export TTS_VOICE=""
export TTS_RATE=""
export WAKE_WORD="伊娃"
export WAKE_ACK_MESSAGE="我在。"
export INLINE_WAKE_ACK_MESSAGE=""
export SLEEP_COMMAND="退下吧"
export SLEEP_ACK_MESSAGE="好的，我先待命。"
export WAKE_TIMEOUT_SECONDS="60"
```

If your wake or sleep phrases include Chinese, keep `ASR_LANGUAGE="auto"` so Whisper can auto-detect instead of forcing English-only transcription.

If your main practice language is English, set `ASR_SECOND_PASS_LANGUAGE="en"` so
weak auto-detected transcripts get one more pass with forced English and VAD
disabled. If you mainly speak Chinese, leave it as `auto`/empty instead.

If follow-up listening starts too early because of ambient noise, try increasing
`SPEECH_START_CHUNKS` or `SILENCE_THRESHOLD`. `PREROLL_DURATION_SECONDS` keeps a
small amount of audio before speech detection so the first syllable is less
likely to be clipped.

If the room has constant background noise, Eva now calibrates a short ambient
noise floor before speech starts. Increase `AMBIENT_NOISE_SECONDS` to make that
baseline steadier, or raise `SPEECH_START_THRESHOLD_MULTIPLIER` when fans,
air-conditioners, or keyboard noise still trigger early recording. Keep
`SPEECH_END_THRESHOLD_MULTIPLIER` lower than the start multiplier so Eva can
stay locked onto a real utterance without clipping the ending.

If Eva sometimes re-hears its own reply, try headphones first, then increase
`FOLLOWUP_COOLDOWN_SECONDS` slightly, for example from `0.6` to `0.8`.
If that is not enough, keep `ECHO_FILTER_WINDOW_SECONDS` enabled so Eva can
ignore transcripts that closely match its own recent TTS output.

When ASR confidence is weak but not empty, Eva now asks a short confirmation
question instead of always forcing a full repeat. `LOW_CONFIDENCE_CONFIRMATION_TIMEOUT_SECONDS`
controls how long Eva keeps that pending clarification alive.

Provider selection:

- `LLM_PROVIDER=openai_compatible`: uses the remote compatible API and defaults to `gpt-5.1-2025-11-13`
- `LLM_PROVIDER=ollama`: uses the local Ollama model in `OLLAMA_MODEL`
- When `LLM_PROVIDER=openai_compatible`, Eva automatically falls back to local Ollama for the current session if remote preflight or runtime requests fail and Ollama is available

On macOS, Eva can also use a specific system voice with `TTS_VOICE` and an
optional speaking speed with `TTS_RATE`. To list the voices installed on your
machine, run:

```bash
say -v '?'
```

Examples that are available on this machine include:

- `Eddy (中文（中国大陆）)`
- `Flo (中文（中国大陆）)`
- `Eddy (英语（美国）)`
- `Grandma (英语（美国）)`
- `Daniel`

Example configuration:

```bash
export TTS_VOICE="Eddy (中文（中国大陆）)"
export TTS_RATE="190"
```

One-command mode switching:

```bash
./scripts/use_default_mode.sh
./scripts/use_high_quality_mode.sh
```

Startup preflight now runs automatically before the voice loop. It checks:

- Whisper model path availability
- Active LLM backend connectivity
- OpenAI-compatible API key and lightweight model probe
- Ollama endpoint reachability and selected local model
- If the remote backend fails but Ollama is healthy, startup continues with Ollama instead of exiting

Wake interaction is smoother now:

- You can say wake word and command in one sentence, for example `伊娃，帮我翻译 hello`
- Inline wake commands skip the full wake confirmation by default so the answer starts faster
- After each reply, Eva stays awake and keeps listening for follow-up questions until timeout
- Eva now gives a short wake/sleep confirmation, and both messages are configurable
- Structured tutor replies are normalized before TTS so they sound more natural when spoken

If you still want a short spoken cue for inline commands, set
`INLINE_WAKE_ACK_MESSAGE` to a brief phrase such as `好的` or `嗯`.

Learning mode is now stateful. You can say commands like:

- `进入翻译模式`
- `进入纠错模式`
- `进入语法模式`
- `进入口语模式`
- `退出模式`

Family English scenes are also stateful. You can say:

- `进入早餐英语场景`
- `进入亲子互动英语场景`
- `进入睡前英语场景`
- `退出场景`

You can also ask Eva:

- `当前是什么模式`
- `当前是什么场景`

The runtime keeps a short in-memory conversation history and emits structured
JSON logs to stdout for audio, ASR, intent routing, LLM, TTS, and wake/sleep
events.

The same JSON logs can also be written to a local file such as
`logs/eva_robot.jsonl` for later debugging and replay.

If ASR confidence looks too low, Eva now asks you to repeat instead of trying
to answer based on a shaky transcription.

Automated tests are planned but are not included in the repository yet. When a
test suite is added, it can be run with:

```bash
pytest
```

For a lightweight local regression check of routing, learning mode, and
follow-up behavior, run:

```bash
python3 scripts/smoke_regression.py
```

For microphone tuning on your own machine, run:

```bash
python3 scripts/calibrate_voice_frontend.py
```

This captures a few seconds of room noise plus a short spoken sample, then
prints recommended values for:

- `SILENCE_THRESHOLD`
- `AMBIENT_NOISE_SECONDS`
- `SPEECH_START_THRESHOLD_MULTIPLIER`
- `SPEECH_END_THRESHOLD_MULTIPLIER`
- `SPEECH_START_CHUNKS`
- `PREROLL_DURATION_SECONDS`
- `FOLLOWUP_COOLDOWN_SECONDS`

If the recommendations sound right, rerun with `--write-env` to update
`.env.local` directly.

## Configuration

A future `.env`-based setup is recommended for runtime configuration, such as:

- API keys
- Runtime mode (`dev`, `prod`)
- Logging level

## Roadmap

- [ ] Deliver minimal runnable MVP
- [ ] Add command routing with handler registry
- [ ] Add scheduled task support
- [ ] Add plugin system and external integrations
- [ ] Improve observability and monitoring

## Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a branch: `git checkout -b feature/your-feature`.
3. Commit your changes: `git commit -m "feat: add your feature"`.
4. Push to your fork and open a Pull Request.

Please keep changes focused, tested, and documented.

## Changelog

All notable changes will be documented in this section.

### [0.1.0] - 2026-03-04

- Initialized project structure
- Added foundational README

## Contact

Project Maintainer: `gongcy`  
GitHub: [gongcy](https://github.com/gongcy958)

## License

MIT License. See [LICENSE](./LICENSE) for details.
