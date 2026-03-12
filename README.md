# Eva Robot

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)](#roadmap)

An extensible Python robot assistant focused on conversation, command execution, and automation workflows.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
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

Configuration is loaded from environment variables and from local `.env` / `.env.local`
files if present. Copy `.env.example` to `.env.local` for a safe local setup:

```bash
cp .env.example .env.local
```

Key variables:

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
export OLLAMA_URL="http://127.0.0.1:11434/api/generate"
export OLLAMA_MODEL="qwen2.5:7b-instruct"
export SAMPLE_RATE="16000"
export RECORD_SECONDS="3"
export MIN_RECORD_SECONDS="1.0"
export MAX_RECORD_SECONDS="12.0"
export SILENCE_DURATION_SECONDS="0.8"
export SILENCE_THRESHOLD="0.01"
export NO_SPEECH_TIMEOUT_SECONDS="2.0"
export ASR_VAD_FILTER="true"
export ASR_BEAM_SIZE="5"
export ASR_TEMPERATURE="0.0"
export CONVERSATION_MEMORY_TURNS="3"
export LOG_LEVEL="INFO"
export LOG_FILE_PATH="logs/eva_robot.jsonl"
export SKIP_STARTUP_CHECKS="false"
export WAKE_WORD="伊娃"
export WAKE_ACK_MESSAGE="我在。"
export SLEEP_COMMAND="退下吧"
export SLEEP_ACK_MESSAGE="好的，我先待命。"
export WAKE_TIMEOUT_SECONDS="60"
```

If your wake or sleep phrases include Chinese, keep `ASR_LANGUAGE="auto"` so Whisper can auto-detect instead of forcing English-only transcription.

Provider selection:

- `LLM_PROVIDER=openai_compatible`: uses the remote compatible API and defaults to `gpt-5.1-2025-11-13`
- `LLM_PROVIDER=ollama`: uses the local Ollama model in `OLLAMA_MODEL`
- When `LLM_PROVIDER=openai_compatible`, Eva automatically falls back to local Ollama for the current session if remote preflight or runtime requests fail and Ollama is available

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
- After each reply, Eva stays awake and keeps listening for follow-up questions until timeout
- Eva now gives a short wake/sleep confirmation, and both messages are configurable
- Structured tutor replies are normalized before TTS so they sound more natural when spoken

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
