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
- Logging and runtime observability (planned)

## Project Structure

```text
eva-robot/
├─ README.md
├─ requirements.txt
├─ src/
│  └─ eva_robot/
│     ├─ __init__.py
│     └─ main.py
└─ tests/
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

This repository also includes a voice-based English robot MVP script:

```bash
python home_english_robot_stable.py
```

Prerequisites:

- Ollama is running locally and accessible at `http://127.0.0.1:11434`
- Whisper model files are available locally (default path is shown below)
- Microphone permission is granted to your terminal/IDE

Environment variables (optional):

```bash
export WHISPER_MODEL_PATH="/Users/mine/.cache/faster-whisper/small"
export WHISPER_DEVICE="cpu"
export WHISPER_COMPUTE_TYPE="int8"
export OLLAMA_URL="http://127.0.0.1:11434/api/generate"
export OLLAMA_MODEL="llama3:latest"
export SAMPLE_RATE="16000"
export RECORD_SECONDS="3"
```

Run tests (when tests are added):

```bash
pytest
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
