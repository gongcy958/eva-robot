# Eva Robot

[English README](README.md)

一个以语音交互、英语陪练、翻译和家庭场景对话为核心的 Python 机器人助手项目。

## 快速开始

### 1. 克隆项目并安装依赖

```bash
git clone https://github.com/gongcy958/eva-robot.git
cd eva-robot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 复制本地配置文件

```bash
cp .env.example .env.local
```

复制之后，优先编辑 `.env.local`，不要直接改 `.env.example`。

---

## 5 分钟跑通

如果你只是想先把项目在本地跑起来，可以直接照下面二选一配置。

### 最快本地启动：Ollama

```bash
cp .env.example .env.local
ollama pull qwen2.5:7b-instruct
```

把下面这些写进 `.env.local`：

```bash
LLM_PROVIDER=ollama
OLLAMA_URL=http://127.0.0.1:11434/api/generate
OLLAMA_MODEL=qwen2.5:7b-instruct

WHISPER_MODEL_PATH=small
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
ASR_LANGUAGE=auto
```

然后运行：

```bash
python -m src.eva_robot.main
```

### 最快远程启动：OpenAI-compatible

把下面这些写进 `.env.local`：

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

然后运行：

```bash
python -m src.eva_robot.main
```

如果你想同时保留本地兜底，再补上：

```bash
OLLAMA_URL=http://127.0.0.1:11434/api/generate
OLLAMA_MODEL=qwen2.5:7b-instruct
```

---

## 三种启动方式

拉下源码后，最关键的是先决定你要用：

- 远程模型
- 本地 Ollama 模型
- 远程模型 + 本地 Ollama 兜底

下面分别说明。

### 方式 A：使用远程 OpenAI-compatible 模型

适合你已经有远程模型服务，并且对方提供了兼容 OpenAI 的接口。

`.env.local` 最小配置示例：

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

说明：

- `LLM_BASE_URL`：填你的远程模型服务给你的 base URL
- `LLM_API_KEY`：填对应的 API Key
- `LLM_MODEL`：建议显式指定远程模型名，避免启动时用到不是你预期的模型
- `WHISPER_MODEL_PATH`：可以填本地路径，也可以直接填 `small`、`medium`、`large-v3` 这类命名模型

---

### 方式 B：只使用本地 Ollama 模型

适合你希望本地直接跑起来，不依赖远程接口。

先确保 Ollama 已安装并启动，然后拉一个模型：

```bash
ollama pull qwen2.5:7b-instruct
```

然后配置 `.env.local`：

```bash
LLM_PROVIDER=ollama
OLLAMA_URL=http://127.0.0.1:11434/api/generate
OLLAMA_MODEL=qwen2.5:7b-instruct

WHISPER_MODEL_PATH=small
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
ASR_LANGUAGE=auto
```

说明：

- `OLLAMA_URL` 默认就是 `http://127.0.0.1:11434/api/generate`
- `OLLAMA_MODEL` 必须是你本地已经 pull 下来的模型名

---

### 方式 C：远程模型优先，本地 Ollama 兜底

这是最推荐的启动方式。

好处是：

- 远程模型正常时，优先用远程
- 远程模型挂了或不可用时，自动切回本地 Ollama

配置示例：

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

当 `LLM_PROVIDER=openai_compatible` 时，如果远程预检失败或者运行时请求失败，而本地 Ollama 正常，Eva 会自动切到本地模型继续当前会话。

---

## 最低启动检查清单

在第一次运行前，至少确认下面几件事：

- 已执行 `pip install -r requirements.txt`
- 已有 `.env.local`
- `.env.local` 中已经填写好远程模型或本地模型配置
- `WHISPER_MODEL_PATH` 是有效本地路径，或者有效命名模型
- 终端 / IDE 已开启麦克风权限
- 如果你用 Ollama，本机 Ollama 已经启动

---

## 启动命令

主入口：

```bash
python -m src.eva_robot.main
```

兼容脚本入口：

```bash
python home_english_robot_stable.py
```

---

## 常用配置说明

### 远程模型相关

- `LLM_PROVIDER=openai_compatible`
- `LLM_BASE_URL`：远程模型服务的 base URL
- `LLM_API_KEY`：远程模型服务的密钥
- `LLM_MODEL`：推荐显式填写

### 本地模型相关

- `LLM_PROVIDER=ollama`
- `OLLAMA_URL=http://127.0.0.1:11434/api/generate`
- `OLLAMA_MODEL=qwen2.5:7b-instruct`

### ASR 相关

- `WHISPER_MODEL_PATH=small`
- `WHISPER_DEVICE=cpu`
- `WHISPER_COMPUTE_TYPE=int8`
- `ASR_LANGUAGE=auto`

如果你只是想先跑起来，`small + cpu + int8 + auto` 是一个比较省事的起点。

---

## TTS 语音配置

在 macOS 上，可以配置系统语音和语速：

```bash
TTS_VOICE=Eddy (英语（美国）)
TTS_RATE=180
```

列出本机可用语音：

```bash
say -v '?'
```

---

## 推荐的首次体验路径

如果你是第一次拉源码，建议按下面顺序：

1. 先用本地 Ollama 跑通
2. 再加远程模型配置
3. 最后根据自己的设备和环境微调语音参数

如果你要做语音调参，可以运行：

```bash
python3 scripts/calibrate_voice_frontend.py
```

如果要把推荐参数写入 `.env.local`：

```bash
python3 scripts/calibrate_voice_frontend.py --write-env
```

---

## 其他文档

- 英文版：`README.md`
- 周末语音测试清单：`WEEKEND_VOICE_TEST_CHECKLIST.md`
