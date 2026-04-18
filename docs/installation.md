# Installation

## Basic install

```bash
git clone https://github.com/ccaprani/crucible.git
cd crucible
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Crucible requires Python 3.10 or newer.

## Model backends

Crucible supports six backends. Model routing is automatic — the model name determines the backend.

| Model name pattern | Backend | Auth |
|---|---|---|
| GGUF file name or `.gguf` path | llama.cpp (llama-server) | Local, no auth |
| Starts with `claude-` | Claude Code CLI | Anthropic subscription |
| Starts with `gpt-`, `o1`, `o3`, `o4-`, `codex-` | OpenAI Codex CLI | ChatGPT subscription |
| Starts with `openai-codex/` | OpenAI Codex CLI | ChatGPT subscription |
| Starts with `google-gemini-cli/` | Google Gemini CLI | Google login (free tier) |
| Contains `/` (e.g. `provider/model`) | OpenRouter | API key |
| Everything else | Ollama | Local, no auth |

### llama.cpp (GGUF files)

Build and install [llama.cpp](https://github.com/ggerganov/llama.cpp) so that `llama-server` is on your `$PATH`. Place GGUF model files in `~/models/`:

```bash
# Download a model from Hugging Face
hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf --local-dir ~/models

# Run it
crucible run -m Qwen3.5-9B-Q4_K_M
```

Crucible manages the `llama-server` process automatically — it starts the server before running tests and stops it afterwards. When benchmarking multiple GGUF models, each model is loaded once and all tests run against it before moving to the next, minimising load times.

Substring matching works as with Ollama models:

```bash
crucible run -m qwen3.5          # matches Qwen3.5-9B-Q4_K_M.gguf
```

To use a different models directory:

```bash
crucible run -m qwen3.5 --gguf-dir /data/models
```

### Ollama (local)

Install and run [Ollama](https://ollama.com), then pull at least one model:

```bash
ollama pull gemma4:31b
```

### Claude Code CLI (Anthropic subscription)

If you have a [Claude Code](https://docs.anthropic.com/en/docs/claude-code) subscription, models are called via `claude -p`:

```bash
crucible run -m claude-opus-4 claude-sonnet-4 claude-haiku-4-5
```

Also used as the LLM judge backend (Opus 4.6 with `--effort max`).

### OpenAI Codex CLI (ChatGPT subscription)

If you have the [Codex CLI](https://github.com/openai/codex) installed with a ChatGPT subscription:

```bash
crucible run -m gpt-5.4 gpt-5.3-codex
```

Note: only `gpt-5.4` and `gpt-5.3-codex` are available on the ChatGPT subscription tier. Older models (`gpt-4.1`, `o3`) require an OpenAI API key and should be accessed via OpenRouter instead.

The `openai-codex/` prefix is also supported for explicit routing:

```bash
crucible run -m openai-codex/gpt-5.4
```

### Google Gemini CLI (free tier)

If you have the [Gemini CLI](https://github.com/google-gemini/gemini-cli) installed:

```bash
crucible run -m google-gemini-cli/gemini-2.5-flash
```

Uses your Google login. The free tier has rate limits — for heavy benchmarking, use Gemini models via OpenRouter instead.

### OpenRouter (pay-per-token)

Register your key once:

```bash
crucible register openrouter <your-key>
```

Then use `provider/model` names for any provider:

```bash
crucible run -m google/gemini-3.1-pro-preview x-ai/grok-4
crucible run -m deepseek/deepseek-v3.2 meta-llama/llama-4-maverick
crucible run -m mistralai/mistral-large-2512
```

### Image-generating models

Models that produce raster images (e.g. `openai/gpt-5-image`, `google/gemini-3-pro-image-preview`) are routed through OpenRouter. Generated images are saved to `results/images/` and referenced in the text response. Useful for testing diagram generation tasks where the output modality matters.

```bash
crucible run -m openai/gpt-5-image -n diagram_load_path
```

### Image input (vision)

Crucible supports image input for vision-capable models. Prompts containing `[Image: filename]` references are automatically resolved against the test directory. Images are encoded as base64 and sent via the appropriate API (Ollama multimodal messages or OpenRouter vision format). This enables OCR and site-photo interpretation fixtures.

See [Benchmark Authoring](benchmark-authoring.md) for details on using image references in test prompts.

### Mixing backends

All backends can be mixed freely in a single run:

```bash
crucible run -m Qwen3.5-9B-Q4_K_M gemma4:31b claude-opus-4 gpt-5.4 x-ai/grok-4
```

## Documentation build

The narrative docs are written in Markdown and built with Sphinx using the Shibuya theme.
Install the docs extras with:

```bash
pip install -e .[docs]
```

Then build:

```bash
sphinx-build -b html docs docs/_build/html
```
