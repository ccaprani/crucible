# crucible

A bespoke benchmarking tool for comparing AI models on domain-specific tasks. Generic benchmarks tell you how a model performs on average; this tells you how it performs on *your* work.

Runs local models via [Ollama](https://ollama.com) or [llama.cpp](https://github.com/ggerganov/llama.cpp) (GGUF files), cloud models via [OpenRouter](https://openrouter.ai), and subscription-based models via the [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code), [OpenAI Codex CLI](https://github.com/openai/codex), and [Google Gemini CLI](https://github.com/google-gemini/gemini-cli) — all from one command.

The default test suite targets **structural engineering academics** — beam design to AS 3600, reliability analysis, bridge loading standards (AS 5100.2), PhD supervision, technical writing, and scientific Python development. Tests are defined in YAML and trivially extensible to any domain.

Colin Caprani · Monash University · [github.com/ccaprani/crucible](https://github.com/ccaprani/crucible)

## Documentation

Project documentation lives in `docs/` and is built with Sphinx using Markdown pages plus an autodoc API reference.

```bash
pip install -e .[docs]
sphinx-build -b html docs docs/_build/html
```

The docs site includes installation, tutorials, benchmark authoring notes, API reference, and an example standalone HTML report artifact.

## Packaging and release

The PyPI distribution name is `crucible-ai`, while the import package and CLI remain `crucible`.

Release automation is set up through GitHub Actions for:

- CI across supported Python versions
- Sphinx docs deployment to GitHub Pages
- PyPI publishing on GitHub release

## Quick start

```bash
git clone https://github.com/ccaprani/crucible.git
cd crucible
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Requires [Ollama](https://ollama.com) running locally with at least one model pulled, or [llama.cpp](https://github.com/ggerganov/llama.cpp) installed with GGUF model files in `~/models/`.

## Usage

### Interactive mode (recommended)

```bash
crucible
```

Presents a numbered model picker and category selector — no typing model names.

### Direct mode

```bash
crucible run -m gemma4:31b qwen3.5:27b              # full comparison
crucible run -m 2 10                                  # by number from `crucible list`
crucible run -m gemma qwen3.5 -n beam                 # substring match, single test
crucible run -m gemma4:31b -c code_generation          # single category
crucible run --test-dir tests_extra -s tier2           # run an alternate suite
crucible run -m 2 10 --timeout 900 --tokens 4096       # with limits
```

### Cloud and API models

Crucible auto-detects the backend from the model name — six backends, zero configuration:

```bash
# llama.cpp via GGUF files in ~/models/ (auto-manages llama-server)
crucible run -m Qwen3.5-9B-Q4_K_M Qwen3.6-35B-A3B-UD-Q4_K_S

# Ollama (local server)
crucible run -m gemma4:31b qwen3.5:27b

# Claude via CLI (uses your Anthropic subscription)
crucible run -m claude-opus-4 claude-sonnet-4

# OpenAI via Codex CLI (uses your ChatGPT subscription)
crucible run -m gpt-5.4 gpt-5.3-codex

# Google via Gemini CLI (uses your Google login, free tier)
crucible run -m google-gemini-cli/gemini-2.5-flash

# Any provider via OpenRouter (pay-per-token)
crucible run -m google/gemini-3.1-pro-preview x-ai/grok-4
crucible run -m deepseek/deepseek-v3.2 meta-llama/llama-4-maverick

# Image-generating models (raster output saved to results/images/)
crucible run -m openai/gpt-5-image google/gemini-3-pro-image-preview

# Mix local and cloud in one run
crucible run -m Qwen3.5-9B-Q4_K_M claude-opus-4 gpt-5.4 x-ai/grok-4
```

| Model name pattern | Backend | Auth |
|---|---|---|
| GGUF file name or `.gguf` path | llama.cpp (llama-server) | Local, no auth |
| Starts with `claude-` | Claude Code CLI | Anthropic subscription |
| Starts with `gpt-`, `o1`, `o3`, `o4-`, `codex-` | OpenAI Codex CLI | ChatGPT subscription |
| Starts with `openai-codex/` | OpenAI Codex CLI | ChatGPT subscription |
| Starts with `google-gemini-cli/` | Google Gemini CLI | Google login (free tier) |
| Contains `/` (e.g. `provider/model`) | OpenRouter | `crucible register openrouter <key>` |
| Everything else | Ollama | Local, no auth |

**Image models** (e.g. `openai/gpt-5-image`, `google/gemini-3-pro-image-preview`) produce raster images via OpenRouter. Images are saved to `results/images/` and referenced in the response text. The LLM judge evaluates the model's reasoning about the image task; human review of the generated images is recommended via `crucible compare`.

**Image input** is also supported for vision-capable models. Prompts containing `[Image: filename]` references are resolved against the test directory, base64-encoded, and sent via the model's vision API. This enables OCR and site-photo interpretation tests.

### Register API keys

```bash
crucible register openrouter sk-or-v1-abc123...     # store OpenRouter key
crucible register anthropic sk-ant-abc123...         # store Anthropic key
crucible register openrouter                         # check status
```

Keys stored in `~/.config/crucible/keys.json` (owner-only permissions). Environment variables (`OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`) also work and take precedence.

### Quick smoke tests

```bash
crucible run -n beam -m 2 10                           # one test, two models by number
crucible run -n fizzbuzz -m gemma                      # quick code test
crucible run -n traceback -m qwen3.5 --timeout 60      # fast check
crucible run -n beam -m qwen3.5 -v                     # verbose — see the full response
```

The `-n` flag does **substring matching** on test names. Use `crucible list tests` to see all available names.

### Alternate test directories and suites

Crucible normally loads tracked tests from `./tests`.
You can point it at any other directory containing compatible YAML files:

```bash
crucible list tests --test-dir tests_extra
crucible run --test-dir tests_extra -s tier2 -c practice_personas -m gemma4:31b
crucible judge --test-dir tests_extra -s tier2
crucible judge review --test-dir tests_extra -s tier2 -n existing_bridge
```

Notes:

- `--test-dir` recursively loads `*.yaml` under the given directory
- if omitted, Crucible defaults to the repo's tracked `./tests`
- `suite:` can be declared in YAML, or inferred from subfolders beneath the chosen test directory
- test identity is now `suite/category/name`, so duplicate test names are safe across different suites

### Incremental results

Crucible automatically **reuses previous passing results**. If a test passed last night, running the full suite today skips it. Failed, timed-out, and deferred tests are always re-run.

```bash
crucible run -m gemma4 qwen3.5 --timeout 900          # reuses cached passes
crucible run -m gemma4 qwen3.5 --fresh                 # force re-run everything
```

Results are stored in **per-model JSON databases** (`results/gemma4_31b.json`). When a test prompt is edited, the cached result is automatically invalidated via prompt hashing. Each test is saved incrementally — interrupted runs never lose completed work.

### Live response streaming

During a run, the model's response (or thinking) streams live in a red-bordered panel below the progress display. You can see immediately when a model is going off the rails — useful for small models that spin in circles.

### Verbose output

Use `-v` to see full model responses after each test. With 2+ models, responses show side-by-side.

```bash
crucible run -n beam -m gemma4 qwen3.5 -v              # side-by-side comparison
```

### Discover models, tests, and results

```bash
crucible list models       # all models with params, quantisation, VRAM estimate, and tier
crucible list tests        # all test names, grouped by category
crucible list tests --test-dir tests_extra
crucible list prompts      # full prompts for every test
crucible list results      # per-model result databases with pass counts
crucible list              # everything
```

### Visual HTML report

Generate a self-contained HTML dashboard with charts — share it, screenshot it, attach it.

```bash
crucible report                                        # all models, opens in browser
crucible report -m gemma4:31b qwen3.5:27b              # filter to specific models
crucible report -m 2 3 --title "24GB VRAM Tier"        # custom title
crucible report --no-open -o results/report_16gb.html   # save without opening
```

Includes radar chart (per-category scores), pass rates, generation speed (tk/s), per-test grouped bars, and a full results table. Interactive model and tier filter buttons. Each model card shows parameters, quantisation, and VRAM tier.

### Compare responses side-by-side

```bash
crucible compare                                       # all models, all tests
crucible compare -m gemma4:31b claude-opus-4-6         # specific models
crucible compare -n beam                               # single test
crucible compare -c reasoning                          # one category
```

### Judge with Opus

Subjective tests (student feedback, professional emails, standards interpretation) use `llm_judge` evaluation. Opus scores stored model responses via a domain-specific rubric.

```bash
crucible judge                                         # score all llm_judge tests, all models
crucible judge -m gemma4                               # just gemma models
crucible judge -n feedback email                       # just those tests
```

The judge tries (in order): Claude CLI (`claude -p`, uses existing subscription), OpenRouter, Anthropic API. During `crucible run`, `llm_judge` tests are scored automatically using the same cascade.

The summary table groups models by VRAM tier (24GB → 16GB → 8GB → API), sorted by generation speed within each tier.

### Review judge reasoning

See what Opus thought of each model's response — the full judge reasoning alongside a preview of the response:

```bash
crucible judge review                                  # all models, all judge tests
crucible judge review -m qwen3:30b                     # one model
crucible judge review -n feedback                      # one test
crucible judge review -m gemma4 -n email               # combine filters
```

## Thinking model support

Models like `qwen3`, `qwen3.5`, `deepseek-r1`, and `gemma4` use internal chain-of-thought reasoning. Crucible handles these automatically:

- **Token budget** — inflated 4× for thinking models when `--tokens` is set
- **Progress display** — shows "Thinking: 342 tokens · 12.5 tk/s" during reasoning
- **Live streaming** — thinking text visible in the response panel
- **Timing** — TTFT reflects first *content* token, not first thinking token
- **Reporting** — thinking token counts tracked separately

With unlimited tokens (the default), thinking models generate until done — `--timeout` is the safety net.

## VRAM tiers

Models are automatically assigned a VRAM tier based on parameter count and quantisation:

| Tier | VRAM | Example models |
|------|------|---------------|
| **24GB** | 17-24 GB | gemma4:31b, qwen3.5:27b, qwen3:30b-a3b |
| **16GB** | 9-16 GB | qwen3:14b, qwen2.5:14b, mistral-nemo |
| **8GB** | < 9 GB | qwen3.5:9b, phi3.5, llama3.1:8b |

Use `crucible report -m ...` to generate tier-specific reports for sharing.

## Test categories

### Code generation (`code_generation.yaml`)
- **section_properties** — polygon section properties via numpy
- **reliability_pf** — probability of failure using scipy
- **numpy_vectorisation** — refactor loops to vectorised numpy
- **pytest_suite** — write tests for a beam moment function

### Technical writing (`technical_writing.yaml`)
- **structured_explanation** — Monte Carlo reliability as structured JSON
- **student_feedback** — constructive PhD feedback on a lit review gap
- **professional_email** — CPD webinar announcement to chapter chairs
- **standards_interpretation** — AS 5100.2 SM1600 vs T44 loading models

### Reasoning (`reasoning.yaml`)
- **beam_design_as3600** — simply supported beam design to AS 3600
- **load_combination_stats** — statistical combination of random loads
- **dependency_resolution** — Python package version constraint logic
- **supervision_triage** — PhD student time allocation

### Summarization (`summarization.yaml`)
- **code_clause_extraction** — extract load factors from standard text as YAML
- **traceback_triage** — diagnose a Python/OpenSeesPy error
- **abstract_distillation** — extract method/finding/limitation from abstract
- **meeting_notes_extraction** — messy notes to structured JSON

### Generic benchmarks (`generic_benchmarks.yaml`)
- **gsm8k_arithmetic** — multi-step arithmetic word problem
- **logic_puzzle** — seating arrangement deduction
- **code_fizzbuzz_variant** — FizzBuzz sum variant
- **text_extraction** — extract people, amounts, dates from dense text

## Evaluation types

| Type | What it checks |
|------|---------------|
| `code_exec` | Extracts Python, runs it, checks output against expected values |
| `json_valid` | Parses as JSON, optionally checks required keys |
| `yaml_valid` | Parses as YAML, optionally checks required keys |
| `reference` | Compares numeric values against known answers (configurable tolerance) |
| `contains` | Checks for required strings/phrases in response |
| `llm_judge` | Scored by Opus with domain-specific rubric (auto or via `crucible judge`) |
| `manual` | Captured for human review via `crucible compare` |

## Adding your own tests

Create a YAML file in `tests/`:

```yaml
category: my_domain
description: Tests for my specific use case

tests:
  - name: my_test
    description: What this tests
    timeout: 300
    prompt: |
      Your prompt here...
    eval_type: contains
    eval_config:
      required:
        - expected_keyword
      any_of:
        - ["option_a", "option_b"]
```

Crucible picks up new YAML files automatically — no code changes needed.

## Output

- **Console** — Rich table with colour-coded pass/fail, timing, scores, and tk/s
- **Live streaming** — model response/thinking visible in real time during runs
- **Per-model JSON** — `results/<model>.json` living databases, updated each run
- **HTML report** — `crucible report` generates a visual dashboard with charts
- **Markdown** — timestamped markdown reports for archiving

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally (for Ollama models)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) with `llama-server` on PATH (optional — for GGUF models)
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI (optional — Claude models and LLM judge)
- [Codex CLI](https://github.com/openai/codex) (optional — OpenAI models via ChatGPT subscription)
- [Gemini CLI](https://github.com/google-gemini/gemini-cli) (optional — Google models via free tier)
- [OpenRouter](https://openrouter.ai) API key (optional — any provider pay-per-token, including image models)

## Licence

MIT
