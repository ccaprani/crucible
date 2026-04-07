# crucible

A bespoke benchmarking tool for comparing local AI models via [Ollama](https://ollama.com). Generic benchmarks tell you how a model performs on average; this tells you how it performs on *your* work.

The default test suite targets **structural engineering academics** — beam design to AS 3600, reliability analysis, bridge loading standards (AS 5100.2), PhD supervision, technical writing, and scientific Python development. Tests are defined in YAML and trivially extensible to any domain.

## Quick start

```bash
git clone https://github.com/ccaprani/crucible.git
cd crucible
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Requires [Ollama](https://ollama.com) running locally with at least one model pulled.

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
crucible run -m 2 10 --timeout 900 --tokens 4096       # with limits
```

### Quick smoke tests

```bash
crucible run -n beam -m 2 10                           # one test, two models by number
crucible run -n fizzbuzz -m gemma                      # quick code test
crucible run -n traceback -m qwen3.5 --timeout 60      # fast check
crucible run -n beam -m qwen3.5 -v                     # verbose — see the full response
```

The `-n` flag does **substring matching** on test names. Use `crucible list tests` to see all available names.

### Incremental results

Crucible automatically **reuses previous passing results**. If a test passed last night, running the full suite today skips it. Failed, timed-out, and deferred tests are always re-run.

```bash
crucible run -m gemma4 qwen3.5 --timeout 900          # reuses cached passes
crucible run -m gemma4 qwen3.5 --fresh                 # force re-run everything
```

Results are stored in **per-model JSON databases** (`results/gemma4_31b.json`). When a test prompt is edited, the cached result is automatically invalidated via prompt hashing.

### Verbose output

Use `-v` to see full model responses. With 2+ models, responses show side-by-side.

```bash
crucible run -n beam -m gemma4 qwen3.5 -v              # side-by-side comparison
```

### Discover models, tests, and results

```bash
crucible list models       # all models with params, quantisation, VRAM estimate, and tier
crucible list tests        # all test names, grouped by category
crucible list prompts      # full prompts for every test
crucible list results      # per-model result databases with pass counts
crucible list              # everything
```

Model listing shows quantisation and VRAM tier to help pick models for your GPU:

```
Ollama models (14):
   1  deepseek-coder:6.7b  7B · Q4_0 · ~5.9GB · 8GB
   2  gemma4:26b  25.8B · Q4_K_M · ~17.5GB · 24GB
   3  gemma4:31b  31.3B · Q4_K_M · ~20.8GB · 24GB
   ...
```

### Visual HTML report

Generate a self-contained HTML dashboard with charts — share it, screenshot it, attach it.

```bash
crucible report                                        # all models, opens in browser
crucible report -m gemma4:31b qwen3.5:27b              # filter to specific models
crucible report -m 2 3 --title "24GB VRAM Tier"        # custom title
crucible report --no-open -o results/report_16gb.html   # save without opening
```

Includes radar chart (per-category scores), pass rates, generation speed (tk/s), per-test grouped bars, and a full results table. Each model card shows parameters, quantisation, and VRAM tier.

### Compare responses side-by-side

```bash
crucible compare                                       # all models, all tests
crucible compare -n beam                               # single test
crucible compare -c reasoning                          # one category
```

### Judge with Claude Code

Subjective tests (student feedback, professional emails, standards interpretation) are scored using [Claude Code](https://claude.ai/claude-code) via `--print` mode — uses your Max subscription, not API tokens.

```bash
crucible judge results/gemma4_31b.json                 # all llm_judge tests
crucible judge results/gemma4_31b.json -n feedback     # subset
```

## Thinking model support

Models like `qwen3`, `qwen3.5`, `deepseek-r1`, and `gemma4` use internal chain-of-thought reasoning. Crucible handles these automatically:

- **Token budget** — inflated 4× for thinking models when `--tokens` is set
- **Progress display** — shows "Thinking: 342 tokens" during reasoning
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
| `llm_judge` | Scored by Claude with domain-specific rubric via `crucible judge` |
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
- **Per-model JSON** — `results/<model>.json` living databases, updated each run
- **HTML report** — `crucible report` generates a visual dashboard with charts
- **Markdown** — timestamped markdown reports for archiving

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally
- [Claude Code](https://claude.ai/claude-code) for `crucible judge` (optional)

## Licence

MIT
