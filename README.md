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
crucible run -m 2 10 --timeout 300 --tokens 4096       # with timeout and token limit
```

### Quick smoke tests

Run a single test against a couple of models to check things are working:

```bash
crucible run -n beam -m 2 10                           # one test, two models by number
crucible run -n fizzbuzz -m gemma                      # quick code test, substring model match
crucible run -n section -m 1                           # section_properties against first model
crucible run -n beam -m qwen3.5 -v                     # verbose — see the full response
crucible run -n traceback -m qwen3.5 --timeout 60      # fast timeout for a quick check
```

The `-n` flag does **substring matching** on test names — `beam` matches `beam_design_as3600`, `section` matches `section_properties`, etc. Use `crucible list tests` to see all available names.

### Incremental results

Crucible automatically **reuses previous passing results**. If you ran `beam_design` last night and it passed, running the full suite today skips it and injects the cached result — no wasted compute.

```bash
crucible run -m gemma4 qwen3.5 --timeout 900          # skips previously passing tests
crucible run -m gemma4 qwen3.5 --fresh                 # force re-run everything
```

Previously passing results show as `♻ 100% (reused)` in the summary table. Failed, timed-out, and deferred tests are always re-run.

### Verbose output

Use `-v` to see full model responses after each test — great for single-test debugging. With 2+ models, responses are shown side-by-side.

```bash
crucible run -n beam -m gemma4 qwen3.5 -v              # side-by-side responses
crucible run -n fizzbuzz -m 1 -v                        # single model, full output
```

### Discover available tests and models

```bash
crucible list tests        # all test names, grouped by category, with eval types
crucible list prompts      # full prompts for every test
crucible list models       # numbered model list (use numbers with -m)
crucible list              # everything: models, tests, and previous results
```

Test names shown in `crucible list tests` are what you pass to `-n`. Category names shown as group headings are what you pass to `-c`.

### List results

```bash
crucible list results      # previous runs
```

### Compare responses side-by-side

```bash
crucible compare results/2026-04-06_123456.json              # all tests
crucible compare results/2026-04-06_123456.json -n beam       # single test
crucible compare results/2026-04-06_123456.json -c reasoning  # one category
```

Shows the prompt (if stored) followed by model responses in side-by-side panels.

### Judge with Claude Code (no API tokens)

Subjective tests (student feedback, professional emails, standards interpretation) are captured during `run` and scored later using [Claude Code](https://claude.ai/claude-code) via `--print` mode — uses your Max subscription, not API tokens.

```bash
crucible judge results/2026-04-06_123456.json              # all llm_judge tests
crucible judge results/2026-04-06_123456.json -n feedback   # subset
```

## Thinking model support

Models like `qwen3`, `qwen3.5`, `deepseek-r1`, and `gemma4` use internal chain-of-thought reasoning. Crucible handles these automatically:

- **Token budget** — thinking tokens count against `num_predict`, so Crucible inflates the budget (4×) when `--tokens` is set
- **Progress display** — shows "Thinking: 342 tokens" during the reasoning phase
- **Timing** — TTFT reflects time to first *content* token, not first thinking token
- **Reporting** — thinking token counts tracked separately in results

With unlimited tokens (the default), thinking models generate until done — the `--timeout` is the safety net.

## Test categories

### Code generation (`code_generation.yaml`)
- **section_properties** — polygon section properties via numpy (validated against known rectangle)
- **reliability_pf** — probability of failure using scipy (analytical + numerical)
- **numpy_vectorisation** — refactor loop-based code to vectorised numpy
- **pytest_suite** — write tests for a beam moment function

### Technical writing (`technical_writing.yaml`)
- **structured_explanation** — Monte Carlo reliability as structured JSON
- **student_feedback** — constructive PhD feedback on a lit review gap
- **professional_email** — CPD webinar announcement to chapter chairs
- **standards_interpretation** — AS 5100.2 SM1600 vs T44 loading models

### Reasoning (`reasoning.yaml`)
- **beam_design_as3600** — simply supported beam design with rectangular stress block
- **load_combination_stats** — statistical combination of random loads
- **dependency_resolution** — Python package version constraint logic
- **supervision_triage** — PhD student time allocation with justification

### Summarization (`summarization.yaml`)
- **code_clause_extraction** — extract load factors from standard text as YAML
- **traceback_triage** — diagnose a Python/OpenSeesPy error
- **abstract_distillation** — extract method/finding/limitation from a research abstract
- **meeting_notes_extraction** — messy notes to structured JSON (decisions, actions, unresolved)

### Generic benchmarks (`generic_benchmarks.yaml`)
- **gsm8k_arithmetic** — multi-step arithmetic word problem
- **logic_puzzle** — seating arrangement deduction
- **code_fizzbuzz_variant** — FizzBuzz sum variant
- **text_extraction** — extract people, amounts, dates from dense text as JSON

## Evaluation types

| Type | What it checks |
|------|---------------|
| `code_exec` | Extracts Python from response, runs it, checks output against expected values |
| `json_valid` | Parses as JSON, optionally checks required keys |
| `yaml_valid` | Parses as YAML, optionally checks required keys |
| `reference` | Compares numeric values against known answers (configurable tolerance) |
| `contains` | Checks for required strings/phrases in response |
| `llm_judge` | Deferred to `crucible judge` — scored by Claude with domain-specific rubric |
| `manual` | Captured for human review via `crucible compare` |

## Adding your own tests

Create a YAML file in `tests/`:

```yaml
category: my_domain
description: Tests for my specific use case

tests:
  - name: my_test
    description: What this tests
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

Each run produces:
- **Console** — Rich table with colour-coded pass/fail, timing, and scores
- **Markdown** — `results/YYYY-MM-DD_HHMMSS.md` with full prompts, responses, and evaluation details
- **JSON** — `results/YYYY-MM-DD_HHMMSS.json` for programmatic analysis and `crucible judge`/`compare`

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally
- [Claude Code](https://claude.ai/claude-code) for `crucible judge` (optional)

## Licence

MIT
