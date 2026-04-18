# Benchmark Authoring

Crucible benchmarks are defined in YAML.
Each test file can represent a suite, a category family, or any other organization that suits the benchmark program.

## Core ideas

- A benchmark is a collection of YAML test definitions.
- Each test has a stable identity of `suite/category/name`.
- `suite:` can be stated explicitly in YAML or inferred from subfolders under the chosen test directory.
- Different benchmark programs can therefore reuse test names safely.

## Minimal example

```yaml
suite: default
category: reasoning
tests:
  - name: beam_design_as3600
    prompt: |
      Design the tensile reinforcement for a simply supported reinforced concrete beam.
    eval:
      type: contains
      must_contain:
        - reinforcement
        - AS 3600
```

## Evaluation modes

Crucible supports:

- `code_exec`
- `json_valid`
- `yaml_valid`
- `reference`
- `contains`
- `llm_judge`
- `manual`

The right evaluation mode depends on the task. Objective tasks should use executable or structured checks where possible. Judgement-heavy tasks can use `llm_judge` with a rubric.

## Capability-aware suites

Crucible is designed for more than a single flat benchmark.
A good benchmark program can include:

- `expected` tasks for the capability class the model is meant to handle
- `boundary` tasks near the edge of plausible use
- `misuse` tasks that should fail or be deemed inadmissible

This is particularly important when benchmarking different capability classes such as LLMs, OCR models, vision models, and embedding models.

## Test directories

By default Crucible loads tracked tests from `./tests`.
You can point it at another directory with `--test-dir`:

```bash
crucible run --test-dir tests_extra -s tier2
```

Crucible will recursively load `*.yaml` files under that directory.

## Image input in prompts

Crucible supports embedding images in test prompts for vision-capable models. To include an image, use the `[Image: filename]` tag in your prompt text. The image file should be placed in an `images/` subdirectory within the test directory.

An optional description can follow the filename after ` — ` (em dash), which is stripped during resolution and not sent to the model:

```yaml
tests:
  - name: bridge_damage_assessment
    prompt: |
      Examine the following photograph of a bridge soffit.
      Identify the type and severity of damage visible.

      [Image: soffit_crack.jpg — Bridge soffit damage]
    eval:
      type: llm_judge
      rubric: |
        Award marks for correctly identifying crack type, location, and severity.
```

Images are base64-encoded and sent via the appropriate API — Ollama multimodal messages for local models, or OpenRouter vision format for cloud models. This enables OCR tasks, site-photo interpretation, and diagram reading fixtures.
