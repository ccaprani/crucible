# Design Notes

## Why Crucible exists

Generic model benchmarks are useful, but they do not tell you whether a model is fit for a specific professional workflow.
Crucible is designed to benchmark domain-specific work directly, with an emphasis on:

- extensible YAML-defined suites
- mixed objective and rubric-based scoring
- live execution visibility
- per-model incremental storage
- shareable HTML reporting

## Architecture

The runtime is intentionally small:

- [`crucible.run`](https://github.com/ccaprani/crucible/blob/main/crucible/run.py) provides the CLI.
- [`crucible.runner`](https://github.com/ccaprani/crucible/blob/main/crucible/runner.py) loads tests and executes them.
- [`crucible.models`](https://github.com/ccaprani/crucible/blob/main/crucible/models.py) routes requests to Ollama, OpenRouter, or Claude CLI.
- [`crucible.evaluate`](https://github.com/ccaprani/crucible/blob/main/crucible/evaluate.py) scores objective outputs.
- [`crucible.judge`](https://github.com/ccaprani/crucible/blob/main/crucible/judge.py) handles rubric-based model judging.
- [`crucible.report`](https://github.com/ccaprani/crucible/blob/main/crucible/report.py) stores results and builds the standalone HTML report.

## Result identity

Stored results are keyed by a stable test identifier:

```text
suite/category/name
```

This avoids collisions between different benchmark programs and allows category or suite-specific comparisons and reports.

## Related work

Crucible sits alongside general benchmark frameworks and domain-specific suites, but focuses on workflow-grounded benchmarking rather than one universal leaderboard.
That makes it a better fit for research and professional programs where the benchmark itself must reflect the work being done.
