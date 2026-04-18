# crucible

Crucible is a domain-specific AI benchmarking tool for real engineering work.
It compares local and cloud models against structured YAML test suites, streams live responses during execution, and produces self-contained HTML reports for sharing and review.

The project is built around a simple idea: generic leaderboards do not tell you whether a model is fit for *your* workflow.
Crucible lets you define that workflow, run the benchmark, and inspect the results at the level of suites, categories, and individual tests.

## What Crucible Covers

- Local models via Ollama, frontier models via OpenRouter, and Claude via the CLI.
- Objective checks such as code execution, JSON/YAML validity, and numeric references.
- Subjective scoring via `llm_judge` for judgement-heavy tasks.
- Suite-aware result identity so different benchmark programs can coexist cleanly.
- Standalone HTML reports that open locally and can be shared directly.

## Typical Workflow

1. Install Crucible and pull or register the models you want to test.
2. Run the default tracked test suite, or point Crucible at an alternate benchmark directory.
3. Judge the subjective tasks.
4. Generate a standalone HTML report and compare results across models.

## Documentation Map

```{toctree}
:maxdepth: 2
:hidden:

installation
tutorials
benchmark-authoring
reports
design-notes
api/index
```

## Related Project Material

- [GitHub repository](https://github.com/ccaprani/crucible)
- [README quick start](https://github.com/ccaprani/crucible#readme)
- [Example standalone report](./_static/example-report.html)

## Why the Report Matters

The generated report is a product feature in its own right. It is not a thin export layer.
Crucible writes a self-contained HTML dashboard with charts, model cards, grouped per-test results, and filter controls. That makes it easy to inspect results locally, email an artifact to a collaborator, or archive a benchmark run as evidence.

See [Reports](reports.md) for the report workflow and the example artifact.
