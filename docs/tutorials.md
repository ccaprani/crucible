# Tutorials

## First benchmark run

Run the default tracked suite against two local models:

```bash
crucible run -m gemma4:31b qwen3.5:27b
```

This will:

- discover tests from `./tests`
- stream model output live
- reuse prior passing results when possible
- save results incrementally per model in `results/*.json`

## Run an alternate benchmark directory

Point Crucible at any directory containing compatible YAML files:

```bash
crucible list tests --test-dir tests_extra
crucible run --test-dir tests_extra -s tier2 -c practice_personas -m gemma4:31b
```

This is useful for:

- private research suites
- experimental benchmark families
- sector-specific or client-specific task sets

## Judge subjective tasks

For `llm_judge` tests:

```bash
crucible judge --test-dir tests_extra -s tier2
crucible judge review --test-dir tests_extra -s tier2 -n existing_bridge
```

## Generate a report

```bash
crucible report --title "24GB Tier Benchmark"
```

This writes a standalone HTML report, by default to `results/report.html`, and opens it in your browser unless `--no-open` is supplied.

## Run an overnight campaign

The project includes a sequential campaign runner for long unattended batches:

```bash
python3 paper_dev/run_campaign.py --no-stop-on-error
```

This runs one benchmark process at a time to avoid GPU contention.
