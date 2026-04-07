"""Report generation — Rich console tables, per-model JSON database, markdown export."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

from .runner import TestResult


def _prompt_hash(prompt: str) -> str:
    """Short hash of a prompt for staleness detection."""
    return hashlib.sha256(prompt.strip().encode()).hexdigest()[:12]


def _model_filename(model: str) -> str:
    """Convert model name to a safe filename: 'gemma4:31b' → 'gemma4_31b'."""
    return model.replace(":", "_").replace("/", "_")


# ── Per-model result database ────────────────────────────────────────


def load_model_results(results_dir: Path, model: str) -> dict[str, dict]:
    """Load the result database for a single model.

    Returns dict mapping test_name → result entry.
    """
    path = results_dir / f"{_model_filename(model)}.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        return {entry["test"]: entry for entry in data.get("results", [])}
    except (json.JSONDecodeError, KeyError, OSError):
        return {}


def save_model_results(
    results: list[TestResult],
    model: str,
    output_dir: Path,
) -> Path:
    """Save/update the per-model result database. Returns the file path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{_model_filename(model)}.json"

    # Load existing results to preserve tests not in this run
    existing = {}
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            existing = {entry["test"]: entry for entry in data.get("results", [])}
        except (json.JSONDecodeError, KeyError, OSError):
            pass

    # Update with new results
    for r in results:
        entry = {
            "test": r.test.name,
            "category": r.test.category,
            "prompt": r.test.prompt.strip(),
            "prompt_hash": _prompt_hash(r.test.prompt),
            "eval_type": r.test.eval_type,
            "score": r.eval_result.score,
            "passed": r.eval_result.passed,
            "eval_details": r.eval_result.details,
            "total_time": round(r.model_response.total_time, 3),
            "ttft": round(r.model_response.time_to_first_token, 3),
            "prompt_tokens": r.model_response.prompt_tokens,
            "completion_tokens": r.model_response.completion_tokens,
            "tokens_per_sec": round(r.model_response.completion_tokens / r.model_response.total_time, 1) if r.model_response.total_time > 0 else 0,
            "response": r.model_response.response,
            "timestamp": datetime.now().isoformat(),
        }
        if r.model_response.thinking_tokens:
            entry["thinking_tokens"] = r.model_response.thinking_tokens
        if r.reused:
            entry["reused"] = True
        existing[r.test.name] = entry

    # Write back
    out = {
        "model": model,
        "updated": datetime.now().isoformat(),
        "results": list(existing.values()),
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    return path


def load_previous_results(
    results_dir: Path,
    models: list[str],
    tests: list,
) -> dict[tuple[str, str], dict]:
    """Load previous passing results for given models, checking prompt staleness.

    Returns dict mapping (model_name, test_name) → result entry,
    only for entries where:
      - the test previously passed
      - the prompt hash still matches (question hasn't been edited)
    """
    # Build current prompt hashes from test definitions
    current_hashes = {t.name: _prompt_hash(t.prompt) for t in tests}
    test_names = set(current_hashes.keys())

    cache: dict[tuple[str, str], dict] = {}

    for model in models:
        db = load_model_results(results_dir, model)
        for test_name, entry in db.items():
            if test_name not in test_names:
                continue
            if not entry.get("passed", False):
                continue
            # Check staleness — prompt edited since last run?
            stored_hash = entry.get("prompt_hash", "")
            if stored_hash and stored_hash != current_hashes[test_name]:
                continue  # stale — prompt changed, re-run
            cache[(model, test_name)] = entry

    # Also scan legacy timestamped files for backward compatibility
    for json_file in sorted(results_dir.glob("*.json"), reverse=True):
        stem = json_file.stem
        # Skip per-model DB files and judged files
        if "_judged" in stem or not stem[0].isdigit():
            continue
        try:
            with open(json_file) as f:
                data = json.load(f)
            for model_name, entries in data.get("models", {}).items():
                if model_name not in models:
                    continue
                for entry in entries:
                    key = (model_name, entry["test"])
                    if key in cache or entry["test"] not in test_names:
                        continue
                    if not entry.get("passed", False):
                        continue
                    # Legacy files don't have prompt_hash — accept them
                    cache[key] = entry
        except (json.JSONDecodeError, KeyError, OSError):
            continue

    return cache


# ── Console display ──────────────────────────────────────────────────


def print_summary(
    results: dict[str, list[TestResult]],
    console: Console | None = None,
) -> None:
    """Print a Rich summary table to the console."""
    if console is None:
        console = Console()

    models = list(results.keys())
    if not models:
        console.print("[red]No results to display.[/red]")
        return

    # Group tests by category
    categories: dict[str, list[str]] = {}
    for result in results[models[0]]:
        cat = result.test.category
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(result.test.name)

    # Summary table
    table = Table(
        title="Crucible Results",
        show_header=True,
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Category", style="bold")
    table.add_column("Test", style="dim")
    for model in models:
        table.add_column(model, justify="center")

    # Build lookup: (model, test_name) -> TestResult
    lookup: dict[tuple[str, str], TestResult] = {}
    for model, model_results in results.items():
        for r in model_results:
            lookup[(model, r.test.name)] = r

    for cat, test_names in categories.items():
        for i, test_name in enumerate(test_names):
            cat_label = cat.replace("_", " ").title() if i == 0 else ""
            row = [cat_label, test_name]

            for model in models:
                r = lookup.get((model, test_name))
                if r is None:
                    row.append("—")
                    continue

                score = r.eval_result.score
                passed = r.eval_result.passed
                time_s = r.model_response.total_time
                empty = not r.model_response.response.strip()
                timed_out = r.model_response.timed_out

                if timed_out:
                    cell = Text(f"⏱ timeout ({time_s:.1f}s)", style="red")
                elif empty:
                    cell = Text(f"✗ empty ({time_s:.1f}s)", style="red")
                elif r.test.eval_type == "manual":
                    cell = Text(f"review ({time_s:.1f}s)", style="yellow")
                elif passed and r.reused:
                    cell = Text(f"♻ {score:.0%} (reused)", style="green")
                elif passed:
                    cell = Text(f"✓ {score:.0%} ({time_s:.1f}s)", style="green")
                else:
                    cell = Text(f"✗ {score:.0%} ({time_s:.1f}s)", style="red")

                row.append(cell)

            table.add_row(*row)

    console.print()
    console.print(table)

    # Per-model summary
    console.print()
    summary_table = Table(
        title="Model Summary",
        show_header=True,
        header_style="bold cyan",
    )
    summary_table.add_column("Metric")
    for model in models:
        summary_table.add_column(model, justify="center")

    # Check if any model used thinking tokens
    any_thinking = any(
        r.model_response.thinking_tokens > 0
        for rs in results.values() for r in rs
    )

    # Helper: average generation speed (completion tokens / total time)
    def _avg_tks(rs: list) -> str:
        active = [r for r in rs if not r.reused and r.model_response.total_time > 0]
        if not active:
            return "—"
        total_toks = sum(r.model_response.completion_tokens for r in active)
        total_time = sum(r.model_response.total_time for r in active)
        return f"{total_toks / total_time:.1f}"

    # Compute aggregates
    metrics = [
        ("Pass rate", lambda rs: f"{sum(1 for r in rs if r.eval_result.passed) / len(rs):.0%}"),
        ("Avg score", lambda rs: f"{sum(r.eval_result.score for r in rs) / len(rs):.2f}"),
        ("Avg time", lambda rs: f"{sum(r.model_response.total_time for r in rs) / len(rs):.1f}s"),
        ("Avg TTFT", lambda rs: f"{sum(r.model_response.time_to_first_token for r in rs) / len(rs):.2f}s"),
        ("Avg tk/s", _avg_tks),
        ("Total tokens", lambda rs: f"{sum(r.model_response.total_tokens for r in rs):,}"),
    ]
    if any_thinking:
        metrics.append(
            ("Thinking tokens", lambda rs: f"{sum(r.model_response.thinking_tokens for r in rs):,}")
        )
    for metric_name, metric_fn in metrics:
        row = [metric_name]
        for model in models:
            row.append(metric_fn(results[model]))
        summary_table.add_row(*row)

    console.print(summary_table)


# ── File export ──────────────────────────────────────────────────────


def save_results(
    results: dict[str, list[TestResult]],
    output_dir: Path,
) -> list[Path]:
    """Save results to per-model JSON databases. Returns list of paths written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for model, model_results in results.items():
        path = save_model_results(model_results, model, output_dir)
        paths.append(path)
    return paths


def save_markdown(
    results: dict[str, list[TestResult]],
    output_dir: Path,
) -> Path:
    """Save a timestamped markdown report. Returns the path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    md_path = output_dir / f"{timestamp}.md"
    with open(md_path, "w") as f:
        f.write(_build_markdown(results, timestamp))
    return md_path


def _build_markdown(results: dict[str, list[TestResult]], timestamp: str) -> str:
    """Build markdown report."""
    lines = [
        f"# Crucible Report — {timestamp}",
        "",
    ]

    models = list(results.keys())

    # Summary table
    lines.append("## Summary")
    lines.append("")
    header = "| Test | " + " | ".join(models) + " |"
    sep = "|------|" + "|".join(["------"] * len(models)) + "|"
    lines.extend([header, sep])

    # Assume all models ran same tests in same order
    for i, r in enumerate(results[models[0]]):
        row = f"| {r.test.category}/{r.test.name} |"
        for model in models:
            mr = results[model][i]
            icon = "✓" if mr.eval_result.passed else "✗"
            row += f" {icon} {mr.eval_result.score:.0%} ({mr.model_response.total_time:.1f}s) |"
        lines.append(row)

    lines.append("")

    # Detailed results
    lines.append("## Detailed Results")
    lines.append("")

    for i, r0 in enumerate(results[models[0]]):
        lines.append(f"### {r0.test.category}/{r0.test.name}")
        lines.append("")
        lines.append(f"**Description:** {r0.test.description}")
        lines.append("")
        lines.append("**Prompt:**")
        lines.append("```")
        lines.append(r0.test.prompt.strip())
        lines.append("```")
        lines.append("")

        for model in models:
            mr = results[model][i]
            icon = "✓" if mr.eval_result.passed else "✗"
            lines.append(f"#### {model} — {icon} {mr.eval_result.score:.0%}")
            lines.append("")
            lines.append(f"- **Time:** {mr.model_response.total_time:.1f}s (TTFT: {mr.model_response.time_to_first_token:.2f}s)")
            lines.append(f"- **Tokens:** {mr.model_response.prompt_tokens} prompt + {mr.model_response.completion_tokens} completion")
            lines.append(f"- **Eval:** {mr.eval_result.details}")
            lines.append("")
            lines.append("<details>")
            lines.append("<summary>Full response</summary>")
            lines.append("")
            lines.append(mr.model_response.response)
            lines.append("")
            lines.append("</details>")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ── HTML visual report ──────────────────────────────────────────────


def generate_html_report(
    models: list[str],
    lookup: dict[tuple[str, str], dict],
    output_path: Path,
    title: str = "Crucible Benchmark Report",
    model_infos: dict | None = None,
) -> Path:
    """Generate a self-contained HTML report with Chart.js visualisations.

    Args:
        models: list of model names
        lookup: mapping (model, test_name) → result dict
        output_path: where to write the HTML file
        title: report title
        model_infos: optional dict of model_name → ModelInfo with params/quant/tier
    Returns:
        The output path
    """
    if model_infos is None:
        model_infos = {}
    # ── Gather all tests and categories ──
    all_tests: dict[str, str] = {}  # test_name → category
    for (model, test_name), entry in lookup.items():
        if test_name not in all_tests:
            all_tests[test_name] = entry.get("category", "unknown")

    # Sort tests by category then name
    sorted_tests = sorted(all_tests.items(), key=lambda x: (x[1], x[0]))
    test_names = [t[0] for t in sorted_tests]
    categories = sorted(set(all_tests.values()))

    # ── Compute data for charts ──

    # 1. Per-category average scores (radar chart)
    cat_scores: dict[str, dict[str, float]] = {m: {} for m in models}
    for model in models:
        for cat in categories:
            cat_tests = [t for t, c in all_tests.items() if c == cat]
            scores = [lookup[(model, t)]["score"] for t in cat_tests if (model, t) in lookup]
            cat_scores[model][cat] = sum(scores) / len(scores) if scores else 0

    # 2. Pass rates (bar chart)
    pass_rates: dict[str, float] = {}
    for model in models:
        entries = [lookup[(model, t)] for t in test_names if (model, t) in lookup]
        pass_rates[model] = sum(1 for e in entries if e.get("passed")) / len(entries) if entries else 0

    # 3. Per-test scores (grouped bar chart)
    test_scores: dict[str, dict[str, float | None]] = {}
    for t in test_names:
        test_scores[t] = {}
        for model in models:
            entry = lookup.get((model, t))
            test_scores[t][model] = entry["score"] if entry else None

    # 4. Performance: avg tk/s and avg TTFT
    perf_tks: dict[str, float] = {}
    perf_ttft: dict[str, float] = {}
    for model in models:
        entries = [lookup[(model, t)] for t in test_names
                   if (model, t) in lookup and not lookup[(model, t)].get("reused")]
        if entries:
            total_comp = sum(e.get("completion_tokens", 0) for e in entries)
            total_time = sum(e.get("total_time", 0) for e in entries)
            perf_tks[model] = round(total_comp / total_time, 1) if total_time > 0 else 0
            perf_ttft[model] = round(sum(e.get("ttft", 0) for e in entries) / len(entries), 2)
        else:
            perf_tks[model] = 0
            perf_ttft[model] = 0

    # 5. Summary stats
    total_tests = len(test_names)
    summary: dict[str, dict] = {}
    for model in models:
        entries = [lookup[(model, t)] for t in test_names if (model, t) in lookup]
        n = len(entries)
        if n == 0:
            continue
        summary[model] = {
            "tests": n,
            "total_tests": total_tests,
            "passed": sum(1 for e in entries if e.get("passed")),
            "avg_score": round(sum(e["score"] for e in entries) / n, 3),
            "avg_time": round(sum(e.get("total_time", 0) for e in entries) / n, 1),
            "total_tokens": sum(e.get("prompt_tokens", 0) + e.get("completion_tokens", 0) for e in entries),
            "tk_s": perf_tks.get(model, 0),
            "complete": n == total_tests,
        }

    # ── Colours for models ──
    palette = ["#4dc9f6", "#f67019", "#f53794", "#537bc4", "#acc236",
               "#166a8f", "#00a950", "#8549ba", "#e8c3b9", "#c45850"]
    model_colours = {m: palette[i % len(palette)] for i, m in enumerate(models)}

    # ── Build the JSON data blob for JS ──
    chart_data = json.dumps({
        "models": models,
        "categories": [c.replace("_", " ").title() for c in categories],
        "categoriesRaw": categories,
        "testNames": test_names,
        "testLabels": [t.replace("_", " ") for t in test_names],
        "testCategories": [all_tests[t].replace("_", " ").title() for t in test_names],
        "catScores": cat_scores,
        "passRates": pass_rates,
        "testScores": test_scores,
        "perfTks": perf_tks,
        "perfTtft": perf_ttft,
        "summary": summary,
        "colours": model_colours,
        "modelMeta": {m: {
            "params": mi.parameter_size,
            "quant": mi.quantization,
            "family": mi.family,
            "vram_gb": mi.vram_gb,
            "tier": mi.vram_tier,
        } for m, mi in model_infos.items()},
        "lookup": {f"{m}|{t}": {
            "score": e["score"],
            "passed": e.get("passed", False),
            "time": e.get("total_time", 0),
            "tks": e.get("tokens_per_sec", 0),
            "ttft": e.get("ttft", 0),
        } for (m, t), e in lookup.items()},
    })

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = _HTML_TEMPLATE.replace("{{TITLE}}", title)
    html = html.replace("{{TIMESTAMP}}", timestamp)
    html = html.replace("{{N_MODELS}}", str(len(models)))
    html = html.replace("{{N_TESTS}}", str(len(test_names)))
    html = html.replace("{{CHART_DATA}}", chart_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    return output_path


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{TITLE}}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  :root {
    --bg: #0f0f1a;
    --card: #1a1a2e;
    --card-border: #2a2a4a;
    --text: #e8e8f0;
    --text-dim: #8888aa;
    --accent: #4dc9f6;
    --green: #4caf50;
    --red: #ef5350;
    --yellow: #ffc107;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    padding: 24px;
    max-width: 1400px;
    margin: 0 auto;
  }
  header {
    text-align: center;
    margin-bottom: 32px;
    padding: 32px 0;
    border-bottom: 1px solid var(--card-border);
  }
  header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent), #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
  }
  header .subtitle {
    color: var(--text-dim);
    font-size: 0.95rem;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
    gap: 20px;
    margin-bottom: 20px;
  }
  .card {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 24px;
  }
  .card h2 {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-dim);
    margin-bottom: 16px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-size: 0.85rem;
  }
  .card.wide { grid-column: 1 / -1; }
  .card.hero { grid-column: 1 / -1; }
  .hero-inner {
    display: flex;
    gap: 32px;
    align-items: flex-start;
  }
  .hero-inner .chart-wrap {
    flex: 0 0 480px;
    max-height: 420px;
  }
  .summary-cards {
    flex: 1;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    align-content: start;
  }
  .model-card {
    background: var(--bg);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    border-top: 3px solid var(--accent);
  }
  .model-card .name {
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 8px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .model-card .stat {
    color: var(--text-dim);
    font-size: 0.8rem;
    margin: 4px 0;
  }
  .model-card .big {
    font-size: 1.6rem;
    font-weight: 700;
    margin: 4px 0;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
  }
  th {
    background: var(--bg);
    color: var(--text-dim);
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.5px;
    padding: 10px 12px;
    text-align: left;
    position: sticky;
    top: 0;
  }
  td {
    padding: 8px 12px;
    border-top: 1px solid var(--card-border);
  }
  tr:hover td { background: rgba(77, 201, 246, 0.05); }
  .pass { color: var(--green); font-weight: 600; }
  .fail { color: var(--red); font-weight: 600; }
  .score-cell { text-align: center; }
  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
  }
  .badge-pass { background: rgba(76, 175, 80, 0.15); color: var(--green); }
  .badge-fail { background: rgba(239, 83, 80, 0.15); color: var(--red); }
  .badge-tier {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
    background: rgba(77, 201, 246, 0.12);
    color: var(--accent);
    letter-spacing: 0.3px;
  }
  footer {
    text-align: center;
    color: var(--text-dim);
    font-size: 0.8rem;
    margin-top: 32px;
    padding-top: 16px;
    border-top: 1px solid var(--card-border);
  }
  footer a { color: var(--accent); text-decoration: none; }
  .filter-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    margin-bottom: 24px;
    padding: 16px 20px;
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 12px;
  }
  .filter-bar .label {
    color: var(--text-dim);
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-right: 8px;
  }
  .filter-btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 6px;
    border: 1px solid var(--card-border);
    background: var(--bg);
    color: var(--text);
    font-size: 0.82rem;
    cursor: pointer;
    transition: all 0.15s;
    user-select: none;
  }
  .filter-btn:hover { border-color: var(--accent); }
  .filter-btn.active {
    border-color: var(--accent);
    background: rgba(77, 201, 246, 0.1);
  }
  .filter-btn .dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
  }
  .filter-btn .tier-tag {
    font-size: 0.65rem;
    color: var(--text-dim);
    font-weight: 600;
  }
  .filter-tier-btns {
    margin-left: auto;
    display: flex;
    gap: 6px;
  }
  .tier-btn {
    padding: 4px 10px;
    border-radius: 4px;
    border: 1px solid var(--card-border);
    background: var(--bg);
    color: var(--text-dim);
    font-size: 0.72rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
  }
  .tier-btn:hover { border-color: var(--accent); color: var(--text); }
  .tier-btn.active { border-color: var(--accent); background: rgba(77, 201, 246, 0.1); color: var(--accent); }
  @media (max-width: 600px) {
    .grid { grid-template-columns: 1fr; }
    .hero-inner { flex-direction: column; }
    .hero-inner .chart-wrap { flex: 1; max-width: 100%; }
    body { padding: 12px; }
  }
</style>
</head>
<body>

<header>
  <h1>{{TITLE}}</h1>
  <p class="subtitle">Generated {{TIMESTAMP}} &middot; {{N_MODELS}} models &middot; {{N_TESTS}} tests</p>
</header>

<div class="filter-bar" id="filterBar">
  <span class="label">Models</span>
  <div id="modelFilters"></div>
  <div class="filter-tier-btns" id="tierFilters"></div>
</div>

<!-- Hero: radar + model summary cards -->
<div class="card hero">
  <h2>Category Performance</h2>
  <div class="hero-inner">
    <div class="chart-wrap">
      <canvas id="radarChart"></canvas>
    </div>
    <div class="summary-cards" id="summaryCards"></div>
  </div>
</div>

<div class="grid">
  <div class="card">
    <h2>Pass Rate by Model</h2>
    <canvas id="passRateChart"></canvas>
  </div>
  <div class="card">
    <h2>Generation Speed (tk/s)</h2>
    <canvas id="perfChart"></canvas>
  </div>
</div>

<div class="card wide">
  <h2>Per-Test Scores</h2>
  <canvas id="testScoresChart" style="max-height:400px;"></canvas>
</div>

<div class="card wide">
  <h2>Full Results</h2>
  <div style="overflow-x:auto;">
    <table id="resultsTable"></table>
  </div>
</div>

<footer>
  Crucible &mdash; LLM benchmark for structural engineering academics &middot;
  <a href="https://github.com/ccaprani/crucible">github.com/ccaprani/crucible</a>
</footer>

<script>
const D = {{CHART_DATA}};

// Chart.js global defaults
Chart.defaults.color = '#8888aa';
Chart.defaults.borderColor = '#2a2a4a';
Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";

// Inline plugin: draw value labels to the right of horizontal bars
const barLabelPlugin = {
  id: 'barLabels',
  afterDatasetsDraw(chart) {
    const ctx = chart.ctx;
    chart.data.datasets.forEach((ds, di) => {
      const meta = chart.getDatasetMeta(di);
      meta.data.forEach((bar, i) => {
        const val = ds.data[i];
        if (val == null) return;
        const label = ds._labelFn ? ds._labelFn(val) : val;
        ctx.save();
        ctx.fillStyle = '#e8e8f0';
        ctx.font = 'bold 12px -apple-system, BlinkMacSystemFont, sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, bar.x + 8, bar.y);
        ctx.restore();
      });
    });
  }
};

// ── Filter state ──
let activeModels = new Set(D.models);
let charts = {};

// ── Build filter bar ──
const filterEl = document.getElementById('modelFilters');
const tierEl = document.getElementById('tierFilters');

// Get unique tiers
const tiers = [...new Set(D.models.map(m => (D.modelMeta[m] || {}).tier).filter(Boolean))];
const tierOrder = {'8GB': 0, '16GB': 1, '24GB': 2};
tiers.sort((a, b) => (tierOrder[a] || 0) - (tierOrder[b] || 0));

// Model toggle buttons
D.models.forEach(m => {
  const btn = document.createElement('button');
  btn.className = 'filter-btn active';
  const meta = D.modelMeta[m];
  const tierTag = meta ? `<span class="tier-tag">${meta.tier}</span>` : '';
  btn.innerHTML = `<span class="dot" style="background:${D.colours[m]}"></span>${m} ${tierTag}`;
  btn.onclick = () => {
    if (activeModels.has(m)) {
      if (activeModels.size <= 1) return; // keep at least one
      activeModels.delete(m);
      btn.classList.remove('active');
    } else {
      activeModels.add(m);
      btn.classList.add('active');
    }
    rebuild();
  };
  filterEl.appendChild(btn);
});

// Tier quick-select buttons
tiers.forEach(tier => {
  const btn = document.createElement('button');
  btn.className = 'tier-btn';
  btn.textContent = tier;
  btn.onclick = () => {
    const tierModels = D.models.filter(m => (D.modelMeta[m] || {}).tier === tier);
    // If all tier models are active and only them, show all instead (toggle off)
    const allActive = tierModels.every(m => activeModels.has(m));
    const onlyThem = activeModels.size === tierModels.length;
    if (allActive && onlyThem) {
      // Reset to all
      activeModels = new Set(D.models);
    } else {
      activeModels = new Set(tierModels);
    }
    // Update model button states
    document.querySelectorAll('#modelFilters .filter-btn').forEach((b, i) => {
      b.classList.toggle('active', activeModels.has(D.models[i]));
    });
    // Update tier button states
    document.querySelectorAll('#tierFilters .tier-btn').forEach(b => {
      b.classList.toggle('active', b.textContent === tier && !allActive);
    });
    rebuild();
  };
  tierEl.appendChild(btn);
});

// All button
const allBtn = document.createElement('button');
allBtn.className = 'tier-btn active';
allBtn.textContent = 'All';
allBtn.onclick = () => {
  activeModels = new Set(D.models);
  document.querySelectorAll('#modelFilters .filter-btn').forEach(b => b.classList.add('active'));
  document.querySelectorAll('#tierFilters .tier-btn').forEach(b => b.classList.remove('active'));
  allBtn.classList.add('active');
  rebuild();
};
tierEl.insertBefore(allBtn, tierEl.firstChild);

function rebuild() {
  const models = D.models.filter(m => activeModels.has(m));

  // Update All button state
  allBtn.classList.toggle('active', models.length === D.models.length);

  // ── Summary cards ──
  const cardsEl = document.getElementById('summaryCards');
  cardsEl.innerHTML = '';
  models.forEach(m => {
    const s = D.summary[m];
    if (!s) return;
    const col = D.colours[m];
    const pct = Math.round(s.passed / s.tests * 100);
    const meta = D.modelMeta[m];
    const metaLine = meta
      ? `<div class="stat">${meta.params} &middot; ${meta.quant} &middot; ~${meta.vram_gb}GB</div>
         <div class="stat badge-tier" style="margin-top:4px">${meta.tier} VRAM tier</div>`
      : '';
    cardsEl.innerHTML += `
      <div class="model-card" style="border-top-color:${col}">
        <div class="name">${m}</div>
        ${metaLine}
        <div class="big" style="color:${col}">${pct}%</div>
        <div class="stat">${s.complete ? '' : `<span style="color:var(--yellow);font-weight:600">&#9888; ${s.tests}/${s.total_tests} tests run</span><br>`}pass rate: ${s.passed}/${s.tests}</div>
        <div class="stat">avg score: ${(s.avg_score * 100).toFixed(0)}%</div>
        <div class="stat">${s.tk_s} tk/s &middot; ${s.avg_time}s avg</div>
        <div class="stat">${s.total_tokens.toLocaleString()} tokens</div>
      </div>`;
  });

  // ── Radar chart ──
  if (charts.radar) charts.radar.destroy();
  charts.radar = new Chart(document.getElementById('radarChart'), {
    type: 'radar',
    data: {
      labels: D.categories,
      datasets: models.map(m => ({
        label: m,
        data: D.categories.map((_, i) => D.catScores[m][D.categoriesRaw[i]]),
        borderColor: D.colours[m],
        backgroundColor: D.colours[m] + '22',
        borderWidth: 2,
        pointRadius: 4,
        pointBackgroundColor: D.colours[m],
      }))
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      scales: {
        r: {
          min: 0, max: 1,
          ticks: { stepSize: 0.2, backdropColor: 'transparent' },
          grid: { color: '#2a2a4a' },
          angleLines: { color: '#2a2a4a' },
          pointLabels: { font: { size: 13 }, color: '#e8e8f0' },
        }
      },
      plugins: {
        legend: { position: 'bottom', labels: { padding: 20, usePointStyle: true } },
        tooltip: {
          callbacks: {
            label: ctx => `${ctx.dataset.label}: ${(ctx.parsed.r * 100).toFixed(0)}%`
          }
        }
      }
    }
  });

  // ── Pass rate bar chart ──
  if (charts.passRate) charts.passRate.destroy();
  charts.passRate = new Chart(document.getElementById('passRateChart'), {
    type: 'bar',
    data: {
      labels: models,
      datasets: [{
        data: models.map(m => D.passRates[m]),
        backgroundColor: models.map(m => D.colours[m] + 'cc'),
        borderColor: models.map(m => D.colours[m]),
        borderWidth: 1,
        borderRadius: 6,
        _labelFn: v => (v * 100).toFixed(0) + '%',
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      layout: { padding: { right: 50 } },
      plugins: { legend: { display: false },
        tooltip: { callbacks: { label: ctx => (ctx.parsed.x * 100).toFixed(0) + '%' } }
      },
      scales: {
        x: { min: 0, max: 1, ticks: { callback: v => (v * 100) + '%' } },
        y: { ticks: { font: { size: 12 } } }
      }
    },
    plugins: [barLabelPlugin]
  });

  // ── Performance chart ──
  if (charts.perf) charts.perf.destroy();
  charts.perf = new Chart(document.getElementById('perfChart'), {
    type: 'bar',
    data: {
      labels: models,
      datasets: [{
        label: 'tk/s',
        data: models.map(m => D.perfTks[m]),
        backgroundColor: models.map(m => D.colours[m] + 'cc'),
        borderColor: models.map(m => D.colours[m]),
        borderWidth: 1,
        borderRadius: 6,
        _labelFn: v => v.toFixed(1) + ' tk/s',
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      layout: { padding: { right: 70 } },
      plugins: { legend: { display: false },
        tooltip: { callbacks: { label: ctx => ctx.parsed.x.toFixed(1) + ' tk/s' } }
      },
      scales: {
        x: { beginAtZero: true, title: { display: true, text: 'tokens / second' } },
        y: { ticks: { font: { size: 12 } } }
      }
    },
    plugins: [barLabelPlugin]
  });

  // ── Per-test grouped bar chart ──
  if (charts.testScores) charts.testScores.destroy();
  charts.testScores = new Chart(document.getElementById('testScoresChart'), {
    type: 'bar',
    data: {
      labels: D.testLabels,
      datasets: models.map(m => ({
        label: m,
        data: D.testNames.map(t => D.testScores[t][m]),
        backgroundColor: D.colours[m] + 'cc',
        borderColor: D.colours[m],
        borderWidth: 1,
        borderRadius: 4,
      }))
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'bottom', labels: { usePointStyle: true, padding: 16 } },
        tooltip: {
          callbacks: {
            title: items => D.testCategories[items[0].dataIndex] + ' / ' + items[0].label,
            label: ctx => ctx.dataset.label + ': ' + (ctx.parsed.y != null ? (ctx.parsed.y * 100).toFixed(0) + '%' : 'N/A'),
          }
        }
      },
      scales: {
        x: { ticks: { maxRotation: 45, minRotation: 30, font: { size: 11 } } },
        y: { min: 0, max: 1, ticks: { callback: v => (v * 100) + '%' } },
      }
    }
  });

  // ── Results table ──
  const tbl = document.getElementById('resultsTable');
  let thead = '<thead><tr><th>Category</th><th>Test</th>';
  models.forEach(m => { thead += `<th class="score-cell">${m}</th>`; });
  thead += '</tr></thead>';

  let tbody = '<tbody>';
  let lastCat = '';
  D.testNames.forEach((t, i) => {
    const cat = D.testCategories[i];
    const catLabel = cat !== lastCat ? cat : '';
    lastCat = cat;
    tbody += `<tr><td>${catLabel}</td><td>${D.testLabels[i]}</td>`;
    models.forEach(m => {
      const key = m + '|' + t;
      const r = D.lookup[key];
      if (!r) { tbody += '<td class="score-cell">&mdash;</td>'; return; }
      const cls = r.passed ? 'badge-pass' : 'badge-fail';
      const icon = r.passed ? '✓' : '✗';
      tbody += `<td class="score-cell"><span class="badge ${cls}">${icon} ${(r.score * 100).toFixed(0)}%</span> <span style="color:#8888aa;font-size:0.75rem">${r.time.toFixed(1)}s</span></td>`;
    });
    tbody += '</tr>';
  });
  tbody += '</tbody>';
  tbl.innerHTML = thead + tbody;
}

// Initial render
rebuild();
</script>
</body>
</html>"""
