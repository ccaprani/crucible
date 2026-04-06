"""Report generation — Rich console tables and markdown/JSON file export."""

import json
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

from .runner import TestResult


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

                if r.test.eval_type == "manual":
                    cell = Text(f"review ({time_s:.1f}s)", style="yellow")
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

    # Compute aggregates
    for metric_name, metric_fn in [
        ("Pass rate", lambda rs: f"{sum(1 for r in rs if r.eval_result.passed) / len(rs):.0%}"),
        ("Avg score", lambda rs: f"{sum(r.eval_result.score for r in rs) / len(rs):.2f}"),
        ("Avg time", lambda rs: f"{sum(r.model_response.total_time for r in rs) / len(rs):.1f}s"),
        ("Avg TTFT", lambda rs: f"{sum(r.model_response.time_to_first_token for r in rs) / len(rs):.2f}s"),
        ("Total tokens", lambda rs: f"{sum(r.model_response.total_tokens for r in rs):,}"),
    ]:
        row = [metric_name]
        for model in models:
            row.append(metric_fn(results[model]))
        summary_table.add_row(*row)

    console.print(summary_table)


def save_results(
    results: dict[str, list[TestResult]],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Save results as both markdown and JSON. Returns (md_path, json_path)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    md_path = output_dir / f"{timestamp}.md"
    json_path = output_dir / f"{timestamp}.json"

    # Build JSON-serializable structure
    json_data = _build_json(results)

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    with open(md_path, "w") as f:
        f.write(_build_markdown(results, timestamp))

    return md_path, json_path


def _build_json(results: dict[str, list[TestResult]]) -> dict:
    """Build JSON-serializable results dict."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "models": {},
    }

    for model, model_results in results.items():
        model_data = []
        for r in model_results:
            model_data.append({
                "test": r.test.name,
                "category": r.test.category,
                "eval_type": r.test.eval_type,
                "score": r.eval_result.score,
                "passed": r.eval_result.passed,
                "eval_details": r.eval_result.details,
                "total_time": round(r.model_response.total_time, 3),
                "ttft": round(r.model_response.time_to_first_token, 3),
                "prompt_tokens": r.model_response.prompt_tokens,
                "completion_tokens": r.model_response.completion_tokens,
                "response": r.model_response.response,
            })
        data["models"][model] = model_data

    return data


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
