"""Post-hoc LLM judging — score previously captured responses for llm_judge tests."""

import json
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.table import Table

from .evaluate import _eval_llm_judge, EvalResult
from .runner import load_tests
from .report import result_test_id

TESTS_DIR = Path(__file__).resolve().parent.parent / "tests"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def judge_results(
    results_dir: Path | None = None,
    tests_dir: Path | None = None,
    console: Console | None = None,
    model_names: list[str] | None = None,
    suite_names: list[str] | None = None,
    test_names: list[str] | None = None,
) -> list[Path]:
    """Score llm_judge tests across all per-model result files.

    Reads stored responses, sends them through the judge cascade
    (Claude CLI → OpenRouter → Anthropic API), and writes scores
    back into the per-model JSON files.

    Args:
        results_dir: Path to results directory (default: results/)
        model_names: Optional list of model name substrings to filter
        test_names: Optional list of test name substrings to filter

    Returns:
        List of updated file paths.
    """
    if console is None:
        console = Console()
    if results_dir is None:
        results_dir = RESULTS_DIR
    if tests_dir is None:
        tests_dir = TESTS_DIR

    # Load test definitions to find llm_judge tests and their criteria
    all_tests = load_tests(tests_dir)
    judge_tests = {t.test_id: t for t in all_tests if t.eval_type == "llm_judge"}

    if not judge_tests:
        console.print("[yellow]No llm_judge tests found in test definitions.[/yellow]")
        return []

    # Apply test name filter
    if test_names:
        judge_tests = {
            test_id: t for test_id, t in judge_tests.items()
            if any(n.lower() in test_id.lower() for n in test_names)
        }

    if suite_names:
        judge_tests = {
            test_id: t for test_id, t in judge_tests.items()
            if t.suite in suite_names
        }

    if not judge_tests:
        console.print("[yellow]No matching llm_judge tests.[/yellow]")
        return []

    # Find all per-model result files
    model_files = sorted(
        results_dir.glob("*.json")
    )

    if not model_files:
        console.print("[red]No result files found.[/red]")
        return []

    # Build list of (file_path, model_name, test_name, result_entry, test_def) to judge
    to_judge: list[tuple[Path, str, str, dict, object]] = []

    for model_file in model_files:
        try:
            with open(model_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        model_name = data.get("model", model_file.stem)

        # Filter by model name if specified
        if model_names:
            if not any(n.lower() in model_name.lower() for n in model_names):
                continue

        for entry in data.get("results", []):
            test_id = result_test_id(entry)
            if test_id not in judge_tests:
                continue

            # Skip if already judged (has a real judge score, not 0.5 deferred)
            eval_details = entry.get("eval_details", "")
            if "Judge" in eval_details and entry.get("score", 0.5) != 0.5:
                continue

            to_judge.append((model_file, model_name, test_id, entry, judge_tests[test_id]))

    if not to_judge:
        console.print("[yellow]No responses need judging (all already scored).[/yellow]")
        return []

    console.print(
        f"\n[bold]Judging {len(to_judge)} responses across "
        f"{len(set(m for _, m, _, _, _ in to_judge))} models[/bold]"
    )
    test_labels = [n.replace("_", " ").title() for n in judge_tests]
    console.print(f"[dim]Tests: {', '.join(test_labels)}[/dim]\n")

    completed = 0
    result_lines: list[str] = []
    updated_files: set[Path] = set()

    def build_table() -> Table:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("label", style="dim", width=12)
        table.add_column("value")
        table.add_row("Progress", f"[bold]{completed}/{len(to_judge)}[/bold]")
        for line in result_lines[-8:]:
            table.add_row("", line)
        return table

    with Live(build_table(), console=console, refresh_per_second=1) as live:
        for model_file, model_name, test_name, entry, test_def in to_judge:
            eval_result = _eval_llm_judge(
                entry.get("response", ""),
                test_def.eval_config,
                test_def.prompt,
            )

            # Update the entry in-place
            entry["score"] = eval_result.score
            entry["passed"] = eval_result.passed
            entry["eval_details"] = eval_result.details

            updated_files.add(model_file)
            completed += 1

            icon = "✓" if eval_result.passed else "✗"
            style = "green" if eval_result.passed else "red"
            result_lines.append(
                f"[{style}]{icon}[/{style}]  {model_name} / {test_name} — "
                f"{eval_result.score:.0%} [dim]{eval_result.details[:60]}[/dim]"
            )
            live.update(build_table())

    # Write updated files back
    updated_paths = []
    for model_file in updated_files:
        with open(model_file) as f:
            data = json.load(f)

        # Rebuild results list with updated entries
        result_lookup = {}
        for item in to_judge:
            if item[0] == model_file:
                result_lookup[item[2]] = item[3]  # test_id → updated entry

        for i, entry in enumerate(data.get("results", [])):
            test_id = result_test_id(entry)
            if test_id in result_lookup:
                data["results"][i] = result_lookup[test_id]

        with open(model_file, "w") as f:
            json.dump(data, f, indent=2)
        updated_paths.append(model_file)

    console.print(f"\n[bold green]Done.[/bold green] Updated {len(updated_paths)} files.")
    for p in updated_paths:
        console.print(f"  [dim]{p}[/dim]")

    _print_judge_summary(results_dir, judge_tests, console)
    return updated_paths


_TIER_ORDER = {"24GB": 0, "16GB": 1, "8GB": 2, "API": 3}


def _model_meta(model_name: str, results_dir: Path) -> tuple[str, str, float]:
    """Return (vram_tier, quantization, avg_tks) for a model."""
    from .models import is_api_model

    if is_api_model(model_name):
        tier = "API"
        quant = ""
    else:
        try:
            from .models import get_model_info
            info = get_model_info(model_name)
            tier = info.vram_tier
            quant = info.quantization if info.quantization != "?" else ""
        except Exception:
            tier = "8GB"
            quant = ""

    avg_tks = 0.0
    path = results_dir / f"{model_name.replace(':', '_').replace('/', '_')}.json"
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            tks_vals = [r.get("tokens_per_sec", 0) for r in data.get("results", [])
                        if r.get("tokens_per_sec", 0) > 0]
            if tks_vals:
                avg_tks = sum(tks_vals) / len(tks_vals)
        except (json.JSONDecodeError, OSError):
            pass

    return (tier, quant, avg_tks)


def _model_sort_key(model_name: str, results_dir: Path) -> tuple[int, float]:
    """Sort key: VRAM tier descending (24GB first), then avg tk/s descending."""
    tier, _quant, avg_tks = _model_meta(model_name, results_dir)
    return (_TIER_ORDER.get(tier, 2), -avg_tks)


def _print_judge_summary(results_dir: Path, judge_tests: dict, console: Console):
    """Print a summary table of judge scores across all models."""
    model_files = sorted(
        results_dir.glob("*.json")
    )

    models: list[str] = []
    scores: dict[str, dict[str, tuple[float, bool]]] = {}  # model → {test → (score, passed)}

    for model_file in model_files:
        try:
            with open(model_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        model_name = data.get("model", model_file.stem)
        models.append(model_name)
        scores[model_name] = {}

        for entry in data.get("results", []):
            test_id = result_test_id(entry)
            if test_id in judge_tests:
                scores[model_name][test_id] = (
                    entry.get("score", 0),
                    entry.get("passed", False),
                )

    if not models:
        return

    # Sort by VRAM tier then avg tk/s
    models.sort(key=lambda m: _model_sort_key(m, results_dir))

    # Pre-compute metadata for all models
    meta = {m: _model_meta(m, results_dir) for m in models}

    table = Table(
        title="LLM Judge Scores",
        show_header=True,
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("VRAM", style="dim", width=5)
    table.add_column("Model", style="bold")
    table.add_column("Q", style="dim", width=7)
    table.add_column("tk/s", justify="right", width=5)
    for test_name in judge_tests:
        label = test_name.replace("_", " ").title()
        table.add_column(label, justify="center")

    prev_tier = None
    for model in models:
        tier, quant, avg_tks = meta[model]

        # Section break between tiers
        if prev_tier is not None and tier != prev_tier:
            table.add_section()

        # Only show tier label on first row of each group
        tier_cell = f"[bold]{tier}[/bold]" if tier != prev_tier else ""
        tks_cell = f"{avg_tks:.0f}" if avg_tks > 0 else "—"

        row = [tier_cell, model, quant, tks_cell]
        for test_name in judge_tests:
            if test_name in scores.get(model, {}):
                score, passed = scores[model][test_name]
                style = "green" if passed else "red"
                row.append(f"[{style}]{score:.0%}[/{style}]")
            else:
                row.append("—")
        table.add_row(*row)
        prev_tier = tier

    console.print()
    console.print(table)


def review_judge_results(
    results_dir: Path | None = None,
    tests_dir: Path | None = None,
    console: Console | None = None,
    model_names: list[str] | None = None,
    suite_names: list[str] | None = None,
    test_names: list[str] | None = None,
) -> None:
    """Display the judge's reasoning for scored llm_judge tests."""
    from rich.panel import Panel
    from rich.text import Text

    if console is None:
        console = Console()
    if results_dir is None:
        results_dir = RESULTS_DIR
    if tests_dir is None:
        tests_dir = TESTS_DIR

    all_tests = load_tests(tests_dir)
    judge_tests = {t.test_id: t for t in all_tests if t.eval_type == "llm_judge"}

    if test_names:
        judge_tests = {
            test_id: t for test_id, t in judge_tests.items()
            if any(n.lower() in test_id.lower() for n in test_names)
        }

    if suite_names:
        judge_tests = {
            test_id: t for test_id, t in judge_tests.items()
            if t.suite in suite_names
        }

    if not judge_tests:
        console.print("[yellow]No matching llm_judge tests.[/yellow]")
        return

    model_files = sorted(
        results_dir.glob("*.json")
    )

    # Collect all entries: (model, test_id, score, passed, eval_details, response)
    entries: list[tuple[str, str, float, bool, str, str]] = []

    for model_file in model_files:
        try:
            with open(model_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        model_name = data.get("model", model_file.stem)

        if model_names:
            if not any(n.lower() in model_name.lower() for n in model_names):
                continue

        for entry in data.get("results", []):
            test_id = result_test_id(entry)
            if test_id not in judge_tests:
                continue
            eval_details = entry.get("eval_details", "")
            if not eval_details:
                continue
            entries.append((
                model_name,
                test_id,
                entry.get("score", 0),
                entry.get("passed", False),
                eval_details,
                entry.get("response", ""),
            ))

    if not entries:
        console.print("[yellow]No judged results found.[/yellow]")
        return

    # Group by test, then show each model's response + judge reasoning
    from collections import defaultdict
    by_test: dict[str, list] = defaultdict(list)
    for model, test_id, score, passed, details, response in entries:
        by_test[test_id].append((model, score, passed, details, response))

    # Sort models within each test by VRAM tier / tk/s
    for test_id in by_test:
        by_test[test_id].sort(key=lambda x: _model_sort_key(x[0], results_dir))

    for test_id in judge_tests:
        if test_id not in by_test:
            continue

        console.print(f"\n[bold cyan]━━━ {test_id} ━━━[/bold cyan]")
        desc = judge_tests[test_id].description
        if desc:
            console.print(f"[dim]{desc}[/dim]")

        for model, score, passed, details, response in by_test[test_id]:
            style = "green" if passed else "red"
            header = f"{model}  [{style}]{score:.0%}[/{style}]"

            # Truncate response for readability
            resp_lines = response.strip().splitlines()
            if len(resp_lines) > 12:
                resp_preview = "\n".join(resp_lines[:10]) + f"\n[dim]… ({len(resp_lines) - 10} more lines)[/dim]"
            else:
                resp_preview = response.strip()

            # The judge reasoning is in eval_details
            body = f"{resp_preview}\n\n[bold]Judge:[/bold] {details}"
            console.print(Panel(body, title=header, border_style=style, padding=(1, 2)))
