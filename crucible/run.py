#!/usr/bin/env python3
"""CLI entry point for Crucible."""

import argparse
import json
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.columns import Columns
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from crucible.models import list_available_models
from crucible.report import (
    load_model_results, load_previous_results,
    print_summary, save_markdown, save_results,
)
from crucible.runner import load_tests, run_tests

_PKG_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = _PKG_ROOT / "tests"
RESULTS_DIR = _PKG_ROOT / "results"


def _pick_models(console: Console) -> list[str]:
    """Interactive model picker."""
    available = list_available_models()
    available = [m for m in available if "embed" not in m.lower()]

    if not available:
        console.print("[red]No Ollama models found.[/red]")
        sys.exit(1)

    console.print("\n[bold]Available models:[/bold]")
    for i, name in enumerate(available, 1):
        console.print(f"  [cyan]{i:2d}[/cyan]  {name}")

    console.print(
        "\n[dim]Enter model numbers separated by spaces (e.g. 1 3 5)[/dim]"
    )
    raw = Prompt.ask("Models")

    selected = []
    for token in raw.split():
        token = token.strip().rstrip(",")
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(available):
                selected.append(available[idx])
            else:
                console.print(f"[yellow]Skipping invalid index: {token}[/yellow]")
        elif token in available:
            selected.append(token)
        else:
            console.print(f"[yellow]Unknown model: {token}[/yellow]")

    if not selected:
        console.print("[red]No models selected.[/red]")
        sys.exit(1)

    seen = set()
    deduped = []
    for m in selected:
        if m not in seen:
            seen.add(m)
            deduped.append(m)

    console.print(f"\n[bold]Selected:[/bold] {', '.join(deduped)}")
    return deduped


def _pick_categories(console: Console) -> list[str] | None:
    """Interactive category picker. Returns None for all."""
    tests = load_tests(TESTS_DIR)
    categories = list(dict.fromkeys(t.category for t in tests))

    console.print("\n[bold]Test categories:[/bold]")
    console.print(f"  [cyan] 0[/cyan]  [bold]all[/bold] ({len(tests)} tests)")
    for i, cat in enumerate(categories, 1):
        count = sum(1 for t in tests if t.category == cat)
        console.print(f"  [cyan]{i:2d}[/cyan]  {cat} ({count} tests)")

    console.print(
        "\n[dim]Enter category numbers separated by spaces, or 0 for all[/dim]"
    )
    raw = Prompt.ask("Categories", default="0")

    if raw.strip() == "0":
        return None

    selected = []
    for token in raw.split():
        token = token.strip().rstrip(",")
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(categories):
                selected.append(categories[idx])
        elif token in categories:
            selected.append(token)

    return selected or None


def _format_time(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _resolve_models(raw: list[str], console: Console) -> list[str]:
    """Resolve model references — accepts numbers, names, or substrings."""
    available = list_available_models()
    available_no_embed = [m for m in available if "embed" not in m.lower()]

    resolved = []
    for token in raw:
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(available_no_embed):
                resolved.append(available_no_embed[idx])
            else:
                console.print(f"[yellow]Invalid index {token} (max {len(available_no_embed)})[/yellow]")
        elif token in available:
            resolved.append(token)
        else:
            # Substring match
            matches = [m for m in available if token.lower() in m.lower()]
            if len(matches) == 1:
                resolved.append(matches[0])
            elif len(matches) > 1:
                console.print(f"[yellow]'{token}' matches multiple: {', '.join(matches)}[/yellow]")
            else:
                console.print(f"[red]Unknown model: {token}[/red]")

    if not resolved:
        console.print("[red]No valid models resolved.[/red]")
        console.print("[dim]Available (use number, name, or substring):[/dim]")
        for i, name in enumerate(available_no_embed, 1):
            console.print(f"  [cyan]{i:2d}[/cyan]  {name}")
        sys.exit(1)

    # Deduplicate
    seen = set()
    deduped = [m for m in resolved if m not in seen and not seen.add(m)]
    return deduped


def _filter_tests(tests, categories=None, test_names=None):
    """Filter tests by category and/or name (substring match)."""
    if categories:
        tests = [t for t in tests if t.category in categories]
    if test_names:
        filtered = []
        for t in tests:
            for name in test_names:
                if name.lower() in t.name.lower():
                    filtered.append(t)
                    break
        tests = filtered
    return tests


# ── crucible list ─────────────────────────────────────────────────────


def _cmd_list(args, console: Console):
    """List models, tests, and/or previous results."""
    what = getattr(args, "what", "all")

    if what in ("all", "models"):
        available = list_available_models()
        available_no_embed = [m for m in available if "embed" not in m.lower()]
        console.print(f"\n[bold]Ollama models ({len(available_no_embed)}):[/bold]")
        for i, name in enumerate(available_no_embed, 1):
            console.print(f"  [cyan]{i:2d}[/cyan]  {name}")

    if what in ("all", "tests"):
        tests = load_tests(TESTS_DIR)
        console.print(f"\n[bold]Tests ({len(tests)}):[/bold]")
        current_cat = ""
        for t in tests:
            if t.category != current_cat:
                current_cat = t.category
                console.print(f"\n  [bold cyan]{current_cat}[/bold cyan]")
            console.print(f"    {t.name}: {t.description} [dim][{t.eval_type}][/dim]")

    if what == "prompts":
        tests = load_tests(TESTS_DIR)
        console.print(f"\n[bold]Test prompts ({len(tests)}):[/bold]")
        for t in tests:
            console.print(Panel(
                t.prompt.strip(),
                title=f"[bold]{t.category}[/bold] / {t.name}",
                subtitle=f"[dim]{t.description}  [{t.eval_type}][/dim]",
                title_align="left",
                subtitle_align="left",
                border_style="cyan",
                width=min(console.width, 120),
            ))
            console.print()

    if what in ("all", "results"):
        # Show per-model result databases
        model_files = sorted(
            f for f in RESULTS_DIR.glob("*.json")
            if not f.stem[0].isdigit() and "_judged" not in f.stem
        )
        legacy_files = sorted(
            (f for f in RESULTS_DIR.glob("*.json") if f.stem[0].isdigit()),
            reverse=True,
        )

        console.print(f"\n[bold]Model results ({len(model_files)}):[/bold]")
        if not model_files:
            console.print("  [dim]No results yet.[/dim]")
        for f in model_files:
            try:
                with open(f) as fh:
                    data = json.load(fh)
                model = data.get("model", f.stem)
                n_tests = len(data.get("results", []))
                n_pass = sum(1 for r in data.get("results", []) if r.get("passed"))
                updated = data.get("updated", "")[:16]
                console.print(
                    f"  [cyan]{model}[/cyan]  {n_pass}/{n_tests} passing  "
                    f"[dim]{f.name}  updated {updated}[/dim]"
                )
            except Exception:
                console.print(f"  {f.name}")

        if legacy_files:
            console.print(f"\n  [dim]Legacy run files ({len(legacy_files)}):[/dim]")
            for f in legacy_files[:5]:
                console.print(f"    [dim]{f.name}[/dim]")

    console.print()


# ── crucible run ──────────────────────────────────────────────────────


def _cmd_run(args, console: Console):
    """Run tests against local models."""
    models = args.models
    categories = args.category
    test_names = args.test

    # Resolve numeric model references (e.g. -m 1 3 4)
    if models:
        models = _resolve_models(models, console)
    else:
        if args.interactive or not args.models:
            models = _pick_models(console)

    if not categories and not test_names and args.interactive:
        categories = _pick_categories(console)
    elif not categories and not test_names and not args.models:
        categories = _pick_categories(console)

    if not models:
        console.print("[red]No models specified.[/red]")
        sys.exit(1)

    tests = load_tests(TESTS_DIR)
    tests = _filter_tests(tests, categories, test_names)
    if not tests:
        console.print("[red]No tests matched.[/red]")
        sys.exit(1)

    # Load previous passing results (unless --fresh)
    # Checks prompt hashes — edited questions are automatically re-run
    previous_results: dict[tuple[str, str], dict] = {}
    if not args.fresh:
        previous_results = load_previous_results(RESULTS_DIR, models, tests)

    skip_count = len(previous_results)

    # Count llm_judge tests that will be deferred
    judge_count = sum(1 for t in tests if t.eval_type == "llm_judge")
    if judge_count:
        console.print(
            f"[dim]{judge_count} tests use llm_judge — responses captured "
            f"for later scoring via [bold]crucible judge[/bold][/dim]"
        )

    total = len(tests) * len(models)
    fresh_count = total - skip_count
    if skip_count:
        console.print(
            f"[dim]♻ {skip_count} previously passing — will reuse "
            f"({fresh_count} to run)[/dim]"
        )
    completed = 0
    run_start = time.perf_counter()

    current_model = ""
    current_test = ""
    current_tokens = 0
    current_elapsed = 0.0
    completed_results: list[str] = []

    def build_progress_table() -> Table:
        elapsed_total = time.perf_counter() - run_start
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("label", style="dim", width=12)
        table.add_column("value")
        table.add_row(
            "Progress",
            f"[bold]{completed}/{total}[/bold] tests "
            f"({_format_time(elapsed_total)} elapsed)"
        )
        if 0 < completed < total:
            avg = elapsed_total / completed
            eta = avg * (total - completed)
            table.add_row("ETA", f"~{_format_time(eta)} remaining")
        if current_model:
            table.add_row(
                "Running",
                f"[cyan]{current_model}[/cyan] / [white]{current_test}[/white]"
            )
            if current_tokens < 0:
                # Negative means thinking tokens (from thinking models)
                table.add_row(
                    "Thinking",
                    f"[dim]{-current_tokens} tokens ({_format_time(current_elapsed)})[/dim]"
                )
            else:
                table.add_row(
                    "Generating",
                    f"{current_tokens} tokens ({_format_time(current_elapsed)})"
                )
        for line in completed_results[-5:]:
            table.add_row("", line)
        return table

    def on_token(model, test_name, token_count, elapsed):
        nonlocal current_model, current_test, current_tokens, current_elapsed
        current_model = model
        current_test = test_name
        current_tokens = token_count  # negative = thinking tokens
        current_elapsed = elapsed
        live.update(build_progress_table())

    def on_result(model, test_name, result):
        nonlocal completed, current_model
        completed += 1
        current_model = ""
        t = result.model_response.total_time
        score = result.eval_result.score
        passed = result.eval_result.passed
        timed_out = result.model_response.timed_out
        is_deferred = result.test.eval_type == "llm_judge" and score == 0.5
        empty_response = not result.model_response.response.strip()
        thinking_tokens = result.model_response.thinking_tokens

        # Build timing info — include thinking tokens if present
        time_info = _format_time(t)
        if thinking_tokens:
            time_info += f", {thinking_tokens} thinking tokens"

        if timed_out:
            line = f"[red]⏱ [/red] {model} / {test_name} — timeout ({time_info})"
        elif empty_response:
            line = f"[red]✗[/red]  {model} / {test_name} — [red]empty response[/red] ({time_info})"
        elif is_deferred:
            line = f"[blue]⏳[/blue] {model} / {test_name} — captured ({time_info})"
        elif result.test.eval_type == "manual":
            line = f"[yellow]?[/yellow]  {model} / {test_name} — review ({time_info})"
        elif passed:
            line = f"[green]✓[/green]  {model} / {test_name} — {score:.0%} ({time_info})"
        else:
            # Show brief failure reason
            reason = result.eval_result.details
            if len(reason) > 60:
                reason = reason[:57] + "..."
            line = f"[red]✗[/red]  {model} / {test_name} — {score:.0%} [dim]{reason}[/dim] ({time_info})"
        completed_results.append(line)
        live.update(build_progress_table())

    def on_skip(model, test_name, result):
        nonlocal completed, current_model
        completed += 1
        current_model = ""
        score = result.eval_result.score
        line = f"[blue]♻[/blue]  {model} / {test_name} — reused {score:.0%}"
        completed_results.append(line)
        live.update(build_progress_table())

    console.print(
        f"\n[bold]Running {len(tests)} tests x {len(models)} models "
        f"= {total} evaluations[/bold]"
    )
    token_info = f"max {args.max_tokens} tokens" if args.max_tokens else "unlimited tokens"
    console.print(
        f"[dim]Timeout: {_format_time(args.timeout)}/test, "
        f"{token_info}/test[/dim]\n"
    )

    with Live(build_progress_table(), console=console, refresh_per_second=2) as live:
        results = run_tests(
            models, tests,
            on_result=on_result,
            on_token=on_token,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            previous_results=previous_results,
            on_skip=on_skip,
        )

    # Verbose: show full responses after each test
    if args.verbose:
        console.print()
        n_models = len(models)
        for test in tests:
            console.print(f"\n[bold cyan]━━━ {test.category}/{test.name} ━━━[/bold cyan]")
            console.print(f"[dim]{test.description}[/dim]\n")

            # Build panels for each model
            panels = []
            for model in models:
                model_results = results[model]
                r = next((r for r in model_results if r.test.name == test.name), None)
                if r is None:
                    continue

                score = r.eval_result.score
                passed = r.eval_result.passed
                time_s = r.model_response.total_time
                style = "green" if passed else "red"
                header = f"{model}  [{style}]{score:.0%}[/{style}]  {time_s:.1f}s"
                response = r.model_response.response or "[dim]<empty response>[/dim]"
                eval_line = f"\n\n[dim]Eval: {r.eval_result.details}[/dim]"

                panels.append(Panel(
                    response + eval_line,
                    title=header,
                    title_align="left",
                    border_style="cyan" if passed else "red",
                ))

            if n_models > 1:
                # Side-by-side using a table to hold the panels
                side_table = Table(
                    show_header=False, expand=True,
                    show_edge=False, box=None, padding=(0, 1),
                )
                for _ in panels:
                    side_table.add_column(ratio=1)
                side_table.add_row(*panels)
                console.print(side_table)
            else:
                for panel in panels:
                    panel.width = min(console.width, 120)
                    console.print(panel)

            console.print()

    print_summary(results, console)

    if not args.no_save:
        model_paths = save_results(results, RESULTS_DIR)
        md_path = save_markdown(results, RESULTS_DIR)
        console.print(f"\n[dim]Results saved to:[/dim]")
        for p in model_paths:
            console.print(f"  {p}")
        console.print(f"  {md_path}")
        if judge_count:
            paths_str = " ".join(str(p) for p in model_paths)
            console.print(
                f"\n[bold]To score with Claude:[/bold] "
                f"crucible judge {paths_str}"
            )


# ── crucible judge ────────────────────────────────────────────────────


def _cmd_judge(args, console: Console):
    """Score previously captured responses with Claude Code."""
    from crucible.judge import judge_results

    results_path = Path(args.results_file)
    if not results_path.exists():
        results_path = RESULTS_DIR / args.results_file
    if not results_path.exists():
        console.print(f"[red]File not found: {args.results_file}[/red]")
        console.print(f"[dim]Available results:[/dim]")
        for f in sorted(RESULTS_DIR.glob("*.json")):
            console.print(f"  {f.name}")
        sys.exit(1)

    judge_results(
        results_path, console,
        test_names=args.test,
    )


# ── crucible compare ──────────────────────────────────────────────────


def _load_compare_data(files: list[str], console: Console) -> tuple[list[str], dict]:
    """Load result data from one or more files. Returns (models, lookup).

    Handles both per-model files (new format) and legacy multi-model files.
    lookup maps (model, test_name) → result dict.
    """
    lookup = {}
    models_seen = []

    for raw_path in files:
        path = Path(raw_path)
        if not path.exists():
            path = RESULTS_DIR / raw_path
        if not path.exists():
            console.print(f"[red]File not found: {raw_path}[/red]")
            console.print(f"[dim]Available results:[/dim]")
            for f in sorted(RESULTS_DIR.glob("*.json")):
                console.print(f"  {f.name}")
            sys.exit(1)

        with open(path) as f:
            data = json.load(f)

        if "model" in data and "results" in data:
            # Per-model format
            model = data["model"]
            if model not in models_seen:
                models_seen.append(model)
            for entry in data["results"]:
                lookup[(model, entry["test"])] = entry
        elif "models" in data:
            # Legacy multi-model format
            for model, entries in data["models"].items():
                if model not in models_seen:
                    models_seen.append(model)
                for entry in entries:
                    lookup[(model, entry["test"])] = entry

    return models_seen, lookup


def _cmd_compare(args, console: Console):
    """Side-by-side comparison of responses."""
    files = args.results_files

    # If no files given, auto-load all per-model DBs in results/
    if not files:
        files = [str(f) for f in sorted(RESULTS_DIR.glob("*.json"))
                 if not f.stem[0].isdigit() and "_judged" not in f.stem]
        if not files:
            console.print("[red]No result files found.[/red]")
            sys.exit(1)

    models, lookup = _load_compare_data(files, console)

    if not models:
        console.print("[red]No model data found.[/red]")
        sys.exit(1)

    # Get all test names across all models
    all_tests_set: dict[str, str] = {}  # test_name → category
    for (model, test_name), entry in lookup.items():
        if test_name not in all_tests_set:
            all_tests_set[test_name] = entry.get("category", "")
    all_tests = [(cat, name) for name, cat in all_tests_set.items()]

    # Filter by category and/or test name
    filtered = all_tests
    if args.category:
        filtered = [(c, t) for c, t in filtered if c in args.category]
    if args.test:
        new_filtered = []
        for c, t in filtered:
            for name in args.test:
                if name.lower() in t.lower():
                    new_filtered.append((c, t))
                    break
        filtered = new_filtered

    if not filtered:
        console.print("[red]No tests matched.[/red]")
        console.print("[dim]Available tests:[/dim]")
        for cat, name in all_tests:
            console.print(f"  {cat}/{name}")
        sys.exit(1)

    # Build lookup: (model, test_name) -> result dict
    lookup = {}
    for model in models:
        for r in data["models"][model]:
            lookup[(model, r["test"])] = r

    # Display each matched test
    n_models = len(models)
    for cat, test_name in filtered:
        console.print(f"\n[bold cyan]━━━ {cat}/{test_name} ━━━[/bold cyan]")

        # Show the prompt if stored in results
        first = lookup.get((models[0], test_name), {})
        prompt = first.get("prompt", "")
        if prompt:
            console.print(Panel(
                prompt.strip(),
                title="[dim]Prompt[/dim]",
                title_align="left",
                border_style="dim",
                width=min(console.width, 120),
            ))
        console.print()

        # Build panels for each model
        panels = []
        for model in models:
            r = lookup.get((model, test_name))
            if not r:
                continue
            score_str = f"{r['score']:.0%}" if r['score'] is not None else "—"
            time_str = f"{r['total_time']:.1f}s"
            passed = r.get("passed", False)
            style = "green" if passed else "red"
            header = f"{model}  [{style}]{score_str}[/{style}]  {time_str}"
            response = r.get("response", "") or "[dim]<empty response>[/dim]"

            panels.append(Panel(
                response,
                title=header,
                title_align="left",
                border_style="cyan" if passed else "red",
            ))

        if n_models > 1:
            # Side-by-side using a table to hold the panels
            side_table = Table(
                show_header=False, expand=True,
                show_edge=False, box=None, padding=(0, 1),
            )
            for _ in panels:
                side_table.add_column(ratio=1)
            side_table.add_row(*panels)
            console.print(side_table)
        else:
            for panel in panels:
                panel.width = min(console.width, 120)
                console.print(panel)

        console.print()


# ── crucible report ────────────────────────────────────────────────────


def _cmd_report(args, console: Console):
    """Generate a visual HTML report with charts."""
    from crucible.report import generate_html_report

    files = args.results_files
    if not files:
        files = [str(f) for f in sorted(RESULTS_DIR.glob("*.json"))
                 if not f.stem[0].isdigit() and "_judged" not in f.stem]
        if not files:
            console.print("[red]No result files found in results/[/red]")
            sys.exit(1)

    models, lookup = _load_compare_data(files, console)

    if not models:
        console.print("[red]No model data found.[/red]")
        sys.exit(1)

    output_path = Path(args.output) if args.output else RESULTS_DIR / "report.html"
    path = generate_html_report(models, lookup, output_path, title=args.title)
    console.print(f"[bold green]Report generated:[/bold green] {path}")
    console.print(f"[dim]Open in browser: file://{path.resolve()}[/dim]")


# ── main ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Crucible — compare local Ollama models",
        prog="crucible",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- crucible list ---
    list_p = subparsers.add_parser("list", help="List models and/or tests")
    list_p.add_argument("what", nargs="?", default="all",
                        choices=["all", "models", "tests", "prompts", "results"],
                        help="What to list (default: all)")

    # --- crucible run ---
    run_p = subparsers.add_parser("run", help="Run tests against models")
    run_p.add_argument("--models", "-m", nargs="+")
    run_p.add_argument("--category", "-c", nargs="+",
                       help="Filter by category (e.g. code_generation reasoning)")
    run_p.add_argument("--test", "-n", nargs="+",
                       help="Filter by test name (substring match, e.g. beam section)")
    run_p.add_argument("--no-save", action="store_true")
    run_p.add_argument("--interactive", "-i", action="store_true")
    run_p.add_argument("--timeout", type=float, default=600.0,
                       help="Max seconds per test (default: 600, overridden by per-test timeout in YAML)")
    run_p.add_argument("--tokens", type=int, default=0, dest="max_tokens",
                       help="Max completion tokens (0 = unlimited, default)")
    run_p.add_argument("--verbose", "-v", action="store_true",
                       help="Show full model responses after each test")
    run_p.add_argument("--fresh", action="store_true",
                       help="Ignore previous results and re-run all tests")

    # --- crucible judge ---
    judge_p = subparsers.add_parser("judge", help="Score responses with Claude Code")
    judge_p.add_argument("results_file",
                         help="Results JSON path (or filename in results/)")
    judge_p.add_argument("--test", "-n", nargs="+",
                         help="Only judge these tests (substring match)")

    # --- crucible compare ---
    cmp_p = subparsers.add_parser("compare", help="Side-by-side response comparison")
    cmp_p.add_argument("results_files", nargs="*",
                       help="Result JSON files (default: all per-model files in results/)")
    cmp_p.add_argument("--category", "-c", nargs="+",
                       help="Filter by category")
    cmp_p.add_argument("--test", "-n", nargs="+",
                       help="Filter by test name (substring match)")

    # --- crucible report ---
    report_p = subparsers.add_parser("report", help="Generate HTML visual report")
    report_p.add_argument("results_files", nargs="*",
                          help="Result JSON files (default: all per-model files in results/)")
    report_p.add_argument("-o", "--output", default=None,
                          help="Output HTML path (default: results/report.html)")
    report_p.add_argument("--title", default="Crucible Benchmark Report",
                          help="Report title")

    # If no subcommand given, default to `run` (supports bare `crucible -m 1 2`)
    known_commands = {"list", "run", "judge", "compare", "report"}
    argv = sys.argv[1:]
    if argv and argv[0] not in known_commands and not argv[0].startswith("-h"):
        # Looks like flags without a subcommand — treat as `run`
        argv = ["run"] + argv
    elif not argv:
        # Bare `crucible` → interactive run
        argv = ["run"]

    args = parser.parse_args(argv)
    console = Console()

    if args.command == "list":
        _cmd_list(args, console)
    elif args.command == "judge":
        _cmd_judge(args, console)
    elif args.command == "compare":
        _cmd_compare(args, console)
    elif args.command == "report":
        _cmd_report(args, console)
    else:
        _cmd_run(args, console)


if __name__ == "__main__":
    main()
