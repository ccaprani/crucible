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
from crucible.report import print_summary, save_results
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

    if what in ("all", "results"):
        result_files = sorted(RESULTS_DIR.glob("*.json"), reverse=True)
        console.print(f"\n[bold]Results ({len(result_files)}):[/bold]")
        if not result_files:
            console.print("  [dim]No results yet.[/dim]")
        for f in result_files[:10]:
            # Peek inside for model names
            try:
                with open(f) as fh:
                    data = json.load(fh)
                models = list(data.get("models", {}).keys())
                n_tests = len(next(iter(data["models"].values()), []))
                label = f"{', '.join(models)} ({n_tests} tests)"
            except Exception:
                label = ""
            suffix = " [dim](judged)[/dim]" if "_judged" in f.stem else ""
            console.print(f"  {f.name}{suffix}  [dim]{label}[/dim]")

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

    # Count llm_judge tests that will be deferred
    judge_count = sum(1 for t in tests if t.eval_type == "llm_judge")
    if judge_count:
        console.print(
            f"[dim]{judge_count} tests use llm_judge — responses captured "
            f"for later scoring via [bold]crucible judge[/bold][/dim]"
        )

    total = len(tests) * len(models)
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
        current_tokens = token_count
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

        if timed_out:
            line = f"[red]⏱ [/red] {model} / {test_name} — timeout ({_format_time(t)})"
        elif is_deferred:
            line = f"[blue]⏳[/blue] {model} / {test_name} — captured ({_format_time(t)})"
        elif result.test.eval_type == "manual":
            line = f"[yellow]?[/yellow]  {model} / {test_name} — review ({_format_time(t)})"
        elif passed:
            line = f"[green]✓[/green]  {model} / {test_name} — {score:.0%} ({_format_time(t)})"
        else:
            line = f"[red]✗[/red]  {model} / {test_name} — {score:.0%} ({_format_time(t)})"
        completed_results.append(line)
        live.update(build_progress_table())

    console.print(
        f"\n[bold]Running {len(tests)} tests x {len(models)} models "
        f"= {total} evaluations[/bold]"
    )
    console.print(
        f"[dim]Timeout: {_format_time(args.timeout)}/test, "
        f"max {args.max_tokens} tokens/test[/dim]\n"
    )

    with Live(build_progress_table(), console=console, refresh_per_second=2) as live:
        results = run_tests(
            models, tests,
            on_result=on_result,
            on_token=on_token,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )

    print_summary(results, console)

    if not args.no_save:
        md_path, json_path = save_results(results, RESULTS_DIR)
        console.print(f"\n[dim]Results saved to:[/dim]")
        console.print(f"  {md_path}")
        console.print(f"  {json_path}")
        if judge_count:
            console.print(
                f"\n[bold]To score with Claude:[/bold] "
                f"crucible judge {json_path}"
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


def _cmd_compare(args, console: Console):
    """Side-by-side comparison of responses from a previous run."""
    results_path = Path(args.results_file)
    if not results_path.exists():
        results_path = RESULTS_DIR / args.results_file
    if not results_path.exists():
        console.print(f"[red]File not found: {args.results_file}[/red]")
        console.print(f"[dim]Available results:[/dim]")
        for f in sorted(RESULTS_DIR.glob("*.json")):
            console.print(f"  {f.name}")
        sys.exit(1)

    with open(results_path) as f:
        data = json.load(f)

    models = list(data["models"].keys())
    if not models:
        console.print("[red]No model data in results file.[/red]")
        sys.exit(1)

    # Get all test names from first model
    all_tests = [(r["category"], r["test"]) for r in data["models"][models[0]]]

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
    for cat, test_name in filtered:
        console.print(f"\n[bold cyan]━━━ {cat}/{test_name} ━━━[/bold cyan]\n")

        # Show prompt (from first model's result)
        first = lookup.get((models[0], test_name), {})
        # Prompt isn't stored in results — show eval info instead
        for model in models:
            r = lookup.get((model, test_name))
            if not r:
                continue

            score_str = f"{r['score']:.0%}" if r['score'] is not None else "—"
            time_str = f"{r['total_time']:.1f}s"
            passed = r.get("passed", False)
            judge = r.get("judge_details", "")

            style = "green" if passed else "red"
            header = f"{model}  [{style}]{score_str}[/{style}]  {time_str}"
            if judge:
                header += f"  [dim]{judge[:60]}[/dim]"

            # Truncate very long responses for readability
            response = r.get("response", "")
            if len(response) > 2000:
                response = response[:2000] + "\n\n[dim]... truncated ...[/dim]"

            console.print(Panel(
                response,
                title=header,
                title_align="left",
                border_style="cyan" if passed else "red",
                width=min(console.width, 120),
            ))

        console.print()


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
                        choices=["all", "models", "tests", "results"],
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
    run_p.add_argument("--timeout", "-t", type=float, default=300.0)
    run_p.add_argument("--max-tokens", type=int, default=2048)

    # --- crucible judge ---
    judge_p = subparsers.add_parser("judge", help="Score responses with Claude Code")
    judge_p.add_argument("results_file",
                         help="Results JSON path (or filename in results/)")
    judge_p.add_argument("--test", "-n", nargs="+",
                         help="Only judge these tests (substring match)")

    # --- crucible compare ---
    cmp_p = subparsers.add_parser("compare", help="Side-by-side response comparison")
    cmp_p.add_argument("results_file",
                       help="Results JSON path (or filename in results/)")
    cmp_p.add_argument("--category", "-c", nargs="+",
                       help="Filter by category")
    cmp_p.add_argument("--test", "-n", nargs="+",
                       help="Filter by test name (substring match)")

    # Top-level flags → implicit `run`
    parser.add_argument("--models", "-m", nargs="+")
    parser.add_argument("--category", "-c", nargs="+")
    parser.add_argument("--test", "-n", nargs="+")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--timeout", "-t", type=float, default=300.0)
    parser.add_argument("--max-tokens", type=int, default=2048)

    args = parser.parse_args()
    console = Console()

    if args.command == "list":
        _cmd_list(args, console)
    elif args.command == "judge":
        _cmd_judge(args, console)
    elif args.command == "compare":
        _cmd_compare(args, console)
    else:
        _cmd_run(args, console)


if __name__ == "__main__":
    main()
