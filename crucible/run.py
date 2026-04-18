#!/usr/bin/env python3
"""CLI entry point for Crucible."""

import argparse
import json
import sys
import time
from pathlib import Path

import os

from rich.console import Console, Group
from rich.columns import Columns
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text


_BANNER = "[bold cyan]╔═╗┬─┐┬ ┬┌─┐┬┌┐ ┬  ┌─┐   ┌─┐┬[/bold cyan]\n[bold cyan]║  ├┬┘│ ││  │├┴┐│  ├┤    ├─┤│[/bold cyan]\n[bold cyan]╚═╝┴└─└─┘└─┘┴└─┘┴─┘└─┘   ┴ ┴┴[/bold cyan]"


def _clear_and_banner(console: Console):
    """Clear terminal and print the Crucible banner."""
    if sys.stdout.isatty():
        os.system("clear" if os.name != "nt" else "cls")
    console.print(_BANNER)
    console.print("[dim]crucible AI · LLM benchmark for structural engineering academics[/dim]")
    console.print("[dim]Colin Caprani · Monash University · github.com/ccaprani/crucible[/dim]\n")

from crucible.models import list_available_models
from crucible.report import (
    load_model_results, load_previous_results,
    print_summary, save_markdown, save_model_results, save_results,
)
from crucible.runner import load_tests, run_tests

_PKG_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = _PKG_ROOT / "tests"
RESULTS_DIR = _PKG_ROOT / "results"


def _resolve_test_dir(raw: str | None) -> Path:
    """Resolve the test directory, defaulting to the tracked tests/ folder."""
    if not raw:
        return TESTS_DIR
    path = Path(raw)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Test directory not found: {path}")
    return path


def _pick_models(console: Console, gguf_dir: Path | None = None) -> list[str]:
    """Interactive model picker."""
    available = list_available_models(gguf_dir)
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


def _resolve_models(raw: list[str], console: Console, gguf_dir: Path | None = None) -> list[str]:
    """Resolve model references — accepts numbers, names, substrings, or API model names."""
    from crucible.models import is_api_model
    available = list_available_models(gguf_dir)
    available_no_embed = [m for m in available if "embed" not in m.lower()]

    resolved = []
    for token in raw:
        # API models pass through directly (e.g. claude-opus-4-6)
        if is_api_model(token):
            resolved.append(token)
        elif token.isdigit():
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


def _filter_suites(tests, suites=None):
    """Filter tests by suite name."""
    if suites:
        tests = [t for t in tests if t.suite in suites]
    return tests


def _purge_match(entry: dict, suites=None, categories=None, test_names=None) -> bool:
    """Return True if a stored result entry matches ALL active purge filters (AND logic).

    Each filter is optional; when absent it imposes no constraint.
    When multiple filters are active, the entry must satisfy every one.
    """
    if suites and entry.get("suite", "default") not in suites:
        return False
    if categories and entry.get("category") not in categories:
        return False
    if test_names:
        haystacks = [
            entry.get("test", ""),
            entry.get("test_id", ""),
        ]
        if not any(any(token.lower() in h.lower() for h in haystacks) for token in test_names):
            return False
    return True


# ── crucible list ─────────────────────────────────────────────────────


def _cmd_list(args, console: Console):
    """List models, tests, and/or previous results."""
    _clear_and_banner(console)
    what = getattr(args, "what", "all")
    test_dir = _resolve_test_dir(getattr(args, "test_dir", None))

    if what in ("all", "models"):
        from crucible.models import get_model_info
        gguf_dir = getattr(args, "gguf_dir", None)
        available = list_available_models(gguf_dir)
        available_no_embed = [m for m in available if "embed" not in m.lower()]

        # Gather metadata
        infos = []
        for name in available_no_embed:
            try:
                infos.append(get_model_info(name))
            except Exception:
                infos.append(None)

        _TIER_COLOURS = {"8GB": "green", "16GB": "yellow", "24GB": "red"}

        table = Table(
            show_header=True, header_style="bold dim", box=None, padding=(0, 1),
            title=f"Ollama models ({len(available_no_embed)})",
            title_style="bold", title_justify="left",
        )
        table.add_column("#", style="cyan", justify="right", width=3)
        table.add_column("Model", style="bold white")
        table.add_column("Params", style="magenta", justify="right")
        table.add_column("Quant", style="dim")
        table.add_column("VRAM", style="dim", justify="right")
        table.add_column("Tier", justify="center")

        for i, (name, mi) in enumerate(zip(available_no_embed, infos), 1):
            if mi:
                tc = _TIER_COLOURS.get(mi.vram_tier, "dim")
                table.add_row(
                    str(i), name,
                    mi.parameter_size, mi.quantization,
                    f"~{mi.vram_gb:.1f}GB",
                    f"[{tc}]{mi.vram_tier}[/{tc}]",
                )
            else:
                table.add_row(str(i), name, "", "", "", "")

        console.print()
        console.print(table)

    if what in ("all", "tests"):
        tests = load_tests(test_dir)
        suites = getattr(args, "suite", None)
        tests = _filter_suites(tests, suites)
        console.print(f"\n[bold]Tests ({len(tests)}):[/bold]")
        current_cat = ""
        for t in tests:
            cat_label = t.category if t.suite == "default" else f"{t.suite}/{t.category}"
            if cat_label != current_cat:
                current_cat = cat_label
                console.print(f"\n  [bold cyan]{current_cat}[/bold cyan]")
            console.print(f"    {t.name}: {t.description} [dim][{t.eval_type}][/dim]")

    if what == "prompts":
        tests = load_tests(test_dir)
        tests = _filter_suites(tests, getattr(args, "suite", None))
        console.print(f"\n[bold]Test prompts ({len(tests)}):[/bold]")
        for t in tests:
            console.print(Panel(
                t.prompt.strip(),
                title=f"[bold]{t.suite}[/bold] / {t.category} / {t.name}" if t.suite != "default" else f"[bold]{t.category}[/bold] / {t.name}",
                subtitle=f"[dim]{t.description}  [{t.eval_type}][/dim]",
                title_align="left",
                subtitle_align="left",
                border_style="cyan",
                width=min(console.width, 120),
            ))
            console.print()

    if what in ("all", "results"):
        # Show per-model result databases
        model_files = sorted(RESULTS_DIR.glob("*.json"))

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

    console.print()


# ── crucible run ──────────────────────────────────────────────────────


def _cmd_run(args, console: Console):
    """Run tests against local models."""
    _clear_and_banner(console)
    test_dir = _resolve_test_dir(getattr(args, "test_dir", None))
    models = args.models
    categories = args.category
    suites = args.suite
    test_names = args.test

    # Resolve numeric model references (e.g. -m 1 3 4)
    gguf_dir = getattr(args, "gguf_dir", None)
    if models:
        models = _resolve_models(models, console, gguf_dir)
    else:
        if args.interactive or not args.models:
            models = _pick_models(console, gguf_dir)

    if not categories and not test_names and args.interactive:
        categories = _pick_categories(console)
    elif not categories and not test_names and not args.models:
        categories = _pick_categories(console)

    if not models:
        console.print("[red]No models specified.[/red]")
        sys.exit(1)

    tests = load_tests(test_dir)
    tests = _filter_suites(tests, suites)
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

    # Count llm_judge tests
    from crucible.models import is_api_model as _is_api
    judge_count = sum(1 for t in tests if t.eval_type == "llm_judge")
    all_api = all(_is_api(m) for m in models)
    if judge_count and not all_api:
        console.print(
            f"[dim]{judge_count} llm_judge tests — scored by Opus[/dim]"
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
    stream_text: list[str] = []  # accumulated response text for live panel

    def _stream_panel_height() -> int:
        """Remaining terminal lines after everything above the panel."""
        # Banner: 3 (ASCII art) + 1 (tagline) + 1 (authorship) + 1 (blank) = 6
        # Run header: "Running N tests..." + "Timeout..." + blank = 3
        # Progress table: up to 4 status rows + up to 5 completed results
        banner_rows = 6
        header_rows = 3
        progress_rows = 4 + min(len(completed_results), 5)
        used = banner_rows + header_rows + progress_rows
        return max(6, (console.height or 40) - used - 4)

    def build_display() -> Group:
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
                tks = -current_tokens / current_elapsed if current_elapsed > 0 else 0
                table.add_row(
                    "Thinking",
                    f"[dim]{-current_tokens} tokens ({_format_time(current_elapsed)}) · {tks:.1f} tk/s[/dim]"
                )
            else:
                tks = current_tokens / current_elapsed if current_elapsed > 0 else 0
                table.add_row(
                    "Generating",
                    f"{current_tokens} tokens ({_format_time(current_elapsed)}) · {tks:.1f} tk/s"
                )
        for line in completed_results[-5:]:
            table.add_row("", line)

        # Live response panel — fill remaining screen
        if current_model and stream_text:
            panel_h = _stream_panel_height()
            # Join all text, take last N lines that fit
            full = "".join(stream_text)
            lines = full.split("\n")
            visible = lines[-(panel_h):] if len(lines) > panel_h else lines
            panel_text = "\n".join(visible)

            is_thinking = current_tokens < 0
            panel_title = (
                f"[bold]{'💭 Thinking' if is_thinking else '💬 Response'}[/bold]  "
                f"[dim]{current_model} / {current_test}[/dim]"
            )
            panel = Panel(
                Text(panel_text, style="italic dim" if is_thinking else ""),
                title=panel_title,
                title_align="left",
                border_style="red",
                height=panel_h + 2,  # +2 for border
                width=min(console.width, 120),
            )
            return Group(table, panel)

        return Group(table)

    def on_token(model, test_name, token_count, elapsed, text=""):
        nonlocal current_model, current_test, current_tokens, current_elapsed
        current_model = model
        current_test = test_name
        current_tokens = token_count  # negative = thinking tokens
        current_elapsed = elapsed
        # Strip the ⟨think⟩ prefix for display
        if text.startswith("⟨think⟩"):
            stream_text.append(text[7:])
        else:
            stream_text.append(text)
        live.update(build_display())

    def on_result(model, test_name, result):
        nonlocal completed, current_model
        completed += 1
        current_model = ""
        stream_text.clear()

        # Incremental save — never lose completed work
        if not args.no_save:
            save_model_results([result], model, RESULTS_DIR)

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
        live.update(build_display())

    def on_skip(model, test_name, result):
        nonlocal completed, current_model
        completed += 1
        current_model = ""
        stream_text.clear()
        score = result.eval_result.score
        line = f"[blue]♻[/blue]  {model} / {test_name} — reused {score:.0%}"
        completed_results.append(line)
        live.update(build_display())

    console.print(
        f"\n[bold]Running {len(tests)} tests x {len(models)} models "
        f"= {total} evaluations[/bold]"
    )
    token_info = f"max {args.max_tokens} tokens" if args.max_tokens else "unlimited tokens"
    console.print(
        f"[dim]Timeout: {_format_time(args.timeout)}/test, "
        f"{token_info}/test[/dim]\n"
    )

    with Live(build_display(), console=console, refresh_per_second=4) as live:
        results = run_tests(
            models, tests,
            on_result=on_result,
            on_token=on_token,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            previous_results=previous_results,
            on_skip=on_skip,
            test_dir=test_dir,
            gguf_dir=getattr(args, "gguf_dir", None),
        )

    # Verbose: show full responses after each test
    if args.verbose:
        console.print()
        n_models = len(models)
        for test in tests:
            label = f"{test.category}/{test.name}"
            if test.suite != "default":
                label = f"{test.suite}/{label}"
            console.print(f"\n[bold cyan]━━━ {label} ━━━[/bold cyan]")
            console.print(f"[dim]{test.description}[/dim]\n")

            # Build panels for each model
            panels = []
            for model in models:
                model_results = results[model]
                r = next((r for r in model_results if r.test.test_id == test.test_id), None)
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


# ── crucible judge ────────────────────────────────────────────────────


def _cmd_judge(args, console: Console):
    """Score previously captured responses via LLM judge."""
    _clear_and_banner(console)
    test_dir = _resolve_test_dir(getattr(args, "test_dir", None))

    if getattr(args, "judge_action", None) == "review":
        from crucible.judge import review_judge_results
        review_judge_results(
            results_dir=RESULTS_DIR,
            tests_dir=test_dir,
            console=console,
            model_names=args.models,
            suite_names=args.suite,
            test_names=args.test,
        )
    else:
        from crucible.judge import judge_results
        judge_results(
            results_dir=RESULTS_DIR,
            tests_dir=test_dir,
            console=console,
            model_names=args.models,
            suite_names=args.suite,
            test_names=args.test,
        )


# ── crucible compare ──────────────────────────────────────────────────


def _load_compare_data(files: list[str], console: Console) -> tuple[list[str], dict]:
    """Load result data from one or more files. Returns (models, lookup).

    lookup maps (model, test_id) → result dict.
    """
    from crucible.report import result_test_id

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

        model = data["model"]
        if model not in models_seen:
            models_seen.append(model)
        for entry in data["results"]:
            lookup[(model, result_test_id(entry))] = entry

    return models_seen, lookup


def _cmd_compare(args, console: Console):
    """Side-by-side comparison of responses."""
    _clear_and_banner(console)
    files = args.results_files

    # If no files given, auto-load all per-model DBs in results/
    if not files:
        files = [str(f) for f in sorted(RESULTS_DIR.glob("*.json"))]
        if not files:
            console.print("[red]No result files found.[/red]")
            sys.exit(1)

    all_models, lookup = _load_compare_data(files, console)

    if not all_models:
        console.print("[red]No model data found.[/red]")
        sys.exit(1)

    # Filter models if -m specified
    if args.models:
        models = _filter_report_models(args.models, all_models, console)
        lookup = {k: v for k, v in lookup.items() if k[0] in models}
    else:
        models = all_models

    # Get all test names across selected models
    all_tests_set: dict[str, tuple[str, str]] = {}  # test_id → (suite, category)
    for (_model, test_id), entry in lookup.items():
        if test_id not in all_tests_set:
            all_tests_set[test_id] = (entry.get("suite", "default"), entry.get("category", ""))
    all_tests = [(suite, cat, test_id) for test_id, (suite, cat) in all_tests_set.items()]

    # Filter by category and/or test name
    filtered = all_tests
    if args.suite:
        filtered = [(s, c, t) for s, c, t in filtered if s in args.suite]
    if args.category:
        filtered = [(s, c, t) for s, c, t in filtered if c in args.category]
    if args.test:
        new_filtered = []
        for s, c, t in filtered:
            for name in args.test:
                if name.lower() in t.lower():
                    new_filtered.append((s, c, t))
                    break
        filtered = new_filtered

    if not filtered:
        console.print("[red]No tests matched.[/red]")
        console.print("[dim]Available tests:[/dim]")
        for suite, cat, test_id in all_tests:
            label = f"{cat}/{test_id.split('/')[-1]}"
            if suite != "default":
                label = f"{suite}/{label}"
            console.print(f"  {label}")
        sys.exit(1)

    # Display each matched test
    n_models = len(models)
    for suite, cat, test_id in filtered:
        label = f"{cat}/{test_id.split('/')[-1]}"
        if suite != "default":
            label = f"{suite}/{label}"
        console.print(f"\n[bold cyan]━━━ {label} ━━━[/bold cyan]")

        # Show the prompt if stored in results
        first = lookup.get((models[0], test_id), {})
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
            r = lookup.get((model, test_id))
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


def _cmd_purge(args, console: Console):
    """Remove selected stored result entries from per-model JSON databases."""
    _clear_and_banner(console)

    model_files = sorted(RESULTS_DIR.glob("*.json"))
    if not model_files:
        console.print("[red]No result files found in results/[/red]")
        sys.exit(1)

    available_models = []
    for path in model_files:
        try:
            with open(path) as f:
                data = json.load(f)
            available_models.append(data.get("model", path.stem))
        except Exception:
            continue

    if args.models:
        selected_models = _filter_report_models(args.models, available_models, console)
    else:
        selected_models = available_models

    dry_run = not args.yes
    removed_total = 0
    changed_files = 0

    console.print(f"[bold]Scanning {len(selected_models)} model result database(s)[/bold]")
    if args.suite:
        console.print(f"[dim]Suites:[/dim] {', '.join(args.suite)}")
    if args.category:
        console.print(f"[dim]Categories:[/dim] {', '.join(args.category)}")
    if args.test:
        console.print(f"[dim]Test filter:[/dim] {', '.join(args.test)}")
    console.print(f"[dim]Mode:[/dim] {'dry run' if dry_run else 'purge'}\n")

    for path in model_files:
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            console.print(f"[yellow]Skipping unreadable file:[/yellow] {path.name}")
            continue

        model = data.get("model", path.stem)
        if model not in selected_models:
            continue

        results = data.get("results", [])
        kept = []
        removed = []
        for entry in results:
            if _purge_match(entry, args.suite, args.category, args.test):
                removed.append(entry)
            else:
                kept.append(entry)

        if not removed:
            continue

        changed_files += 1
        removed_total += len(removed)
        console.print(f"[cyan]{model}[/cyan]: {len(removed)} entr{'y' if len(removed) == 1 else 'ies'} matched")
        for entry in removed[:8]:
            label = entry.get("test_id", entry.get("test", "<unknown>"))
            console.print(f"  [dim]- {label}[/dim]")
        if len(removed) > 8:
            console.print(f"  [dim]... and {len(removed) - 8} more[/dim]")

        if not dry_run:
            data["results"] = kept
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

    console.print()
    if changed_files == 0:
        console.print("[yellow]No stored results matched the selected filters.[/yellow]")
        return

    if dry_run:
        console.print(
            f"[bold yellow]Dry run only.[/bold yellow] "
            f"{removed_total} entr{'y' if removed_total == 1 else 'ies'} would be removed "
            f"across {changed_files} file(s)."
        )
        console.print("[dim]Re-run with --yes to apply the purge.[/dim]")
    else:
        console.print(
            f"[bold green]Purged[/bold green] "
            f"{removed_total} entr{'y' if removed_total == 1 else 'ies'} "
            f"across {changed_files} file(s)."
        )



# ── crucible register ─────────────────────────────────────────────────


_KEYS_PATH = Path.home() / ".config" / "crucible" / "keys.json"


def _load_keys() -> dict:
    """Load stored API keys."""
    if _KEYS_PATH.exists():
        try:
            return json.load(open(_KEYS_PATH))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_keys(keys: dict):
    """Save API keys to config file."""
    _KEYS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _KEYS_PATH.write_text(json.dumps(keys, indent=2))
    _KEYS_PATH.chmod(0o600)  # owner-only read/write


def get_api_key(provider: str) -> str | None:
    """Get API key for a provider — checks env var first, then stored keys."""
    env_var = f"{provider.upper()}_API_KEY"
    from_env = os.environ.get(env_var, "").strip()
    if from_env:
        return from_env
    keys = _load_keys()
    return keys.get(provider)


def _cmd_register(args, console: Console):
    """Register an API key for a provider."""
    _clear_and_banner(console)
    provider = args.provider.lower()
    key = args.key

    valid_providers = {"openrouter", "anthropic"}
    if provider not in valid_providers:
        console.print(f"[red]Unknown provider: {provider}[/red]")
        console.print(f"[dim]Valid providers: {', '.join(sorted(valid_providers))}[/dim]")
        sys.exit(1)

    keys = _load_keys()

    if key:
        # Store the key
        keys[provider] = key
        _save_keys(keys)
        masked = key[:8] + "..." + key[-4:]
        console.print(f"[green]✓[/green] {provider} key stored: {masked}")
        console.print(f"[dim]Saved to {_KEYS_PATH}[/dim]")
    else:
        # Show status
        existing = get_api_key(provider)
        if existing:
            masked = existing[:8] + "..." + existing[-4:]
            console.print(f"[green]✓[/green] {provider}: {masked}")
        else:
            console.print(f"[yellow]✗[/yellow] {provider}: not set")
            console.print(f"\n[dim]Usage: crucible register {provider} <api-key>[/dim]")

    # Show all provider statuses
    if not key:
        console.print(f"\n[bold]API key status:[/bold]")
        for p in sorted(valid_providers):
            k = get_api_key(p)
            if k:
                console.print(f"  [green]✓[/green] {p}: {k[:8]}...{k[-4:]}")
            else:
                console.print(f"  [dim]✗ {p}: not set[/dim]")
        console.print(f"\n[dim]Claude CLI (Max subscription) needs no key.[/dim]")


# ── crucible report ────────────────────────────────────────────────────


def _filter_report_models(raw: list[str], available: list[str], console: Console) -> list[str]:
    """Resolve model references against loaded result models (not Ollama)."""
    resolved = []
    for token in raw:
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(available):
                resolved.append(available[idx])
            else:
                console.print(f"[yellow]Invalid index {token} (max {len(available)})[/yellow]")
        elif token in available:
            resolved.append(token)
        else:
            matches = [m for m in available if token.lower() in m.lower()]
            if len(matches) == 1:
                resolved.append(matches[0])
            elif len(matches) > 1:
                console.print(f"[yellow]'{token}' matches multiple: {', '.join(matches)}[/yellow]")
            else:
                console.print(f"[red]No results for model: {token}[/red]")

    if not resolved:
        console.print("[red]No valid models resolved.[/red]")
        console.print("[dim]Available models in results:[/dim]")
        for i, name in enumerate(available, 1):
            console.print(f"  [cyan]{i:2d}[/cyan]  {name}")
        sys.exit(1)

    seen = set()
    return [m for m in resolved if m not in seen and not seen.add(m)]


def _cmd_report(args, console: Console):
    """Generate a visual HTML report with charts."""
    _clear_and_banner(console)
    from crucible.report import generate_html_report

    files = args.results_files
    if not files:
        files = [str(f) for f in sorted(RESULTS_DIR.glob("*.json"))]
        if not files:
            console.print("[red]No result files found in results/[/red]")
            sys.exit(1)

    all_models, lookup = _load_compare_data(files, console)

    if not all_models:
        console.print("[red]No model data found.[/red]")
        sys.exit(1)

    # Filter models if -m specified
    if args.models:
        models = _filter_report_models(args.models, all_models, console)
        # Prune lookup to only selected models
        lookup = {k: v for k, v in lookup.items() if k[0] in models}
    else:
        models = all_models

    if args.suite:
        lookup = {
            k: v for k, v in lookup.items()
            if v.get("suite", "default") in args.suite
        }

    if not lookup:
        console.print("[red]No results matched the selected filters.[/red]")
        sys.exit(1)

    # Fetch model metadata from Ollama (if available)
    model_infos = {}
    try:
        from crucible.models import get_model_info
        for m in models:
            try:
                model_infos[m] = get_model_info(m)
            except Exception:
                pass  # Model not in Ollama — skip metadata
    except Exception:
        pass

    output_path = Path(args.output) if args.output else RESULTS_DIR / "report.html"
    path = generate_html_report(models, lookup, output_path, title=args.title,
                                model_infos=model_infos)
    console.print(f"[bold green]Report generated:[/bold green] {path}")

    # Auto-open in browser unless --no-open
    if not args.no_open:
        import webbrowser
        webbrowser.open(f"file://{path.resolve()}")
    else:
        console.print(f"[dim]Open in browser: file://{path.resolve()}[/dim]")


# ── main ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Crucible — LLM benchmark for structural engineering academics",
        prog="crucible",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- crucible list ---
    list_p = subparsers.add_parser("list", help="List models and/or tests")
    list_p.add_argument("what", nargs="?", default="all",
                        choices=["all", "models", "tests", "prompts", "results"],
                        help="What to list (default: all)")
    list_p.add_argument("--test-dir", default=None,
                        help="Test directory to load (default: tracked tests/)")
    list_p.add_argument("--suite", "-s", nargs="+",
                        help="Filter by suite (e.g. default tier2 personas)")
    list_p.add_argument("--gguf-dir", type=Path, default=None,
                        help="Directory containing GGUF model files (default: ~/models)")

    # --- crucible run ---
    run_p = subparsers.add_parser("run", help="Run tests against models")
    run_p.add_argument("--models", "-m", nargs="+")
    run_p.add_argument("--test-dir", default=None,
                       help="Test directory to load (default: tracked tests/)")
    run_p.add_argument("--suite", "-s", nargs="+",
                       help="Filter by suite (e.g. default tier2 personas)")
    run_p.add_argument("--category", "-c", nargs="+",
                       help="Filter by category (e.g. code_generation reasoning)")
    run_p.add_argument("--test", "-n", nargs="+",
                       help="Filter by test name (substring match, e.g. beam section)")
    run_p.add_argument("--no-save", action="store_true")
    run_p.add_argument("--interactive", "-i", action="store_true")
    run_p.add_argument("--timeout", type=float, default=600.0,
                       help="Max seconds per test — caps per-test YAML timeouts (default: 600)")
    run_p.add_argument("--tokens", type=int, default=0, dest="max_tokens",
                       help="Max completion tokens (0 = unlimited, default)")
    run_p.add_argument("--verbose", "-v", action="store_true",
                       help="Show full model responses after each test")
    run_p.add_argument("--fresh", action="store_true",
                       help="Ignore previous results and re-run all tests")
    run_p.add_argument("--gguf-dir", type=Path, default=None,
                       help="Directory containing GGUF model files (default: ~/models)")

    # --- crucible judge ---
    judge_p = subparsers.add_parser("judge", help="Score llm_judge tests via Opus")
    judge_sub = judge_p.add_subparsers(dest="judge_action")
    judge_p.add_argument("--test-dir", default=None,
                         help="Test directory to load definitions from (default: tracked tests/)")
    judge_p.add_argument("-m", "--models", nargs="+",
                         help="Filter models to judge (substring match)")
    judge_p.add_argument("-s", "--suite", nargs="+",
                         help="Filter suites to judge")
    judge_p.add_argument("-n", "--test", nargs="+",
                         help="Filter tests to judge (substring match)")

    # crucible judge review
    review_p = judge_sub.add_parser("review", help="Show judge reasoning for scored tests")
    review_p.add_argument("-m", "--models", nargs="+",
                          help="Filter models (substring match)")
    review_p.add_argument("-s", "--suite", nargs="+",
                          help="Filter suites")
    review_p.add_argument("-n", "--test", nargs="+",
                          help="Filter tests (substring match)")

    # --- crucible compare ---
    cmp_p = subparsers.add_parser("compare", help="Side-by-side response comparison")
    cmp_p.add_argument("results_files", nargs="*",
                       help="Result JSON files (default: all per-model files in results/)")
    cmp_p.add_argument("-m", "--models", nargs="+",
                       help="Models to compare (number, name, or substring)")
    cmp_p.add_argument("--suite", "-s", nargs="+",
                       help="Filter by suite")
    cmp_p.add_argument("--category", "-c", nargs="+",
                       help="Filter by category")
    cmp_p.add_argument("--test", "-n", nargs="+",
                       help="Filter by test name (substring match)")

    # --- crucible purge ---
    purge_p = subparsers.add_parser("purge", help="Remove selected stored results")
    purge_p.add_argument("-m", "--models", nargs="+",
                         help="Models to purge (number, name, or substring — default: all)")
    purge_p.add_argument("-s", "--suite", nargs="+",
                         help="Filter purge to suites")
    purge_p.add_argument("-c", "--category", nargs="+",
                         help="Filter purge to categories")
    purge_p.add_argument("-n", "--test", nargs="+",
                         help="Filter purge by test name or test_id substring")
    purge_p.add_argument("--yes", action="store_true",
                         help="Apply the purge (default is dry run)")

    # --- crucible register ---
    reg_p = subparsers.add_parser("register", help="Store API keys for providers")
    reg_p.add_argument("provider", help="Provider name (openrouter, anthropic)")
    reg_p.add_argument("key", nargs="?", default=None,
                       help="API key (omit to check status)")

    # --- crucible report ---
    report_p = subparsers.add_parser("report", help="Generate HTML visual report")
    report_p.add_argument("results_files", nargs="*",
                          help="Result JSON files (default: all per-model files in results/)")
    report_p.add_argument("-m", "--models", nargs="+",
                          help="Models to include (number, name, or substring — default: all)")
    report_p.add_argument("-s", "--suite", nargs="+",
                          help="Filter report to suites")
    report_p.add_argument("-o", "--output", default=None,
                          help="Output HTML path (default: results/report.html)")
    report_p.add_argument("--title", default="Crucible Benchmark Report",
                          help="Report title")
    report_p.add_argument("--no-open", action="store_true",
                          help="Don't auto-open in browser")

    # If no subcommand given, default to `run` (supports bare `crucible -m 1 2`)
    known_commands = {"list", "run", "judge", "compare", "purge", "register", "report"}
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
    elif args.command == "purge":
        _cmd_purge(args, console)
    elif args.command == "register":
        _cmd_register(args, console)
    elif args.command == "report":
        _cmd_report(args, console)
    else:
        _cmd_run(args, console)


if __name__ == "__main__":
    main()
