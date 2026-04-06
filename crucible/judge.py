"""Async LLM judging — score previously captured responses via Claude Code CLI."""

import json
import shutil
import subprocess
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.table import Table

from .evaluate import EvalResult
from .runner import load_tests

TESTS_DIR = Path(__file__).resolve().parent.parent / "tests"

# Known Claude Code CLI locations
_CLAUDE_PATHS = [
    "claude",  # on PATH
    Path.home() / ".config" / "Claude" / "claude-code" / "2.1.87" / "claude",
    Path.home() / ".local" / "bin" / "claude",
]


def _find_claude() -> str | None:
    """Find the Claude Code CLI binary."""
    # Check PATH first
    if shutil.which("claude"):
        return "claude"
    # Check known locations
    for p in _CLAUDE_PATHS:
        path = Path(p)
        if path.exists() and path.is_file():
            return str(path)
    return None


def _judge_with_claude(
    prompt: str,
    response: str,
    criteria: str,
    claude_bin: str,
) -> EvalResult:
    """Score a response using Claude Code CLI (--print mode, no API tokens)."""
    judge_prompt = f"""You are an expert evaluator for AI model responses. Score the following response on a scale of 0-10.

ORIGINAL TASK:
{prompt}

EVALUATION CRITERIA:
{criteria}

RESPONSE TO EVALUATE:
{response}

Respond with ONLY valid JSON in this exact format:
{{"score": <0-10>, "reasoning": "<2-3 sentences explaining the score>"}}"""

    try:
        result = subprocess.run(
            [claude_bin, "--print"],
            input=judge_prompt,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            return EvalResult(
                0.5, True,
                f"Claude CLI error (exit {result.returncode}): {result.stderr[:200]}",
            )

        text = result.stdout.strip()

        # Extract JSON — might be wrapped in markdown fences
        import re
        json_match = re.search(r"\{[^}]+\}", text)
        if json_match:
            parsed = json.loads(json_match.group())
        else:
            parsed = json.loads(text)

        score_raw = parsed["score"]
        reasoning = parsed["reasoning"]
        score = max(0.0, min(1.0, score_raw / 10.0))
        return EvalResult(
            score,
            score >= 0.6,
            f"Claude judge: {score_raw}/10 — {reasoning}",
        )
    except subprocess.TimeoutExpired:
        return EvalResult(0.5, True, "Claude CLI timed out (120s)")
    except (json.JSONDecodeError, KeyError) as e:
        return EvalResult(0.5, True, f"Judge parse error: {e}. Raw: {text[:200]}")
    except Exception as e:
        return EvalResult(0.5, True, f"Judge error: {e}")


def judge_results(
    results_path: Path,
    console: Console | None = None,
    test_names: list[str] | None = None,
) -> Path:
    """Read a results JSON, run llm_judge on applicable tests, write updated file."""
    if console is None:
        console = Console()

    # Find Claude Code CLI
    claude_bin = _find_claude()
    if claude_bin is None:
        console.print(
            "[red]Claude Code CLI not found.[/red]\n"
            "Install it or add it to your PATH."
        )
        raise SystemExit(1)

    console.print(f"[dim]Using: {claude_bin}[/dim]")

    # Load results
    with open(results_path) as f:
        data = json.load(f)

    # Load test definitions to get llm_judge criteria
    all_tests = load_tests(TESTS_DIR)
    judge_tests = {
        t.name: t for t in all_tests if t.eval_type == "llm_judge"
    }

    if not judge_tests:
        console.print("[yellow]No llm_judge tests found in test definitions.[/yellow]")
        return results_path

    # Apply test name filter if provided
    if test_names:
        judge_tests = {
            name: t for name, t in judge_tests.items()
            if any(n.lower() in name.lower() for n in test_names)
        }

    # Count what needs judging
    to_judge = []
    for model, model_results in data["models"].items():
        for result in model_results:
            test_name = result["test"]
            if test_name in judge_tests:
                to_judge.append((model, result, judge_tests[test_name]))

    if not to_judge:
        console.print("[yellow]No matching responses found to judge.[/yellow]")
        return results_path

    console.print(
        f"\n[bold]Judging {len(to_judge)} responses via Claude Code[/bold]\n"
    )

    completed = 0
    result_lines: list[str] = []

    def build_table() -> Table:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("label", style="dim", width=12)
        table.add_column("value")
        table.add_row("Progress", f"[bold]{completed}/{len(to_judge)}[/bold]")
        for line in result_lines[-5:]:
            table.add_row("", line)
        return table

    with Live(build_table(), console=console, refresh_per_second=1) as live:
        for model, result, test_def in to_judge:
            eval_result = _judge_with_claude(
                test_def.prompt,
                result["response"],
                test_def.eval_config.get("criteria", "overall quality"),
                claude_bin,
            )

            result["judge_score"] = eval_result.score
            result["judge_passed"] = eval_result.passed
            result["judge_details"] = eval_result.details

            completed += 1
            icon = "✓" if eval_result.passed else "✗"
            style = "green" if eval_result.passed else "red"
            result_lines.append(
                f"[{style}]{icon}[/{style}]  {model} / {result['test']} — "
                f"{eval_result.score:.0%}"
            )
            live.update(build_table())

    # Write updated results
    output_path = results_path.with_stem(results_path.stem + "_judged")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    console.print(f"\n[bold green]Done.[/bold green] Judged results: {output_path}")
    _print_judge_summary(data, judge_tests, console)

    return output_path


def _print_judge_summary(data: dict, judge_tests: dict, console: Console):
    """Print a summary table of judge scores."""
    models = list(data["models"].keys())

    table = Table(
        title="Claude Judge Scores",
        show_header=True,
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Test", style="bold")
    for model in models:
        table.add_column(model, justify="center")

    for test_name in judge_tests:
        row = [test_name]
        for model in models:
            for result in data["models"][model]:
                if result["test"] == test_name and "judge_score" in result:
                    score = result["judge_score"]
                    style = "green" if result["judge_passed"] else "red"
                    row.append(f"[{style}]{score:.0%}[/{style}]")
                    break
            else:
                row.append("—")
        table.add_row(*row)

    console.print()
    console.print(table)
