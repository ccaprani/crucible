"""Test runner — loads YAML tests, runs them against models, collects results."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .evaluate import EvalResult, evaluate
from .models import (
    ModelResponse, is_api_model, run_prompt,
    _is_openrouter_model, _is_cli_model,
    run_prompt_openrouter, run_prompt_cli,
)


@dataclass
class TestCase:
    """A single test definition loaded from YAML."""

    name: str
    description: str
    category: str
    prompt: str
    eval_type: str
    eval_config: dict = field(default_factory=dict)
    timeout: float = 0  # per-test timeout in seconds (0 = use global default)


@dataclass
class TestResult:
    """Result of running one test against one model."""

    test: TestCase
    model_response: ModelResponse
    eval_result: EvalResult
    reused: bool = False


def load_tests(test_dir: Path, categories: list[str] | None = None) -> list[TestCase]:
    """Load test definitions from YAML files in test_dir.

    If categories is provided, only load tests from matching files.
    """
    tests = []
    for yaml_file in sorted(test_dir.glob("*.yaml")):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)

        category = data.get("category", yaml_file.stem)
        if categories and category not in categories:
            continue

        for test_data in data.get("tests", []):
            tests.append(
                TestCase(
                    name=test_data["name"],
                    description=test_data.get("description", ""),
                    category=category,
                    prompt=test_data["prompt"],
                    eval_type=test_data.get("eval_type", "manual"),
                    eval_config=test_data.get("eval_config", {}),
                    timeout=test_data.get("timeout", 0),
                )
            )

    return tests


def _rebuild_test_result(test: TestCase, prev: dict, model: str) -> TestResult:
    """Reconstruct a TestResult from a cached JSON result entry."""
    model_response = ModelResponse(
        model=model,
        prompt=prev.get("prompt", test.prompt),
        response=prev.get("response", ""),
        time_to_first_token=prev.get("ttft", 0.0),
        total_time=prev.get("total_time", 0.0),
        prompt_tokens=prev.get("prompt_tokens", 0),
        completion_tokens=prev.get("completion_tokens", 0),
        total_tokens=prev.get("prompt_tokens", 0) + prev.get("completion_tokens", 0),
        timed_out=False,
        thinking_tokens=prev.get("thinking_tokens", 0),
    )
    eval_result = EvalResult(
        score=prev.get("score", 0.0),
        passed=prev.get("passed", False),
        details=prev.get("eval_details", ""),
    )
    return TestResult(
        test=test,
        model_response=model_response,
        eval_result=eval_result,
        reused=True,
    )


def run_tests(
    models: list[str],
    tests: list[TestCase],
    on_result: Callable | None = None,
    on_token: Callable[[str, str, int, float], None] | None = None,
    max_tokens: int = 2048,
    timeout: float = 300.0,
    previous_results: dict[tuple[str, str], dict] | None = None,
    on_skip: Callable | None = None,
) -> dict[str, list[TestResult]]:
    """Run all tests against all models.

    Args:
        models: List of Ollama model names.
        tests: List of test cases to run.
        on_result: Callback(model, test_name, TestResult) after each test.
        on_token: Callback(model, test_name, token_count, elapsed) during generation.
        max_tokens: Max completion tokens per test.
        timeout: Max seconds per test.
    """
    results: dict[str, list[TestResult]] = {m: [] for m in models}

    for test in tests:
        for model in models:
            # Reuse previous passing result if available
            if previous_results and (model, test.name) in previous_results:
                prev = previous_results[(model, test.name)]
                result = _rebuild_test_result(test, prev, model)
                results[model].append(result)
                if on_skip:
                    on_skip(model, test.name, result)
                continue

            # Build a token callback that includes test context
            token_cb = None
            if on_token:
                def token_cb(count, elapsed, text="", _m=model, _t=test.name):
                    on_token(_m, _t, count, elapsed, text)

            # Per-test timeout from YAML, but CLI --timeout acts as a cap
            effective_timeout = test.timeout if test.timeout > 0 else timeout
            if timeout > 0:
                effective_timeout = min(effective_timeout, timeout)

            if _is_openrouter_model(model):
                response = run_prompt_openrouter(
                    model=model,
                    prompt=test.prompt,
                    max_tokens=max_tokens or 4096,
                    on_token=token_cb,
                )
            elif _is_cli_model(model):
                response = run_prompt_cli(
                    model=model,
                    prompt=test.prompt,
                    max_tokens=max_tokens or 4096,
                    on_token=token_cb,
                )
            else:
                response = run_prompt(
                    model=model,
                    prompt=test.prompt,
                    max_tokens=max_tokens,
                    timeout=effective_timeout,
                    on_token=token_cb,
                )

            # If timed out, mark as failed
            if response.timed_out:
                eval_result = EvalResult(
                    0.0, False,
                    f"Timed out after {effective_timeout:.0f}s ({response.completion_tokens} tokens generated)",
                )
            elif is_api_model(model) and test.eval_type == "llm_judge":
                # Don't judge an API model with itself — mark as reference
                eval_result = EvalResult(1.0, True, "Reference answer (API model)")
            else:
                eval_result = evaluate(
                    response.response, test.eval_type, test.eval_config,
                    prompt=test.prompt,
                )

            result = TestResult(
                test=test,
                model_response=response,
                eval_result=eval_result,
            )
            results[model].append(result)

            if on_result:
                on_result(model, test.name, result)

    return results
