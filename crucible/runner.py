"""Test runner — loads YAML tests, runs them against models, collects results."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .evaluate import EvalResult, evaluate
from .models import ModelResponse, run_prompt


@dataclass
class TestCase:
    """A single test definition loaded from YAML."""

    name: str
    description: str
    category: str
    prompt: str
    eval_type: str
    eval_config: dict = field(default_factory=dict)


@dataclass
class TestResult:
    """Result of running one test against one model."""

    test: TestCase
    model_response: ModelResponse
    eval_result: EvalResult


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
                )
            )

    return tests


def run_tests(
    models: list[str],
    tests: list[TestCase],
    on_result: Callable | None = None,
    on_token: Callable[[str, str, int, float], None] | None = None,
    max_tokens: int = 2048,
    timeout: float = 300.0,
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
            # Build a token callback that includes test context
            token_cb = None
            if on_token:
                def token_cb(count, elapsed, _m=model, _t=test.name):
                    on_token(_m, _t, count, elapsed)

            response = run_prompt(
                model=model,
                prompt=test.prompt,
                max_tokens=max_tokens,
                timeout=timeout,
                on_token=token_cb,
            )

            # If timed out, mark as failed
            if response.timed_out:
                eval_result = EvalResult(
                    0.0, False,
                    f"Timed out after {timeout:.0f}s ({response.completion_tokens} tokens generated)",
                )
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
