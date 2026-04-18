"""Test runner — loads YAML tests, runs them against models, collects results."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .evaluate import EvalResult, evaluate
from .models import (
    ModelResponse, is_api_model, run_prompt,
    _is_openrouter_model, _is_cli_model, _is_codex_model,
    _is_gemini_cli_model, _is_llamacpp_model,
    run_prompt_openrouter, run_prompt_cli, run_prompt_codex,
    run_prompt_gemini, run_prompt_llamacpp,
    resolve_image_refs,
)
from .llamacpp import LlamaCppServer


@dataclass
class TestCase:
    """A single test definition loaded from YAML."""

    suite: str
    name: str
    description: str
    category: str
    prompt: str
    eval_type: str
    eval_config: dict = field(default_factory=dict)
    timeout: float = 0  # per-test timeout in seconds (0 = use global default)

    @property
    def test_id(self) -> str:
        """Stable identity for storage/reporting across suites and categories."""
        return f"{self.suite}/{self.category}/{self.name}"


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
    for yaml_file in sorted(test_dir.rglob("*.yaml")):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)

        rel_parent = yaml_file.parent.relative_to(test_dir)
        inferred_suite = str(rel_parent).replace("\\", "/")
        if inferred_suite == ".":
            inferred_suite = "default"

        suite = data.get("suite", inferred_suite)
        category = data.get("category", yaml_file.stem)
        if categories and category not in categories:
            continue

        for test_data in data.get("tests", []):
            tests.append(
                TestCase(
                    suite=suite,
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


def _run_single(
    model: str,
    test: TestCase,
    effective_prompt: str,
    image_files: list[Path],
    max_tokens: int,
    timeout: float,
    token_cb: Callable | None,
) -> ModelResponse:
    """Dispatch a single (model, test) pair to the appropriate backend."""
    if _is_llamacpp_model(model):
        return run_prompt_llamacpp(
            model=model,
            prompt=effective_prompt,
            max_tokens=max_tokens or 4096,
            timeout=timeout,
            on_token=token_cb,
            image_paths=image_files or None,
        )
    elif _is_openrouter_model(model):
        return run_prompt_openrouter(
            model=model,
            prompt=effective_prompt,
            max_tokens=max_tokens or 4096,
            on_token=token_cb,
            image_paths=image_files or None,
        )
    elif _is_codex_model(model):
        return run_prompt_codex(
            model=model,
            prompt=effective_prompt,
            max_tokens=max_tokens or 4096,
            on_token=token_cb,
        )
    elif _is_gemini_cli_model(model):
        return run_prompt_gemini(
            model=model,
            prompt=effective_prompt,
            max_tokens=max_tokens or 4096,
            on_token=token_cb,
        )
    elif _is_cli_model(model):
        return run_prompt_cli(
            model=model,
            prompt=effective_prompt,
            max_tokens=max_tokens or 4096,
            on_token=token_cb,
        )
    else:
        return run_prompt(
            model=model,
            prompt=effective_prompt,
            max_tokens=max_tokens,
            timeout=timeout,
            on_token=token_cb,
            image_paths=image_files or None,
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
    test_dir: Path | None = None,
    gguf_dir: Path | None = None,
) -> dict[str, list[TestResult]]:
    """Run all tests against all models.

    Iterates model-first so that each local model is loaded once and all
    tests run against it before moving on — avoids costly model swaps.

    Args:
        models: List of model names (Ollama, llama.cpp GGUF, or API).
        tests: List of test cases to run.
        on_result: Callback(model, test_name, TestResult) after each test.
        on_token: Callback(model, test_name, token_count, elapsed) during generation.
        max_tokens: Max completion tokens per test.
        timeout: Max seconds per test.
        test_dir: Base directory for resolving image references in prompts.
        gguf_dir: Directory containing GGUF model files (default ``~/models``).
    """
    # Build image search directories
    _image_search_dirs: list[Path] = []
    if test_dir:
        _image_search_dirs.append(test_dir)

    # Pre-resolve image refs for each test (prompt-level, model-independent)
    resolved_prompts: list[tuple[str, list[Path]]] = []
    for test in tests:
        cleaned_prompt, image_files = resolve_image_refs(
            test.prompt, _image_search_dirs
        )
        resolved_prompts.append((cleaned_prompt, image_files))

    results: dict[str, list[TestResult]] = {m: [] for m in models}

    for model in models:
        # Start llama-server for GGUF models (no-op for other backends)
        server = None
        if _is_llamacpp_model(model, gguf_dir):
            server = LlamaCppServer(model, gguf_dir=gguf_dir)
            server.start()

        try:
            for test, (cleaned_prompt, image_files) in zip(tests, resolved_prompts):
                # Reuse previous passing result if available
                if previous_results and (model, test.test_id) in previous_results:
                    prev = previous_results[(model, test.test_id)]
                    result = _rebuild_test_result(test, prev, model)
                    results[model].append(result)
                    if on_skip:
                        on_skip(model, test.test_id, result)
                    continue

                # Build a token callback that includes test context
                token_cb = None
                if on_token:
                    def token_cb(count, elapsed, text="", _m=model, _t=test.test_id):
                        on_token(_m, _t, count, elapsed, text)

                # Per-test timeout from YAML, but CLI --timeout acts as a cap
                effective_timeout = test.timeout if test.timeout > 0 else timeout
                if timeout > 0:
                    effective_timeout = min(effective_timeout, timeout)

                # Use cleaned prompt (image refs replaced) and pass images
                effective_prompt = cleaned_prompt if image_files else test.prompt

                response = _run_single(
                    model, test, effective_prompt, image_files,
                    max_tokens, effective_timeout, token_cb,
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
                    on_result(model, test.test_id, result)
        finally:
            if server:
                server.stop()

    return results
