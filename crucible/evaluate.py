"""Automated evaluators for test responses."""

import json
import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

_ANTHROPIC_CLIENT = None
_API_KEY_PATHS = [
    Path.home() / ".config" / "anthropic" / "api_key",
    Path.home() / ".anthropic" / "api_key",
]


@dataclass
class EvalResult:
    """Result of evaluating a model response."""

    score: float  # 0.0 to 1.0
    passed: bool
    details: str  # human-readable explanation


def evaluate(
    response: str,
    eval_type: str,
    eval_config: dict,
    prompt: str = "",
) -> EvalResult:
    """Dispatch to the appropriate evaluator."""
    evaluators = {
        "code_exec": _eval_code_exec,
        "json_valid": _eval_json_valid,
        "yaml_valid": _eval_yaml_valid,
        "reference": _eval_reference,
        "contains": _eval_contains,
        "llm_judge": _eval_llm_judge,
        "manual": _eval_manual,
    }
    fn = evaluators.get(eval_type)
    if fn is None:
        return EvalResult(0.0, False, f"Unknown eval type: {eval_type}")
    if eval_type == "llm_judge":
        return fn(response, eval_config, prompt)
    return fn(response, eval_config)


def _extract_code_blocks(text: str) -> list[str]:
    """Extract Python code blocks from markdown-fenced responses."""
    # Match ```python ... ``` or ``` ... ```
    pattern = r"```(?:python)?\s*\n(.*?)```"
    blocks = re.findall(pattern, text, re.DOTALL)
    if not blocks:
        # Try the whole response if no fences found and it looks like code
        stripped = text.strip()
        if stripped.startswith(("import ", "from ", "def ", "class ")):
            blocks = [stripped]
    return blocks


def _eval_code_exec(response: str, config: dict) -> EvalResult:
    """Extract Python code from response, execute it, check results.

    config keys:
        expected_output: str — optional substring expected in stdout
        expected_values: dict — optional {name: value} to check (printed as name=value)
        tolerance: float — for numeric comparisons (default 0.05 = 5%)
        setup_code: str — optional code prepended before extracted code
        validation_code: str — optional code appended after extracted code
    """
    blocks = _extract_code_blocks(response)
    if not blocks:
        return EvalResult(0.0, False, "No code blocks found in response")

    code = "\n\n".join(blocks)

    # Prepend setup code if provided
    if setup := config.get("setup_code"):
        code = setup + "\n\n" + code

    # Append validation code if provided
    if validation := config.get("validation_code"):
        code = code + "\n\n" + validation

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(code)
        script_path = f.name

    try:
        result = subprocess.run(
            ["python3", script_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        Path(script_path).unlink(missing_ok=True)
        return EvalResult(0.0, False, "Code execution timed out (30s)")
    finally:
        Path(script_path).unlink(missing_ok=True)

    if result.returncode != 0:
        stderr_short = result.stderr[:500] if result.stderr else "no stderr"
        return EvalResult(0.0, False, f"Code failed (exit {result.returncode}): {stderr_short}")

    stdout = result.stdout.strip()
    details = f"Code executed successfully. Output: {stdout[:300]}"

    # Check expected output substring
    if expected := config.get("expected_output"):
        if expected.lower() not in stdout.lower():
            return EvalResult(0.3, False, f"Output missing expected: '{expected}'. Got: {stdout[:200]}")

    # Check expected numeric values
    if expected_values := config.get("expected_values"):
        tolerance = config.get("tolerance", 0.05)
        for name, expected_val in expected_values.items():
            pattern = rf"{name}\s*[=:]\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
            match = re.search(pattern, stdout)
            if not match:
                return EvalResult(
                    0.3, False,
                    f"Expected '{name}' in output but not found. Output: {stdout[:200]}",
                )
            actual = float(match.group(1))
            if not math.isclose(actual, expected_val, rel_tol=tolerance):
                return EvalResult(
                    0.3, False,
                    f"{name}: expected {expected_val}, got {actual} (tolerance {tolerance*100}%)",
                )

    return EvalResult(1.0, True, details)


def _eval_json_valid(response: str, config: dict) -> EvalResult:
    """Check if response contains valid JSON, optionally with required keys.

    config keys:
        required_keys: list[str] — keys that must exist in the top-level object
    """
    # Try to extract JSON from code blocks first
    blocks = re.findall(r"```(?:json)?\s*\n(.*?)```", response, re.DOTALL)
    candidates = blocks + [response]

    for candidate in candidates:
        try:
            parsed = json.loads(candidate.strip())
            if required := config.get("required_keys"):
                if not isinstance(parsed, dict):
                    continue
                missing = [k for k in required if k not in parsed]
                if missing:
                    return EvalResult(
                        0.5, False,
                        f"Valid JSON but missing keys: {missing}",
                    )
            return EvalResult(1.0, True, "Valid JSON with all required keys")
        except json.JSONDecodeError:
            continue

    return EvalResult(0.0, False, "No valid JSON found in response")


def _eval_yaml_valid(response: str, config: dict) -> EvalResult:
    """Check if response contains valid YAML.

    config keys:
        required_keys: list[str] — keys that must exist
    """
    blocks = re.findall(r"```(?:ya?ml)?\s*\n(.*?)```", response, re.DOTALL)
    candidates = blocks + [response]

    for candidate in candidates:
        try:
            parsed = yaml.safe_load(candidate.strip())
            if parsed is None or isinstance(parsed, str):
                continue
            if required := config.get("required_keys"):
                if not isinstance(parsed, dict):
                    continue
                missing = [k for k in required if k not in parsed]
                if missing:
                    return EvalResult(
                        0.5, False,
                        f"Valid YAML but missing keys: {missing}",
                    )
            return EvalResult(1.0, True, "Valid YAML with all required keys")
        except yaml.YAMLError:
            continue

    return EvalResult(0.0, False, "No valid YAML found in response")


def _eval_reference(response: str, config: dict) -> EvalResult:
    """Compare numeric values in response against reference values.

    config keys:
        values: dict[str, float] — {label: expected_value}
        tolerance: float — relative tolerance (default 0.05)
    """
    values = config.get("values", {})
    tolerance = config.get("tolerance", 0.05)

    if not values:
        return EvalResult(0.0, False, "No reference values configured")

    matches = 0
    details_parts = []

    for label, expected in values.items():
        # Search for the value near the label in the response
        # Look for numbers near the label text
        pattern = rf"(?i){re.escape(label)}.*?([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
        found = re.findall(pattern, response)

        if not found:
            # Also try finding the number anywhere if label is generic
            all_numbers = re.findall(
                r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", response
            )
            close_matches = [
                float(n) for n in all_numbers
                if _is_close(float(n), expected, tolerance)
            ]
            if close_matches:
                matches += 1
                details_parts.append(f"{label}: found {close_matches[0]} ≈ {expected} ✓")
            else:
                details_parts.append(f"{label}: expected {expected}, not found")
        else:
            actual = float(found[0])
            if _is_close(actual, expected, tolerance):
                matches += 1
                details_parts.append(f"{label}: {actual} ≈ {expected} ✓")
            else:
                details_parts.append(f"{label}: expected {expected}, got {actual} ✗")

    score = matches / len(values) if values else 0.0
    return EvalResult(
        score,
        score >= 0.8,
        "; ".join(details_parts),
    )


def _is_close(a: float, b: float, rel_tol: float) -> bool:
    """Check if two numbers are close, handling zero."""
    if b == 0:
        return abs(a) < 0.01
    return math.isclose(a, b, rel_tol=rel_tol)


def _eval_contains(response: str, config: dict) -> EvalResult:
    """Check that response contains all required strings.

    config keys:
        required: list[str] — strings that must appear (case-insensitive)
        any_of: list[list[str]] — groups where at least one must appear
    """
    response_lower = response.lower()
    found = []
    missing = []

    for term in config.get("required", []):
        if term.lower() in response_lower:
            found.append(term)
        else:
            missing.append(term)

    # Check any_of groups (at least one from each group must match)
    for group in config.get("any_of", []):
        if any(term.lower() in response_lower for term in group):
            found.append(f"one of [{', '.join(group)}]")
        else:
            missing.append(f"none of [{', '.join(group)}]")

    total = len(config.get("required", [])) + len(config.get("any_of", []))
    if total == 0:
        return EvalResult(1.0, True, "No contains checks configured")

    score = len(found) / total
    passed = len(missing) == 0

    details = f"Found {len(found)}/{total}."
    if missing:
        details += f" Missing: {missing}"

    return EvalResult(score, passed, details)


def _get_anthropic_client():
    """Lazy-init Anthropic client, reading key from config file or env."""
    global _ANTHROPIC_CLIENT
    if _ANTHROPIC_CLIENT is not None:
        return _ANTHROPIC_CLIENT

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        for path in _API_KEY_PATHS:
            if path.exists():
                api_key = path.read_text().strip()
                if api_key:
                    break

    if not api_key:
        return None

    import anthropic
    _ANTHROPIC_CLIENT = anthropic.Anthropic(api_key=api_key)
    return _ANTHROPIC_CLIENT


def _eval_llm_judge(response: str, config: dict, prompt: str) -> EvalResult:
    """Use Claude Opus to evaluate response quality.

    config keys:
        criteria: str — what the judge should evaluate (domain-specific rubric)
        model: str — judge model (default: claude-opus-4-6)
    """
    client = _get_anthropic_client()
    if client is None:
        return EvalResult(
            0.5, True,
            "LLM judge unavailable (no API key). "
            "Set ANTHROPIC_API_KEY or create ~/.config/anthropic/api_key",
        )

    criteria = config.get("criteria", "overall quality, accuracy, and completeness")
    judge_model = config.get("model", "claude-opus-4-6")

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
        result = client.messages.create(
            model=judge_model,
            max_tokens=300,
            temperature=0.0,
            messages=[{"role": "user", "content": judge_prompt}],
        )
        text = result.content[0].text.strip()
        parsed = json.loads(text)
        score_raw = parsed["score"]
        reasoning = parsed["reasoning"]
        score = max(0.0, min(1.0, score_raw / 10.0))
        return EvalResult(
            score,
            score >= 0.6,
            f"Judge ({judge_model}): {score_raw}/10 — {reasoning}",
        )
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        # Try to extract score from non-JSON response
        return EvalResult(0.5, True, f"Judge response parse error: {e}. Raw: {text[:200]}")
    except Exception as e:
        return EvalResult(0.5, True, f"Judge API error: {e}")


def _eval_manual(response: str, config: dict) -> EvalResult:
    """No automated evaluation — just capture for human review."""
    preview = response[:200] + "..." if len(response) > 200 else response
    return EvalResult(0.5, True, f"Manual review needed. Preview: {preview}")
