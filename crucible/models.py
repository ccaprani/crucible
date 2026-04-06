"""Ollama interaction layer with timing instrumentation."""

from collections.abc import Callable
from dataclasses import dataclass
import time

import ollama


@dataclass
class ModelResponse:
    """Result from a single model query."""

    model: str
    prompt: str
    response: str
    time_to_first_token: float  # seconds
    total_time: float  # seconds
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    timed_out: bool = False


def list_available_models() -> list[str]:
    """Return names of models available in Ollama."""
    models = ollama.list()
    return sorted(m.model for m in models.models)


def run_prompt(
    model: str,
    prompt: str,
    system: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout: float = 300.0,
    on_token: Callable[[int, float], None] | None = None,
) -> ModelResponse:
    """Send a prompt to an Ollama model and collect response with timing.

    Args:
        model: Ollama model name.
        prompt: User prompt text.
        system: Optional system prompt.
        temperature: Sampling temperature (0.0 for reproducibility).
        max_tokens: Maximum completion tokens (default 2048).
        timeout: Max wall-clock seconds per test (default 300 = 5 min).
        on_token: Callback(token_count, elapsed_seconds) called on each chunk
                  for live progress display.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    chunks: list[str] = []
    ttft: float | None = None
    token_count = 0
    timed_out = False
    t_start = time.perf_counter()

    stream = ollama.chat(
        model=model,
        messages=messages,
        stream=True,
        options={
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    )

    for chunk in stream:
        elapsed = time.perf_counter() - t_start
        if ttft is None:
            ttft = elapsed
        content = chunk.message.content
        if content:
            chunks.append(content)
            token_count += 1
            if on_token:
                on_token(token_count, elapsed)

        # Timeout check
        if elapsed > timeout:
            timed_out = True
            break

    t_total = time.perf_counter() - t_start
    full_response = "".join(chunks)

    prompt_tokens = _estimate_tokens(prompt + (system or ""))
    completion_tokens = _estimate_tokens(full_response)

    return ModelResponse(
        model=model,
        prompt=prompt,
        response=full_response,
        time_to_first_token=ttft or t_total,
        total_time=t_total,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        timed_out=timed_out,
    )


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token for English)."""
    return max(1, len(text) // 4)
