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
    thinking: str = ""  # internal reasoning (thinking models only)
    thinking_tokens: int = 0


def list_available_models() -> list[str]:
    """Return names of models available in Ollama."""
    models = ollama.list()
    return sorted(m.model for m in models.models)


def _is_thinking_model(model: str) -> bool:
    """Detect models that use internal chain-of-thought reasoning.

    These models emit thinking tokens *before* content tokens, both counted
    against num_predict.  We inflate the budget so they don't run out of
    tokens before producing a visible answer.
    """
    name = model.lower()
    # Known thinking model families
    _THINKING_PATTERNS = ("qwen3", "deepseek-r1", "qwq", "gemma4")
    return any(p in name for p in _THINKING_PATTERNS)


# Thinking models burn tokens on internal reasoning before producing
# visible content.  This multiplier ensures num_predict is large enough
# for both the thinking and the answer.
_THINKING_BUDGET_MULTIPLIER = 4


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
    thinking_chunks: list[str] = []
    ttft: float | None = None
    ttft_content: float | None = None  # time to first *content* token
    token_count = 0
    thinking_token_count = 0
    timed_out = False
    t_start = time.perf_counter()

    # Build options for ollama
    opts: dict = {"temperature": temperature}

    if max_tokens > 0:
        # Thinking models need a larger token budget since internal reasoning
        # counts against num_predict before any visible content is produced.
        effective_max = max_tokens
        if _is_thinking_model(model):
            effective_max = max_tokens * _THINKING_BUDGET_MULTIPLIER
        opts["num_predict"] = effective_max
    # max_tokens == 0 → omit num_predict entirely, let the model finish naturally

    stream = ollama.chat(
        model=model,
        messages=messages,
        stream=True,
        options=opts,
    )

    for chunk in stream:
        elapsed = time.perf_counter() - t_start
        if ttft is None:
            ttft = elapsed

        # Thinking models emit reasoning via message.thinking
        thinking = getattr(chunk.message, "thinking", None)
        if thinking:
            thinking_chunks.append(thinking)
            thinking_token_count += 1

        content = chunk.message.content
        if content:
            chunks.append(content)
            token_count += 1
            if ttft_content is None:
                ttft_content = elapsed
            if on_token:
                on_token(token_count, elapsed)

        # For thinking models, report thinking progress too
        if not content and thinking and on_token:
            on_token(-(thinking_token_count), elapsed)

        # Timeout check
        if elapsed > timeout:
            timed_out = True
            break

    t_total = time.perf_counter() - t_start
    full_response = "".join(chunks)
    full_thinking = "".join(thinking_chunks)

    prompt_tokens = _estimate_tokens(prompt + (system or ""))
    completion_tokens = _estimate_tokens(full_response)
    est_thinking_tokens = _estimate_tokens(full_thinking)

    return ModelResponse(
        model=model,
        prompt=prompt,
        response=full_response,
        time_to_first_token=ttft_content or ttft or t_total,
        total_time=t_total,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens + est_thinking_tokens,
        timed_out=timed_out,
        thinking=full_thinking,
        thinking_tokens=est_thinking_tokens,
    )


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token for English)."""
    if not text:
        return 0
    return max(1, len(text) // 4)
