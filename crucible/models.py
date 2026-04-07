"""Ollama interaction layer with timing instrumentation."""

from collections.abc import Callable
from dataclasses import dataclass
import os
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


@dataclass
class ModelInfo:
    """Metadata about a model from Ollama."""
    name: str
    parameter_size: str  # e.g. "31.3B"
    quantization: str    # e.g. "Q4_K_M"
    family: str          # e.g. "gemma4"
    vram_gb: float       # estimated VRAM in GB
    vram_tier: str       # "8GB", "16GB", or "24GB"


def get_model_info(model: str) -> ModelInfo:
    """Fetch model metadata from Ollama and estimate VRAM tier.

    VRAM estimate = weight_size + overhead.
    Weight size uses bits-per-param for common quantisations.
    Overhead (~1.5-2 GB) covers KV cache at default context (2048),
    CUDA kernels, and runtime buffers.  Long-context workloads will
    use more, but tier assignment assumes typical benchmarking context.
    """
    info = ollama.show(model)
    details = info.get("details", {}) if isinstance(info, dict) else getattr(info, "details", None)
    if details and not isinstance(details, dict):
        # Pydantic model — grab attributes
        params = getattr(details, "parameter_size", "?")
        quant = getattr(details, "quantization_level", "?")
        family = getattr(details, "family", "?")
    elif isinstance(details, dict):
        params = details.get("parameter_size", "?")
        quant = details.get("quantization_level", "?")
        family = details.get("family", "?")
    else:
        params, quant, family = "?", "?", "?"

    # Bits per param for common quantisations
    _QUANT_BPP = {
        "Q4_0": 4.5, "Q4_1": 5.0, "Q4_K_S": 4.5, "Q4_K_M": 4.8,
        "Q5_0": 5.5, "Q5_1": 6.0, "Q5_K_S": 5.5, "Q5_K_M": 5.7,
        "Q6_K": 6.6, "Q8_0": 8.5, "F16": 16.0,
    }
    bpp = _QUANT_BPP.get(quant, 4.8)  # default to Q4_K_M

    try:
        param_b = float(params.replace("B", ""))
        weight_gb = param_b * bpp / 8  # bits → bytes → GB
        vram = weight_gb + 2.0  # runtime overhead
    except (ValueError, AttributeError):
        vram = 0.0

    if vram > 16:
        tier = "24GB"
    elif vram > 8:
        tier = "16GB"
    else:
        tier = "8GB"

    return ModelInfo(
        name=model,
        parameter_size=params,
        quantization=quant,
        family=family,
        vram_gb=round(vram, 1),
        vram_tier=tier,
    )


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
    on_token: Callable[[int, float, str], None] | None = None,
) -> ModelResponse:
    """Send a prompt to an Ollama model and collect response with timing.

    Args:
        model: Ollama model name.
        prompt: User prompt text.
        system: Optional system prompt.
        temperature: Sampling temperature (0.0 for reproducibility).
        max_tokens: Maximum completion tokens (default 2048).
        timeout: Max wall-clock seconds per test (default 300 = 5 min).
        on_token: Callback(token_count, elapsed_seconds, text_chunk) called on
                  each chunk for live progress display.  text_chunk is the new
                  content fragment (or thinking fragment prefixed with "⟨think⟩").
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
                on_token(token_count, elapsed, content)

        # For thinking models, report thinking progress too
        if not content and thinking and on_token:
            on_token(-(thinking_token_count), elapsed, f"⟨think⟩{thinking}")

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


# ── Non-Ollama model detection & runners ─────────────────────────────


def is_api_model(model: str) -> bool:
    """Check if a model should be run via a non-Ollama backend.

    Routes:
      - Contains '/' (e.g. 'anthropic/claude-opus-4-6') → OpenRouter
      - Starts with 'claude-' (e.g. 'claude-opus-4-6')  → Claude CLI -p
      - Everything else                                   → Ollama (False)
    """
    return "/" in model or model.lower().startswith("claude-")


def _is_openrouter_model(model: str) -> bool:
    return "/" in model


def _is_cli_model(model: str) -> bool:
    return model.lower().startswith("claude-") and "/" not in model


def run_prompt_openrouter(
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    on_token: Callable[[int, float, str], None] | None = None,
) -> ModelResponse:
    """Run a prompt via OpenRouter (OpenAI-compatible API).

    Requires OPENROUTER_API_KEY environment variable.
    """
    import os
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError(
            "openai package required for OpenRouter. "
            "Install with: pip install openai"
        )

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        # Try crucible's stored keys
        try:
            from crucible.run import get_api_key
            api_key = get_api_key("openrouter") or ""
        except Exception:
            pass
    if not api_key:
        raise RuntimeError(
            "No OpenRouter API key found. "
            "Run: crucible register openrouter <your-key> "
            "(get one at https://openrouter.ai/keys)"
        )

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    t_start = time.perf_counter()
    chunks: list[str] = []
    token_count = 0
    ttft: float | None = None

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
        stream_options={"include_usage": True},
    )

    prompt_tokens = 0
    completion_tokens = 0

    for chunk in stream:
        elapsed = time.perf_counter() - t_start
        if chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens or 0
            completion_tokens = chunk.usage.completion_tokens or 0
        if chunk.choices and chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            if ttft is None:
                ttft = elapsed
            chunks.append(text)
            token_count += 1
            if on_token:
                on_token(token_count, elapsed, text)

    t_total = time.perf_counter() - t_start
    full_response = "".join(chunks)

    # Fall back to estimates if usage not reported
    if not prompt_tokens:
        prompt_tokens = _estimate_tokens(prompt)
    if not completion_tokens:
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
        timed_out=False,
    )


def run_prompt_cli(
    model: str,
    prompt: str,
    max_tokens: int = 4096,
    on_token: Callable[[int, float, str], None] | None = None,
) -> ModelResponse:
    """Run a prompt via Claude Code CLI (--print mode).

    Uses the user's existing Claude subscription — no API key needed.
    Streams stdout for live display.
    """
    import shutil
    import subprocess

    claude_bin = shutil.which("claude")
    if not claude_bin:
        raise RuntimeError(
            "Claude Code CLI not found on PATH. "
            "Install from https://docs.anthropic.com/en/docs/claude-code"
        )

    t_start = time.perf_counter()
    chunks: list[str] = []
    token_count = 0
    ttft: float | None = None

    proc = subprocess.Popen(
        [claude_bin, "-p", "--model", model],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Send prompt and close stdin
    proc.stdin.write(prompt)
    proc.stdin.close()

    # Stream stdout character by character for live display
    buf = ""
    while True:
        ch = proc.stdout.read(1)
        if not ch:
            break
        elapsed = time.perf_counter() - t_start
        if ttft is None:
            ttft = elapsed
        buf += ch
        # Emit chunks on word boundaries or newlines for smoother display
        if ch in (" ", "\n", ".", ",", ";", ":", ")", "]", "}"):
            chunks.append(buf)
            token_count += 1
            if on_token:
                on_token(token_count, elapsed, buf)
            buf = ""

    # Flush remaining buffer
    if buf:
        elapsed = time.perf_counter() - t_start
        chunks.append(buf)
        token_count += 1
        if on_token:
            on_token(token_count, elapsed, buf)

    proc.wait()
    t_total = time.perf_counter() - t_start
    full_response = "".join(chunks)

    prompt_tokens = _estimate_tokens(prompt)
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
        timed_out=False,
    )
