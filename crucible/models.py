"""Model interaction layer with timing instrumentation.

Supports Ollama, llama.cpp (via llama-server), OpenRouter, and CLI backends.
"""

from collections.abc import Callable
from dataclasses import dataclass
import base64
import os
import re
import time
from pathlib import Path

import ollama


# ── Image reference resolution ──────────────────────────────────────────
# Prompts may contain [Image: filename] references that need to be resolved
# to actual file paths and injected as image content for multimodal models.

_IMAGE_REF_PATTERN = re.compile(r"\[Image:\s*([^\]]+)\]")


def resolve_image_refs(
    prompt: str, search_dirs: list[Path] | None = None
) -> tuple[str, list[Path]]:
    """Parse [Image: filename] refs from a prompt and resolve to file paths.

    Returns:
        (cleaned_prompt, image_paths) where cleaned_prompt has the [Image: ...]
        tags replaced with descriptive text, and image_paths is a list of
        resolved Path objects for any found images.
    """
    if not search_dirs:
        search_dirs = []

    image_paths: list[Path] = []
    refs = _IMAGE_REF_PATTERN.findall(prompt)
    for ref in refs:
        ref = ref.strip()
        # Strip optional description after " —" or " -" (e.g. "file.jpg — caption")
        for sep in (" — ", " — ", " - "):
            if sep in ref:
                ref = ref.split(sep, 1)[0].strip()
                break
        # Try each search directory
        found = None
        for d in search_dirs:
            candidate = d / ref
            if candidate.exists():
                found = candidate
                break
            # Also check an images/ subdirectory
            candidate = d / "images" / ref
            if candidate.exists():
                found = candidate
                break
        if found:
            image_paths.append(found)

    # Replace tags with a note — the actual image is sent via the API
    cleaned = _IMAGE_REF_PATTERN.sub("[Attached image]", prompt)
    return cleaned, image_paths


def _encode_image_b64(path: Path) -> str:
    """Read an image file and return its base64 encoding."""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _image_media_type(path: Path) -> str:
    """Infer MIME type from file extension."""
    ext = path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")


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


def list_available_models(gguf_dir: Path | None = None) -> list[str]:
    """Return names of models available across all local backends.

    Combines Ollama models and GGUF files from *gguf_dir* (default ``~/models``).
    GGUF models are listed by stem name (e.g. ``Qwen3.5-9B-Q4_K_M``).
    """
    # Ollama models
    try:
        ollama_models = [m.model for m in ollama.list().models]
    except Exception:
        ollama_models = []

    # GGUF files
    from .llamacpp import list_gguf_models
    gguf_models = list_gguf_models(gguf_dir)

    return sorted(set(ollama_models + gguf_models))


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
    image_paths: list[Path] | None = None,
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
        image_paths: Optional list of image file paths to include (multimodal).
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})

    user_msg: dict = {"role": "user", "content": prompt}
    # Ollama multimodal: images are passed as base64 strings in the 'images' key
    if image_paths:
        user_msg["images"] = [_encode_image_b64(p) for p in image_paths]
    messages.append(user_msg)

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

    Routes (checked in order):
      - GGUF name / path                                   → llama-server
      - 'openai-codex/...'                                 → Codex CLI exec
      - 'google-gemini-cli/...'                             → Gemini CLI
      - Contains '/' (e.g. 'anthropic/claude-opus-4-6')    → OpenRouter
      - Starts with 'claude-' (e.g. 'claude-sonnet-4')     → Claude CLI -p
      - Starts with OpenAI prefix (gpt-, o1, o3, o4-)      → Codex CLI exec
      - Everything else                                     → Ollama (False)
    """
    return (
        _is_llamacpp_model(model)
        or _is_codex_model(model)
        or _is_gemini_cli_model(model)
        or _is_openrouter_model(model)
        or _is_claude_cli_model(model)
    )


def _is_openrouter_model(model: str) -> bool:
    """Models routed via OpenRouter API (provider/model format).

    Excludes CLI-prefixed models that use '/' as a namespace separator.
    """
    if "/" not in model:
        return False
    prefix = model.split("/", 1)[0].lower()
    # These prefixes denote CLI backends, not OpenRouter
    if prefix in ("openai-codex", "google-gemini-cli"):
        return False
    return True


def _is_claude_cli_model(model: str) -> bool:
    return model.lower().startswith("claude-") and "/" not in model


def _is_llamacpp_model(model: str, gguf_dir: Path | None = None) -> bool:
    """Models that should be served via llama-server.

    Matches:
      - Direct .gguf file paths (e.g. ``/home/user/models/Qwen3.5-9B-Q4_K_M.gguf``)
      - Names that resolve to a GGUF file in *gguf_dir* (default ``~/models``)
    """
    if model.endswith(".gguf"):
        return True
    from .llamacpp import resolve_gguf_path
    return resolve_gguf_path(model, gguf_dir) is not None


def _is_codex_model(model: str) -> bool:
    """Detect OpenAI models to be run via Codex CLI.

    Matches bare names: gpt-4o, gpt-4.1, o1, o3, o3-pro, o4-mini,
                        codex-mini-latest
    Matches prefixed:   openai-codex/gpt-5.3-codex, openai-codex/gpt-5.4
    """
    name = model.lower()
    # Prefixed form: openai-codex/model-name
    if name.startswith("openai-codex/"):
        return True
    # Bare form (no slash)
    if "/" in name:
        return False
    return (
        name.startswith("gpt-")
        or name.startswith("o1")
        or name.startswith("o3")
        or name.startswith("o4-")
        or name.startswith("codex-")
    )


def _is_gemini_cli_model(model: str) -> bool:
    """Detect Google models to be run via Gemini CLI.

    Matches: google-gemini-cli/gemini-3-flash-preview, etc.
    """
    return model.lower().startswith("google-gemini-cli/")


def _codex_model_name(model: str) -> str:
    """Extract the actual model name for the codex -m flag.

    'openai-codex/gpt-5.3-codex' → 'gpt-5.3-codex'
    'gpt-4.1'                     → 'gpt-4.1'
    """
    if model.lower().startswith("openai-codex/"):
        return model.split("/", 1)[1]
    return model


_IMAGE_MODELS = {
    "openai/gpt-5-image",
    "openai/gpt-5-image-mini",
    "google/gemini-3-pro-image-preview",
    "google/gemini-3.1-flash-image-preview",
    "google/gemini-2.5-flash-image",
}


def is_image_model(model: str) -> bool:
    """Return True for models that produce raster image output."""
    return model.lower() in _IMAGE_MODELS


def _gemini_model_name(model: str) -> str:
    """Extract the actual model name for the gemini CLI.

    'google-gemini-cli/gemini-3-flash-preview' → 'gemini-3-flash-preview'
    """
    if "/" in model:
        return model.split("/", 1)[1]
    return model


# Keep old name as alias for backwards compatibility
_is_cli_model = _is_claude_cli_model


def run_prompt_openrouter(
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    on_token: Callable[[int, float, str], None] | None = None,
    image_paths: list[Path] | None = None,
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

    # Build the user message — multimodal if images are attached
    def _build_user_message(text: str) -> dict:
        if not image_paths:
            return {"role": "user", "content": text}
        # OpenAI vision format: content is a list of parts
        parts: list[dict] = [{"type": "text", "text": text}]
        for img_path in image_paths:
            b64 = _encode_image_b64(img_path)
            mime = _image_media_type(img_path)
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
        return {"role": "user", "content": parts}

    # ── Image models: non-streaming, extract base64 image ────────────
    if is_image_model(model):
        result = client.chat.completions.create(
            model=model,
            messages=[_build_user_message(prompt)],
            max_tokens=max_tokens,
        )
        t_total = time.perf_counter() - t_start
        msg = result.choices[0].message
        msg_dict = msg.model_dump()

        # Text content (may be None for pure image responses)
        text_response = msg.content or ""

        # Extract image from .images[0].image_url.url (data:image/png;base64,...)
        images = msg_dict.get("images", [])
        output_image_paths: list[str] = []
        if images:
            img_dir = Path("results") / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(images):
                url = img.get("image_url", {}).get("url", "")
                if url.startswith("data:image/"):
                    # data:image/png;base64,<data>
                    header, b64data = url.split(",", 1)
                    ext = header.split("/")[1].split(";")[0]  # png, jpeg, etc.
                    safe_model = model.replace("/", "_")
                    fname = f"{safe_model}_{int(t_start)}_{i}.{ext}"
                    fpath = img_dir / fname
                    fpath.write_bytes(base64.b64decode(b64data))
                    output_image_paths.append(str(fpath))

        # Build response: text + image file references
        parts = []
        if text_response:
            parts.append(text_response)
        for p in output_image_paths:
            parts.append(f"[Image saved: {p}]")
        full_response = "\n".join(parts) if parts else "(no text or image output)"

        # Extract reasoning if present
        reasoning = msg_dict.get("reasoning", "")
        if reasoning:
            full_response = f"[Reasoning]\n{reasoning}\n\n[Response]\n{full_response}"

        usage = result.usage
        prompt_tokens = usage.prompt_tokens if usage else _estimate_tokens(prompt)
        completion_tokens = usage.completion_tokens if usage else 0

        if on_token and full_response:
            on_token(1, t_total, full_response[:80] + "...")

        return ModelResponse(
            model=model,
            prompt=prompt,
            response=full_response,
            time_to_first_token=t_total,
            total_time=t_total,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + (completion_tokens or 0),
            timed_out=False,
        )

    # ── Text models: streaming ───────────────────────────────────────
    chunks: list[str] = []
    token_count = 0
    ttft: float | None = None

    stream = client.chat.completions.create(
        model=model,
        messages=[_build_user_message(prompt)],
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


def run_prompt_llamacpp(
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    timeout: float = 300.0,
    on_token: Callable[[int, float, str], None] | None = None,
    image_paths: list[Path] | None = None,
) -> ModelResponse:
    """Run a prompt via a llama-server instance (OpenAI-compatible API).

    Assumes llama-server is already running (managed by ``LlamaCppServer``
    in the runner).  Connects to ``localhost:PORT/v1``.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError(
            "openai package required for llama.cpp backend. "
            "Install with: pip install openai"
        )

    from .llamacpp import _DEFAULT_PORT

    port = int(os.environ.get("CRUCIBLE_LLAMA_PORT", _DEFAULT_PORT))
    client = OpenAI(
        base_url=f"http://127.0.0.1:{port}/v1",
        api_key="not-needed",
    )

    t_start = time.perf_counter()

    # Build messages
    messages: list[dict] = []
    if image_paths:
        # OpenAI vision format
        parts: list[dict] = [{"type": "text", "text": prompt}]
        for img_path in image_paths:
            b64 = _encode_image_b64(img_path)
            mime = _image_media_type(img_path)
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
        messages.append({"role": "user", "content": parts})
    else:
        messages.append({"role": "user", "content": prompt})

    # Thinking models need a larger budget
    effective_max = max_tokens
    if _is_thinking_model(model):
        effective_max = max_tokens * _THINKING_BUDGET_MULTIPLIER

    chunks: list[str] = []
    thinking_chunks: list[str] = []
    token_count = 0
    thinking_token_count = 0
    ttft: float | None = None
    ttft_content: float | None = None
    timed_out = False

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=effective_max,
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

        if chunk.choices:
            delta = chunk.choices[0].delta

            # Check for thinking/reasoning content
            reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
            if reasoning:
                if ttft is None:
                    ttft = elapsed
                thinking_chunks.append(reasoning)
                thinking_token_count += 1
                if on_token:
                    on_token(-(thinking_token_count), elapsed, f"⟨think⟩{reasoning}")

            content = delta.content
            if content:
                if ttft is None:
                    ttft = elapsed
                if ttft_content is None:
                    ttft_content = elapsed
                chunks.append(content)
                token_count += 1
                if on_token:
                    on_token(token_count, elapsed, content)

        if elapsed > timeout:
            timed_out = True
            break

    t_total = time.perf_counter() - t_start
    full_response = "".join(chunks)
    full_thinking = "".join(thinking_chunks)

    # Fall back to estimates if usage not reported
    if not prompt_tokens:
        prompt_tokens = _estimate_tokens(prompt)
    if not completion_tokens:
        completion_tokens = _estimate_tokens(full_response)
    est_thinking_tokens = _estimate_tokens(full_thinking) if full_thinking else 0

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
        [claude_bin, "-p", "--model", model, "--tools", ""],
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


def run_prompt_codex(
    model: str,
    prompt: str,
    max_tokens: int = 4096,
    on_token: Callable[[int, float, str], None] | None = None,
) -> ModelResponse:
    """Run a prompt via OpenAI Codex CLI (exec mode).

    Uses the user's existing OpenAI login — no separate API key needed.
    The Codex CLI is invoked in non-interactive exec mode with output
    written to a temp file.
    """
    import shutil
    import subprocess
    import tempfile

    codex_bin = shutil.which("codex")
    if not codex_bin:
        raise RuntimeError(
            "Codex CLI not found on PATH. "
            "Install from https://github.com/openai/codex"
        )

    t_start = time.perf_counter()
    ttft: float | None = None

    # Use a temp file for the output — codex exec -o writes the last
    # agent message there, giving us clean text without JSONL noise.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as tmp:
        out_path = tmp.name

    try:
        actual_model = _codex_model_name(model)
        proc = subprocess.Popen(
            [
                codex_bin, "exec",
                "-m", actual_model,
                "--sandbox", "read-only",
                "-o", out_path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Send prompt via stdin and wait for completion.
        # communicate(input=...) handles write + close + wait in one call,
        # avoiding the "I/O operation on closed file" error.
        stdout_data, stderr_data = proc.communicate(input=prompt)
        t_total = time.perf_counter() - t_start
        ttft = t_total  # no streaming, so TTFT = total time

        # Read the -o output file (clean last message)
        full_response = ""
        try:
            with open(out_path) as f:
                full_response = f.read().strip()
        except (OSError, FileNotFoundError):
            pass

        # Fall back to stdout if output file is empty
        if not full_response and stdout_data:
            full_response = stdout_data.strip()

        if on_token and full_response:
            on_token(1, t_total, full_response[:80] + "...")

    finally:
        import pathlib
        pathlib.Path(out_path).unlink(missing_ok=True)

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


def run_prompt_gemini(
    model: str,
    prompt: str,
    max_tokens: int = 4096,
    on_token: Callable[[int, float, str], None] | None = None,
) -> ModelResponse:
    """Run a prompt via Google Gemini CLI.

    Uses the user's existing Google login — no API key needed.
    Prompt is piped via stdin; ``-p ""`` puts the CLI in non-interactive
    (headless) mode and appends stdin to the prompt.
    """
    import shutil
    import subprocess

    gemini_bin = shutil.which("gemini")
    if not gemini_bin:
        raise RuntimeError(
            "Gemini CLI not found on PATH. "
            "Install from https://github.com/google-gemini/gemini-cli"
        )

    actual_model = _gemini_model_name(model)
    t_start = time.perf_counter()
    chunks: list[str] = []
    token_count = 0
    ttft: float | None = None

    # Gemini CLI: pipe prompt on stdin, -p "" enables headless mode.
    proc = subprocess.Popen(
        [gemini_bin, "-m", actual_model, "-p", ""],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Send prompt and stream stdout character-by-character
    # (Gemini CLI streams output, unlike codex exec).
    proc.stdin.write(prompt)
    proc.stdin.close()

    buf = ""
    while True:
        ch = proc.stdout.read(1)
        if not ch:
            break
        elapsed = time.perf_counter() - t_start
        if ttft is None:
            ttft = elapsed
        buf += ch
        if ch in (" ", "\n", ".", ",", ";", ":", ")", "]", "}"):
            chunks.append(buf)
            token_count += 1
            if on_token:
                on_token(token_count, elapsed, buf)
            buf = ""

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
