"""llama.cpp server lifecycle management.

Starts and stops a ``llama-server`` process for running GGUF models via the
OpenAI-compatible ``/v1/chat/completions`` endpoint.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# Default port for the llama-server instance managed by crucible.
# Chosen to avoid clashing with common services (Ollama 11434, etc.)
_DEFAULT_PORT = 8794

# How long to wait for llama-server to become healthy (seconds).
_STARTUP_TIMEOUT = 120

# Default directory for GGUF model files.
DEFAULT_GGUF_DIR = Path.home() / "models"


def find_llama_server() -> str | None:
    """Locate the llama-server binary.

    Checks, in order:
      1. ``CRUCIBLE_LLAMA_SERVER`` environment variable (for non-standard installs)
      2. ``llama-server`` on ``$PATH``
    """
    env = os.environ.get("CRUCIBLE_LLAMA_SERVER", "").strip()
    if env and os.path.isfile(env) and os.access(env, os.X_OK):
        return env
    return shutil.which("llama-server")


def list_gguf_models(gguf_dir: Path | None = None) -> list[str]:
    """Return display names for GGUF files found in *gguf_dir*.

    Args:
        gguf_dir: Directory to scan. Defaults to ``~/models``.
    """
    d = gguf_dir or DEFAULT_GGUF_DIR
    if not d.is_dir():
        return []
    return sorted(
        p.stem for p in d.glob("*.gguf") if p.is_file()
    )


def resolve_gguf_path(name: str, gguf_dir: Path | None = None) -> Path | None:
    """Resolve a model name to a GGUF file path.

    Args:
        name: A direct ``.gguf`` file path, an exact stem, or a
              case-insensitive substring (must be unique).
        gguf_dir: Directory to search. Defaults to ``~/models``.

    Returns:
        Resolved ``Path`` or ``None`` if not found.
    """
    # Direct path
    p = Path(name)
    if p.suffix == ".gguf" and p.is_file():
        return p

    d = gguf_dir or DEFAULT_GGUF_DIR
    if not d.is_dir():
        return None

    # Exact stem match
    exact = d / f"{name}.gguf"
    if exact.is_file():
        return exact

    # Substring match (case-insensitive)
    query = name.lower()
    matches = [p for p in d.glob("*.gguf") if query in p.stem.lower()]
    if len(matches) == 1:
        return matches[0]

    return None


class LlamaCppServer:
    """Manages a llama-server process for a single GGUF model.

    Args:
        model: Model name or path (resolved via :func:`resolve_gguf_path`).
        gguf_dir: Directory containing GGUF files. Defaults to ``~/models``.
        port: Port for the HTTP server. Defaults to 8794.
        ngl: Number of GPU layers to offload. Defaults to 99 (all).

    Usage::

        server = LlamaCppServer("Qwen3.5-9B-Q4_K_M")
        server.start()       # launches llama-server, waits until healthy
        # … run tests via OpenAI client at server.base_url …
        server.stop()        # SIGTERM + wait
    """

    def __init__(
        self,
        model: str,
        gguf_dir: Path | None = None,
        port: int = _DEFAULT_PORT,
        ngl: int = 99,
    ):
        self.model = model
        self.gguf_dir = gguf_dir or DEFAULT_GGUF_DIR
        self.gguf_path = resolve_gguf_path(model, self.gguf_dir)
        self.port = port
        self.ngl = ngl
        self._process: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}/v1"

    def start(self) -> None:
        """Start llama-server and block until it's ready to serve requests."""
        binary = find_llama_server()
        if not binary:
            raise RuntimeError(
                "llama-server not found. Install llama.cpp and ensure "
                "'llama-server' is on your PATH."
            )
        if not self.gguf_path or not self.gguf_path.is_file():
            raise FileNotFoundError(
                f"GGUF model not found: '{self.model}'. "
                f"Place .gguf files in {self.gguf_dir}/"
            )

        cmd = [
            binary,
            "-m", str(self.gguf_path),
            "-ngl", str(self.ngl),
            "--port", str(self.port),
            "--host", "127.0.0.1",
        ]

        # Suppress llama-server output (it's very chatty)
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            # Ensure child is killed if parent dies
            preexec_fn=os.setsid if sys.platform != "win32" else None,
        )

        self._wait_healthy()

    def _wait_healthy(self) -> None:
        """Poll the health endpoint until the server is ready."""
        import urllib.request
        import urllib.error

        url = f"http://127.0.0.1:{self.port}/health"
        deadline = time.monotonic() + _STARTUP_TIMEOUT

        while time.monotonic() < deadline:
            # Check the process hasn't died
            if self._process and self._process.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited unexpectedly (code {self._process.returncode}) "
                    f"while loading '{self.model}'"
                )
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, OSError, TimeoutError):
                pass
            time.sleep(1)

        # Timed out — clean up
        self.stop()
        raise TimeoutError(
            f"llama-server failed to start within {_STARTUP_TIMEOUT}s for '{self.model}'"
        )

    def stop(self) -> None:
        """Gracefully stop the llama-server process."""
        if not self._process:
            return
        try:
            if sys.platform == "win32":
                self._process.terminate()
            else:
                # Kill the whole process group
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            self._process.wait(timeout=10)
        except (ProcessLookupError, OSError):
            pass
        except subprocess.TimeoutExpired:
            if sys.platform == "win32":
                self._process.kill()
            else:
                os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
            self._process.wait(timeout=5)
        finally:
            self._process = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
