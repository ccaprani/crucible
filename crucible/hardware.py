"""Detect hardware for result metadata."""

import os
import subprocess
from dataclasses import dataclass, asdict


@dataclass
class HardwareInfo:
    """System hardware summary."""
    gpu: str = "unknown"
    gpu_vram_gb: float = 0
    gpu_driver: str = ""
    cpu: str = "unknown"
    ram_gb: int = 0

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v}

    def summary(self) -> str:
        parts = [self.gpu]
        if self.gpu_vram_gb:
            parts[0] += f" ({self.gpu_vram_gb:.0f}GB)"
        parts.append(self.cpu)
        if self.ram_gb:
            parts.append(f"{self.ram_gb}GB RAM")
        return " | ".join(parts)


def detect_hardware() -> HardwareInfo:
    """Detect GPU, CPU, and RAM."""
    info = HardwareInfo()

    # ── GPU: try ROCm first, then NVIDIA ──
    info.gpu, info.gpu_vram_gb, info.gpu_driver = _detect_gpu()

    # ── CPU ──
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    info.cpu = line.split(":")[1].strip()
                    break
    except OSError:
        pass

    # ── RAM ──
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemTotal" in line:
                    kb = int(line.split()[1])
                    info.ram_gb = round(kb / 1024 / 1024)
                    break
    except OSError:
        pass

    return info


def _detect_gpu() -> tuple[str, float, str]:
    """Detect GPU name, VRAM, and driver. Returns (name, vram_gb, driver)."""
    # ROCm (AMD)
    try:
        import json as _json
        r = subprocess.run(
            ["rocm-smi", "--showproductname", "--json"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            data = _json.loads(r.stdout)
            # Use first discrete GPU card
            name = "AMD GPU"
            for card, details in data.items():
                name = details.get("Card Series", name)
                break
            # Get VRAM — pick the card with most VRAM (skip iGPU)
            vram = 0.0
            rv = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "--json"],
                capture_output=True, text=True, timeout=5,
            )
            if rv.returncode == 0:
                vdata = _json.loads(rv.stdout)
                for cdata in vdata.values():
                    total = int(cdata.get("VRAM Total Memory (B)", 0))
                    gb = total / (1024 ** 3)
                    if gb > vram:
                        vram = gb
            # Driver version
            driver = ""
            rd = subprocess.run(
                ["rocm-smi", "--showdriverversion", "--json"],
                capture_output=True, text=True, timeout=5,
            )
            if rd.returncode == 0:
                ddata = _json.loads(rd.stdout)
                sysinfo = ddata.get("system", {})
                dv = sysinfo.get("Driver version", "")
                if dv:
                    driver = f"ROCm {dv}"
            return name, round(vram, 1), driver
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    # NVIDIA
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            parts = r.stdout.strip().split(", ")
            name = parts[0] if parts else "NVIDIA GPU"
            vram = float(parts[1]) / 1024 if len(parts) > 1 else 0
            driver = f"CUDA {parts[2]}" if len(parts) > 2 else ""
            return name, vram, driver
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    return "unknown", 0, ""
