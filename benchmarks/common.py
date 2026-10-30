"""Shared helpers for reproducible FlossWare service evaluations."""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path


def command_output(args: list[str]) -> str | None:
    try:
        return subprocess.run(
            args, capture_output=True, text=True, timeout=5, check=False
        ).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def package_versions(names: list[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def resolve_sut_root() -> Path:
    """Resolve the poc-claude-ensemble checkout without coupling it to this repo."""
    configured = os.environ.get("POC_CLAUDE_ENSEMBLE_ROOT")
    sibling = Path(__file__).resolve().parents[1].parent / "poc-claude-ensemble"
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.append(sibling)
    for candidate in candidates:
        root = candidate.resolve()
        if (root / "services").is_dir():
            return root
    searched = ", ".join(str(path) for path in candidates)
    raise SystemExit(
        "Could not locate the poc-claude-ensemble checkout. Set "
        f"POC_CLAUDE_ENSEMBLE_ROOT to its path. Searched: {searched}"
    )


def host_metadata() -> dict:
    cpu = (
        command_output([
            "bash", "-lc",
            "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-"
        ])
        if sys.platform.startswith("linux")
        else platform.processor()
    )
    ram = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                ram = int(line.split()[1]) * 1024
                break
    except OSError:
        pass
    gpu = (
        command_output([
            "bash", "-lc",
            "lspci 2>/dev/null | grep -Ei 'vga|3d|display' | head -10"
        ])
        if sys.platform.startswith("linux")
        else None
    )
    return {
        "hostname": platform.node(),
        "os": platform.platform(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "architecture": platform.machine(),
        "cpu": cpu,
        "ram_bytes": ram,
        "gpu_summary": gpu,
    }


def git_metadata(root: Path) -> dict:
    def git(*args: str) -> str | None:
        return command_output(["git", "-C", str(root), *args])
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(git("status", "--porcelain")),
    }


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize_ms(samples: list[float]) -> dict:
    return {
        "samples": len(samples),
        "mean_ms": sum(samples) / len(samples) if samples else None,
        "p50_ms": percentile(samples, 50),
        "p95_ms": percentile(samples, 95),
        "min_ms": min(samples) if samples else None,
        "max_ms": max(samples) if samples else None,
    }


def write_result(track: str, payload: dict, output_dir: str) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    run_id = (
        f"{track}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    document = {
        "schema_version": 1,
        "run_id": run_id,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "track": track,
        "host": host_metadata(),
        "evaluation_repository": git_metadata(Path(__file__).resolve().parents[1]),
        "system_under_test": {
            "path": str(resolve_sut_root()),
            "git": git_metadata(resolve_sut_root()),
        },
        **payload,
    }
    path = destination / f"{run_id}.json"
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
