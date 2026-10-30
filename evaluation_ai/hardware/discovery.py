"""Best-effort hardware discovery using the Python standard library."""

from __future__ import annotations

import glob
import os
import platform
import re
import shutil
import socket
from datetime import UTC, datetime
from pathlib import Path

from evaluation_ai.hardware.models import (
    SCHEMA_VERSION,
    CPUProfile,
    GPUProfile,
    HardwareProfile,
    HostProfile,
)


_VENDOR_NAMES = {
    "0x8086": "Intel",
    "0x10de": "NVIDIA",
    "0x1022": "AMD",
    "0x1414": "Microsoft",
}


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except (OSError, UnicodeError):
        return None


def _memory_info(proc_root: Path) -> tuple[int | None, int | None]:
    values: dict[str, int] = {}
    content = _read_text(proc_root / "meminfo")
    if content:
        for line in content.splitlines():
            match = re.match(r"^(MemTotal|MemAvailable):\s+(\d+)\s+kB$", line)
            if match:
                values[match.group(1)] = int(match.group(2)) * 1024
    return values.get("MemTotal"), values.get("MemAvailable")


def _distribution(os_release: Path) -> str | None:
    content = _read_text(os_release)
    if not content:
        return None
    for line in content.splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.partition("=")[2].strip().strip('"')
    return None


def _cpu_profile(proc_root: Path) -> CPUProfile:
    content = _read_text(proc_root / "cpuinfo")
    records: list[dict[str, str]] = []
    if content:
        for block in re.split(r"\n\s*\n", content):
            record: dict[str, str] = {}
            for line in block.splitlines():
                key, sep, value = line.partition(":")
                if sep:
                    record[key.strip().lower()] = value.strip()
            if record:
                records.append(record)

    first = records[0] if records else {}
    logical = len([r for r in records if "processor" in r]) or (os.cpu_count() or 1)
    core_pairs = {
        (r.get("physical id", "0"), r.get("core id"))
        for r in records
        if r.get("core id") is not None
    }
    physical = len(core_pairs) or None
    flags = tuple(sorted(set((first.get("flags") or first.get("features") or "").split())))
    return CPUProfile(
        vendor=first.get("vendor_id") or first.get("vendor"),
        model=first.get("model name") or first.get("hardware") or platform.processor() or None,
        family=first.get("cpu family") or None,
        model_number=first.get("model") or None,
        stepping=first.get("stepping") or None,
        physical_cores=physical,
        logical_cores=logical,
        flags=flags,
    )


def _gpu_profiles(sys_root: Path) -> tuple[GPUProfile, ...]:
    devices: list[GPUProfile] = []
    # cardN is used instead of renderDN to avoid duplicate entries per GPU.
    for card_path in sorted(Path(p) for p in glob.glob(str(sys_root / "class/drm/card[0-9]*"))):
        if re.search(r"card\d+$", card_path.name) is None:
            continue
        device_path = card_path / "device"
        vendor_id = _read_text(device_path / "vendor")
        device_id = _read_text(device_path / "device")
        if not vendor_id or not device_id:
            continue
        driver_link = device_path / "driver"
        try:
            driver = driver_link.resolve(strict=True).name
        except OSError:
            driver = None
        bus_id = device_path.resolve().name if device_path.exists() else None
        vram = None
        raw_vram = _read_text(device_path / "mem_info_vram_total")
        if raw_vram and raw_vram.isdigit():
            vram = int(raw_vram)
        model = (
            _read_text(device_path / "product_name")
            or _read_text(device_path / "label")
        )
        devices.append(
            GPUProfile(
                index=len(devices),
                vendor=_VENDOR_NAMES.get(vendor_id.lower()),
                vendor_id=vendor_id.lower(),
                model=model,
                device_id=device_id.lower(),
                pci_bus_id=bus_id,
                driver=driver,
                memory_type="dedicated" if vram is not None else "shared_or_unknown",
                dedicated_memory_bytes=vram,
            )
        )
    return tuple(devices)


def discover_hardware(
    *,
    proc_root: str | Path = "/proc",
    sys_root: str | Path = "/sys",
    etc_root: str | Path = "/etc",
) -> HardwareProfile:
    """Discover host, CPU, Linux DRM GPUs, and installed runtime tools.

    Paths are injectable to make discovery deterministic and testable. Missing
    platform data is represented as unknown rather than guessed.
    """
    proc = Path(proc_root)
    sysfs = Path(sys_root)
    etc = Path(etc_root)
    total, available = _memory_info(proc)
    host = HostProfile(
        system=platform.system(),
        release=platform.release(),
        distribution=_distribution(etc / "os-release"),
        architecture=platform.machine(),
        hostname=socket.gethostname(),
        memory_total_bytes=total,
        memory_available_bytes=available,
    )
    tools = (
        "ollama",
        "vulkaninfo",
        "clinfo",
        "nvidia-smi",
        "rocminfo",
        "sycl-ls",
        "vainfo",
    )
    available_tools = tuple(tool for tool in tools if shutil.which(tool))
    warnings: list[str] = []
    if platform.system() == "Linux" and not (sysfs / "class/drm").exists():
        warnings.append("Linux DRM sysfs is unavailable; GPU discovery was skipped.")
    return HardwareProfile(
        schema_version=SCHEMA_VERSION,
        discovered_at=datetime.now(UTC).isoformat(),
        host=host,
        cpu=_cpu_profile(proc),
        gpus=_gpu_profiles(sysfs),
        available_tools=available_tools,
        discovery_warnings=tuple(warnings),
    )
