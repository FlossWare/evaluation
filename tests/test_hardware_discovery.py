from __future__ import annotations

import json
from pathlib import Path

from evaluation_ai.hardware import SCHEMA_VERSION, discover_hardware


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_discovery_reads_cpu_memory_and_distribution(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    etc = tmp_path / "etc"
    sysfs = tmp_path / "sys"
    _write(
        proc / "cpuinfo",
        "processor : 0\nvendor_id : GenuineIntel\nmodel name : Example CPU\n"
        "cpu family : 6\nmodel : 170\nstepping : 4\nflags : sse avx2\n"
        "physical id : 0\ncore id : 0\n\n"
        "processor : 1\nvendor_id : GenuineIntel\nmodel name : Example CPU\n"
        "physical id : 0\ncore id : 1\n",
    )
    _write(proc / "meminfo", "MemTotal:       65536000 kB\nMemAvailable:   32768000 kB\n")
    _write(etc / "os-release", 'NAME="Test OS"\nPRETTY_NAME="Test OS 1"\n')

    profile = discover_hardware(proc_root=proc, sys_root=sysfs, etc_root=etc)

    assert profile.schema_version == SCHEMA_VERSION
    assert profile.host.distribution == "Test OS 1"
    assert profile.host.memory_total_bytes == 65536000 * 1024
    assert profile.host.memory_available_bytes == 32768000 * 1024
    assert profile.cpu.vendor == "GenuineIntel"
    assert profile.cpu.model == "Example CPU"
    assert profile.cpu.physical_cores == 2
    assert profile.cpu.logical_cores == 2
    assert profile.cpu.flags == ("avx2", "sse")
    assert profile.gpus == ()


def test_discovery_detects_multiple_gpus_and_shared_memory(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    sysfs = tmp_path / "sys"
    etc = tmp_path / "etc"
    _write(proc / "cpuinfo", "processor : 0\nmodel name : Test CPU\n")
    _write(proc / "meminfo", "MemTotal: 1024 kB\n")
    _write(etc / "os-release", 'PRETTY_NAME="Test"\n')

    for card, vendor, device, driver in (
        ("card0", "0x8086", "0x7d55", "i915"),
        ("card1", "0x10de", "0x2684", "nvidia"),
    ):
        device_dir = sysfs / "class/drm" / card / "device"
        _write(device_dir / "vendor", vendor)
        _write(device_dir / "device", device)
        driver_dir = sysfs / "drivers" / driver
        driver_dir.mkdir(parents=True)
        (device_dir / "driver").symlink_to(driver_dir)

    profile = discover_hardware(proc_root=proc, sys_root=sysfs, etc_root=etc)

    assert len(profile.gpus) == 2
    assert profile.gpus[0].vendor == "Intel"
    assert profile.gpus[0].memory_type == "shared_or_unknown"
    assert profile.gpus[0].dedicated_memory_bytes is None
    assert profile.gpus[1].vendor == "NVIDIA"
    assert profile.gpus[1].driver == "nvidia"


def test_profile_serializes_to_json(tmp_path: Path) -> None:
    profile = discover_hardware(
        proc_root=tmp_path / "missing-proc",
        sys_root=tmp_path / "missing-sys",
        etc_root=tmp_path / "missing-etc",
    )
    payload = json.loads(profile.to_json())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["host"]["memory_total_bytes"] is None
    assert payload["gpus"] == []
    assert "discovered_at" in payload
