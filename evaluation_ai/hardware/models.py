"""Versioned, runtime-neutral hardware profile models.

The profile intentionally records facts only. Runtime support and execution
qualification belong to later layers and must not be inferred from detection.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class HostProfile:
    system: str
    release: str
    distribution: str | None
    architecture: str
    hostname: str
    memory_total_bytes: int | None
    memory_available_bytes: int | None


@dataclass(frozen=True)
class CPUProfile:
    vendor: str | None
    model: str | None
    family: str | None
    model_number: str | None
    stepping: str | None
    physical_cores: int | None
    logical_cores: int
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class GPUProfile:
    index: int
    vendor: str | None
    vendor_id: str | None
    model: str | None
    device_id: str | None
    pci_bus_id: str | None
    driver: str | None
    memory_type: str
    dedicated_memory_bytes: int | None


@dataclass(frozen=True)
class HardwareProfile:
    schema_version: int
    discovered_at: str
    host: HostProfile
    cpu: CPUProfile
    gpus: tuple[GPUProfile, ...] = ()
    available_tools: tuple[str, ...] = ()
    discovery_warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize this profile as stable, human-readable JSON."""
        import json

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)
