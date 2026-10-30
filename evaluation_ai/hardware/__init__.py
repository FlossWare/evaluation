"""Hardware discovery and profile APIs.

Discovery reports hardware facts only; it does not qualify compute backends.
"""
from evaluation_ai.hardware.discovery import discover_hardware
from evaluation_ai.hardware.models import (
    SCHEMA_VERSION,
    CPUProfile,
    GPUProfile,
    HardwareProfile,
    HostProfile,
)

__all__ = [
    "SCHEMA_VERSION",
    "CPUProfile",
    "GPUProfile",
    "HardwareProfile",
    "HostProfile",
    "discover_hardware",
]
