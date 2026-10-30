"""Command-line entry point for hardware profile discovery."""

from __future__ import annotations

import argparse

from evaluation_ai.hardware import discover_hardware


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover host hardware.")
    parser.add_argument(
        "command",
        choices=("discover",),
        nargs="?",
        default="discover",
        help="operation to perform (default: discover)",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()
    profile = discover_hardware()
    if args.json:
        print(profile.to_json())
    else:
        display_os = profile.host.distribution or profile.host.system
        print(f"Host: {profile.host.hostname} ({display_os})")
        print(f"Architecture: {profile.host.architecture}")
        print(f"Memory: {profile.host.memory_total_bytes or 'unknown'} bytes")
        print(f"CPU: {profile.cpu.model or 'unknown'}")
        print(f"Logical CPUs: {profile.cpu.logical_cores}")
        if profile.gpus:
            for gpu in profile.gpus:
                print(
                    f"GPU {gpu.index}: {gpu.vendor or 'Unknown'} "
                    f"{gpu.model or gpu.device_id or 'device'} "
                    f"(driver: {gpu.driver or 'unknown'}, memory: {gpu.memory_type})"
                )
        else:
            print("GPU: none detected")
        if profile.available_tools:
            print("Available tools: " + ", ".join(profile.available_tools))
        for warning in profile.discovery_warnings:
            print(f"Warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
