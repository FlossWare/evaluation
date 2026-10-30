# Hardware profiles

The hardware discovery API records host facts independently of inference
runtimes. It uses only the Python standard library and is explicitly invoked.

## Discover a profile

From a source checkout:

```bash
python -m evaluation_ai.hardware discover
python -m evaluation_ai.hardware discover --json
```

The JSON document includes a `schema_version`. Persist the complete profile
when reproducibility matters. A discovery timestamp and hostname are included;
profiles are observations, not immutable hardware identities.

## Profile semantics

- `host`: operating system, distribution when available, architecture, and
  system memory.
- `cpu`: vendor/model identifiers, core counts, and CPU instruction flags.
- `gpus`: devices discovered through Linux DRM sysfs, including PCI vendor
  and device IDs and the bound kernel driver when available.
- `available_tools`: selected runtime/diagnostic executables found on PATH.
  This means only that an executable is installed and discoverable.
- `discovery_warnings`: best-effort diagnostics.

Unknown values remain null. GPU memory is reported as `dedicated` only when
the kernel exposes a dedicated VRAM total. Otherwise it is
`shared_or_unknown`; this deliberately avoids inventing a VRAM capacity for
integrated GPUs.

## Detection is not qualification

This first phase does not claim that OpenVINO, Level Zero, Vulkan, CUDA,
ROCm, Ollama, or llama.cpp can execute on a detected device. Finding a GPU,
driver, or executable is not proof of runtime support or actual GPU execution.
Backend qualification and runtime configuration recommendations are separate
follow-on phases.

## Platform coverage

CPU and host discovery use portable Python interfaces, with Linux
`/proc` providing richer CPU and memory details. GPU discovery currently
uses Linux DRM sysfs. Other operating systems and GPU discovery mechanisms
will be added behind the same profile contract.
