"""Benchmark PhotonicGenerator GPU time and memory.

This script runs CUDA-only sweeps for the photonic QGAN-style generator path:

* batch size with paired Fock spaces, using ``n_photons=n`` and
  ``n_modes=2*n``;
* number of generator heads with fixed image shape;
* Fock and unbunched computation-space sizes with fixed image shape;
* optional output-shape sizes with fixed generator and computation-space setup.

All measured values are written to JSON. Optional plots are generated from the
same JSON-compatible curve data.

Example
-------
PYTHONPATH=$PWD PCVL_PERSISTENT_PATH=.pcvl_home \
python benchmarks/benchmark_photonic_generator_gpu.py \
    --json-out benchmarks/results/photonic_generator_gpu.json \
    --plot-dir docs/source/_static/img/performance/qgan

PYTHONPATH=$PWD python benchmarks/benchmark_photonic_generator_gpu.py \
    --json-in benchmarks/results/photonic_generator_gpu.json \
    --plot-dir docs/source/_static/img/performance/qgan
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import subprocess  # noqa: S404
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    import merlin as ML
else:
    torch = None
    ML = None

BYTES_PER_MIB = 1024 * 1024
DEFAULT_BATCH_FOCK_CASES = "1:2,2:4,4:6,8:8"
DEFAULT_GENERATOR_COUNTS = "1,2,4,8"
DEFAULT_SPACE_CASES = (
    "FOCK:4:2,UNBUNCHED:4:2,"
    "FOCK:8:4,UNBUNCHED:8:4,"
    "FOCK:12:4,UNBUNCHED:12:4,"
    "FOCK:16:6,UNBUNCHED:16:6,"
    "FOCK:20:8,UNBUNCHED:20:8"
)
DEFAULT_OUTPUT_SHAPES = "1x4x4,1x8x8,1x16x16"
BUBBLE_FACE_ALPHA = 0.22
BUBBLE_EDGE_ALPHA = 0.8
CENTER_POINT_AREA = 20.0


def _ensure_runtime_dependencies() -> None:
    """Import benchmark runtime dependencies needed for CUDA execution."""
    global ML, torch

    if torch is None:
        import torch as torch_module

        torch = torch_module
    if ML is None:
        import merlin as merlin_module

        ML = merlin_module


@dataclass(frozen=True)
class GeneratorCase:
    """One PhotonicGenerator benchmark case.

    Parameters
    ----------
    name : str
        Stable case name.
    curve_name : str
        Name of the curve that owns this case.
    generator_count : int
        Number of independent quantum generator heads.
    batch_size : int
        Latent batch size.
    image_shape : tuple[int, int, int]
        Output image shape as ``(channels, height, width)``.
    computation_space : merlin.core.computation_space.ComputationSpace
        Quantum computation space used by each generator head.
    n_modes : int
        Number of photonic modes in each generator head.
    n_photons : int
        Number of photons in each generator head.
    latent_dim : int
        Number of latent features encoded into each generator head.
    depth : int
        Number of trainable entangling blocks after latent encoding.
    x_value : int
        Numeric x-axis value for plotting.
    x_label : str
        Human-readable x-axis label.
    """

    name: str
    curve_name: str
    generator_count: int
    batch_size: int
    image_shape: tuple[int, int, int]
    computation_space: ML.ComputationSpace
    n_modes: int
    n_photons: int
    latent_dim: int
    depth: int
    x_value: int
    x_label: str


@dataclass(frozen=True)
class MemoryPoint:
    """Memory metrics for one benchmark point.

    Parameters
    ----------
    curve_name : str
        Name of the benchmark curve containing the point.
    case_name : str
        Stable benchmark case name.
    x_value : float
        Numeric x-axis value from the benchmark JSON.
    x_label : str
        Human-readable x-axis label from the benchmark JSON.
    computation_space : str
        Computation-space name used by the quantum layer.
    basis_size : int
        System size, meaning the number of basis states in the measured
        computation space.
    n_modes : int
        Number of photonic modes.
    n_photons : int
        Number of photons.
    batch_size : int
        Latent batch size.
    generator_count : int
        Number of generator heads.
    setup_allocated_mib : float
        CUDA memory allocated after setup in MiB.
    peak_allocated_mib : float
        Maximum of forward/backward absolute CUDA allocated peaks in MiB.
    peak_delta_allocated_mib : float
        Maximum of forward/backward CUDA allocated peak deltas in MiB.
    peak_reserved_mib : float
        Maximum of forward/backward absolute CUDA reserved peaks in MiB.
    peak_delta_reserved_mib : float
        Maximum of forward/backward CUDA reserved peak deltas in MiB.
    """

    curve_name: str
    case_name: str
    x_value: float
    x_label: str
    computation_space: str
    basis_size: int
    n_modes: int
    n_photons: int
    batch_size: int
    generator_count: int
    setup_allocated_mib: float
    peak_allocated_mib: float
    peak_delta_allocated_mib: float
    peak_reserved_mib: float
    peak_delta_reserved_mib: float


def _parse_image_shape(value: str) -> tuple[int, int, int]:
    """Parse ``CxHxW`` or ``HxW`` image-shape strings."""
    parts = [int(part) for part in value.lower().split("x")]
    if len(parts) == 2:
        height, width = parts
        shape = (1, height, width)
    elif len(parts) == 3:
        shape = tuple(parts)
    else:
        raise ValueError("Image shape must be HxW or CxHxW.")
    if any(dim <= 0 for dim in shape):
        raise ValueError("Image shape dimensions must be positive.")
    return shape


def _parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated integer list."""
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one integer.")
    if any(item <= 0 for item in items):
        raise ValueError("All integer values must be positive.")
    return items


def _parse_batch_fock_cases(value: str) -> list[tuple[int, int]]:
    """Parse ``batch_size:n_photons`` entries."""
    cases = []
    for item in value.split(","):
        if not item.strip():
            continue
        batch_size, n_photons = [int(part) for part in item.split(":")]
        if batch_size <= 0 or n_photons <= 0:
            raise ValueError("Batch/Fock case values must be positive.")
        cases.append((batch_size, n_photons))
    if not cases:
        raise ValueError("Expected at least one batch/Fock case.")
    return cases


def _parse_space_cases(
    value: str,
) -> list[tuple[ML.ComputationSpace, int, int]]:
    """Parse ``SPACE:n_modes:n_photons`` entries."""
    cases = []
    for item in value.split(","):
        if not item.strip():
            continue
        space_name, n_modes, n_photons = item.split(":")
        computation_space = _computation_space_from_name(space_name)
        modes = int(n_modes)
        photons = int(n_photons)
        if modes <= 0 or photons <= 0:
            raise ValueError("Space case modes and photons must be positive.")
        cases.append((computation_space, modes, photons))
    if not cases:
        raise ValueError("Expected at least one computation-space case.")
    return cases


def _computation_space_from_name(name: str) -> ML.ComputationSpace:
    """Return a supported computation-space enum from a CLI name."""
    normalized_name = name.strip().upper()
    if normalized_name == "FOCK":
        return ML.ComputationSpace.FOCK
    if normalized_name == "UNBUNCHED":
        return ML.ComputationSpace.UNBUNCHED
    raise ValueError(f"Unsupported computation space: {name}.")


def _image_feature_count(image_shape: tuple[int, int, int]) -> int:
    """Return the flattened feature count for an image shape."""
    return math.prod(image_shape)


def _basis_size(
    computation_space: ML.ComputationSpace,
    n_modes: int,
    n_photons: int,
) -> int:
    """Return the computation-space system size."""
    if computation_space is ML.ComputationSpace.FOCK:
        return math.comb(n_modes + n_photons - 1, n_photons)
    if computation_space is ML.ComputationSpace.UNBUNCHED:
        return math.comb(n_modes, n_photons)
    raise ValueError(f"Unsupported computation space: {computation_space}.")


def _validate_case(case: GeneratorCase) -> None:
    """Validate one benchmark case before constructing GPU state."""
    image_features = _image_feature_count(case.image_shape)
    if image_features < case.generator_count:
        raise ValueError(
            f"{case.name}: image features ({image_features}) must be at least "
            f"generator_count ({case.generator_count})."
        )
    if image_features % case.generator_count != 0:
        raise ValueError(
            f"{case.name}: image features ({image_features}) must be divisible "
            f"by generator_count ({case.generator_count}) for headwise output."
        )
    if case.latent_dim > case.n_modes:
        raise ValueError(
            f"{case.name}: latent_dim ({case.latent_dim}) cannot exceed "
            f"n_modes ({case.n_modes})."
        )
    if (
        case.computation_space is ML.ComputationSpace.UNBUNCHED
        and case.n_photons > case.n_modes
    ):
        raise ValueError(f"{case.name}: UNBUNCHED requires n_photons <= n_modes.")


def _build_layer(case: GeneratorCase, dtype: torch.dtype) -> ML.QuantumLayer:
    """Build one CPU QuantumLayer template for a generator case."""
    builder = ML.CircuitBuilder(n_modes=case.n_modes)
    builder.add_entangling_layer(trainable=True, model="mzi", name="pre")
    builder.add_angle_encoding(modes=list(range(case.latent_dim)), name="input")
    for block_index in range(case.depth):
        builder.add_entangling_layer(
            trainable=True,
            model="mzi",
            name=f"var_{block_index}",
        )

    return ML.QuantumLayer(
        input_size=case.latent_dim,
        builder=builder,
        n_photons=case.n_photons,
        measurement_strategy=ML.MeasurementStrategy.probs(
            computation_space=case.computation_space
        ),
        dtype=dtype,
    )


def _build_generator(
    case: GeneratorCase,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> ML.PhotonicGenerator:
    """Build a PhotonicGenerator and move it to CUDA."""
    template_layer = _build_layer(case, dtype)
    generator = ML.PhotonicGenerator(
        layers=template_layer,
        count=case.generator_count,
        output_adapter=ML.ImageAdapter(
            shape=case.image_shape,
            headwise=True,
            normalize_patches=True,
        ),
    ).to(device)
    _assert_generator_on_cuda(generator)
    return generator


def _assert_generator_on_cuda(generator: ML.PhotonicGenerator) -> None:
    """Raise if generator parameters or layer runtime state are not on CUDA."""
    for parameter_name, parameter in generator.named_parameters():
        if parameter.device.type != "cuda":
            raise RuntimeError(f"{parameter_name} is on {parameter.device}, not CUDA.")

    for layer_index, layer in enumerate(generator.layers):
        if not isinstance(layer, ML.QuantumLayer):
            raise RuntimeError(f"generator.layers[{layer_index}] is not QuantumLayer.")
        if torch.device(layer.device).type != "cuda":
            raise RuntimeError(
                f"generator.layers[{layer_index}].device is {layer.device}, not CUDA."
            )
        graph_device = layer.computation_process.simulation_graph.device
        if torch.device(graph_device).type != "cuda":
            raise RuntimeError(
                f"generator.layers[{layer_index}] graph is on {graph_device}, not CUDA."
            )


def _cuda_event_elapsed_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    """Return elapsed CUDA event time in milliseconds."""
    return float(start.elapsed_time(end))


def _summarize_float_samples(samples: list[float]) -> dict[str, Any]:
    """Return JSON-safe summary statistics for float samples."""
    return {
        "samples": samples,
        "mean": mean(samples),
        "std": stdev(samples) if len(samples) > 1 else 0.0,
        "min": min(samples),
        "max": max(samples),
    }


def _summarize_int_samples(samples: list[int]) -> dict[str, Any]:
    """Return JSON-safe summary statistics for integer samples."""
    float_samples = [float(sample) for sample in samples]
    summary = _summarize_float_samples(float_samples)
    summary["samples"] = samples
    summary["mean"] = int(round(summary["mean"]))
    summary["min"] = min(samples)
    summary["max"] = max(samples)
    return summary


def _run_forward_backward_once(
    generator: ML.PhotonicGenerator,
    z: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """Measure one forward/backward pass with CUDA events and peak memory."""
    generator.zero_grad(set_to_none=True)

    forward_baseline_allocated = torch.cuda.memory_allocated(device)
    forward_baseline_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    forward_start.record()
    output = generator(z)
    forward_end.record()
    torch.cuda.synchronize(device)
    if output.device.type != "cuda":
        raise RuntimeError(f"Generator output is on {output.device}, not CUDA.")

    forward_peak_allocated = torch.cuda.max_memory_allocated(device)
    forward_peak_reserved = torch.cuda.max_memory_reserved(device)

    backward_baseline_allocated = torch.cuda.memory_allocated(device)
    backward_baseline_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    backward_start = torch.cuda.Event(enable_timing=True)
    backward_end = torch.cuda.Event(enable_timing=True)
    backward_start.record()
    loss = output.square().mean()
    loss.backward()
    backward_end.record()
    torch.cuda.synchronize(device)

    backward_peak_allocated = torch.cuda.max_memory_allocated(device)
    backward_peak_reserved = torch.cuda.max_memory_reserved(device)
    loss_value = float(loss.detach().cpu())
    output_shape = list(output.shape)
    del output
    del loss

    return {
        "forward_time_ms": _cuda_event_elapsed_ms(forward_start, forward_end),
        "backward_time_ms": _cuda_event_elapsed_ms(backward_start, backward_end),
        "forward_peak_allocated_bytes": int(forward_peak_allocated),
        "forward_peak_reserved_bytes": int(forward_peak_reserved),
        "forward_peak_delta_allocated_bytes": int(
            max(0, forward_peak_allocated - forward_baseline_allocated)
        ),
        "forward_peak_delta_reserved_bytes": int(
            max(0, forward_peak_reserved - forward_baseline_reserved)
        ),
        "backward_peak_allocated_bytes": int(backward_peak_allocated),
        "backward_peak_reserved_bytes": int(backward_peak_reserved),
        "backward_peak_delta_allocated_bytes": int(
            max(0, backward_peak_allocated - backward_baseline_allocated)
        ),
        "backward_peak_delta_reserved_bytes": int(
            max(0, backward_peak_reserved - backward_baseline_reserved)
        ),
        "loss": loss_value,
        "output_shape": output_shape,
    }


def _run_case(
    case: GeneratorCase,
    *,
    dtype: torch.dtype,
    device: torch.device,
    warmup_steps: int,
    repetitions: int,
) -> dict[str, Any]:
    """Run one benchmark case and return a JSON-ready result row."""
    _validate_case(case)
    _cleanup_cuda(device)
    setup_start = time.perf_counter()
    generator = _build_generator(case, dtype=dtype, device=device)
    z = torch.randn(
        case.batch_size,
        case.latent_dim,
        dtype=dtype,
        device=device,
    )
    torch.cuda.synchronize(device)
    setup_time_s = time.perf_counter() - setup_start
    allocated_after_setup = torch.cuda.memory_allocated(device)
    reserved_after_setup = torch.cuda.memory_reserved(device)

    for _ in range(warmup_steps):
        _ = _run_forward_backward_once(generator, z, device)
    torch.cuda.synchronize(device)

    measurements = [
        _run_forward_backward_once(generator, z, device) for _ in range(repetitions)
    ]
    torch.cuda.synchronize(device)

    output_size_per_generator = generator[0].output_size
    parameter_count = sum(parameter.numel() for parameter in generator.parameters())
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in generator.parameters()
    )
    result = {
        "case_name": case.name,
        "curve_name": case.curve_name,
        "x_value": case.x_value,
        "x_label": case.x_label,
        "generator_count": case.generator_count,
        "batch_size": case.batch_size,
        "image_shape": list(case.image_shape),
        "image_features": _image_feature_count(case.image_shape),
        "latent_dim": case.latent_dim,
        "depth": case.depth,
        "computation_space": case.computation_space.name,
        "n_modes": case.n_modes,
        "n_photons": case.n_photons,
        "basis_size": _basis_size(
            case.computation_space,
            case.n_modes,
            case.n_photons,
        ),
        "output_size_per_generator": output_size_per_generator,
        "parameter_count": parameter_count,
        "parameter_bytes": parameter_bytes,
        "setup_time_s": setup_time_s,
        "allocated_after_setup_bytes": int(allocated_after_setup),
        "reserved_after_setup_bytes": int(reserved_after_setup),
        "forward_time_ms": _summarize_float_samples([
            row["forward_time_ms"] for row in measurements
        ]),
        "backward_time_ms": _summarize_float_samples([
            row["backward_time_ms"] for row in measurements
        ]),
        "forward_peak_delta_allocated_bytes": _summarize_int_samples([
            row["forward_peak_delta_allocated_bytes"] for row in measurements
        ]),
        "backward_peak_delta_allocated_bytes": _summarize_int_samples([
            row["backward_peak_delta_allocated_bytes"] for row in measurements
        ]),
        "forward_peak_delta_reserved_bytes": _summarize_int_samples([
            row["forward_peak_delta_reserved_bytes"] for row in measurements
        ]),
        "backward_peak_delta_reserved_bytes": _summarize_int_samples([
            row["backward_peak_delta_reserved_bytes"] for row in measurements
        ]),
        "forward_peak_allocated_bytes": _summarize_int_samples([
            row["forward_peak_allocated_bytes"] for row in measurements
        ]),
        "backward_peak_allocated_bytes": _summarize_int_samples([
            row["backward_peak_allocated_bytes"] for row in measurements
        ]),
        "forward_peak_reserved_bytes": _summarize_int_samples([
            row["forward_peak_reserved_bytes"] for row in measurements
        ]),
        "backward_peak_reserved_bytes": _summarize_int_samples([
            row["backward_peak_reserved_bytes"] for row in measurements
        ]),
        "loss": _summarize_float_samples([row["loss"] for row in measurements]),
        "output_shape": measurements[-1]["output_shape"],
    }

    del generator
    del z
    _cleanup_cuda(device)
    return result


def _cleanup_cuda(device: torch.device) -> None:
    """Collect Python and CUDA cached memory before or after a case."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)


def _dtype_from_name(name: str) -> torch.dtype:
    """Return a supported floating dtype from a CLI name."""
    normalized_name = name.strip().lower()
    if normalized_name in {"float32", "fp32"}:
        return torch.float32
    if normalized_name in {"float64", "fp64"}:
        return torch.float64
    raise ValueError(f"Unsupported dtype: {name}.")


def _case_common_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Return common case fields from parsed CLI arguments."""
    return {
        "latent_dim": args.latent_dim,
        "depth": args.depth,
    }


def _build_cases(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """Build all configured benchmark curves."""
    image_shape = _parse_image_shape(args.image_shape)
    common_kwargs = _case_common_kwargs(args)

    batch_fock_points = []
    for batch_size, n_photons in _parse_batch_fock_cases(args.batch_fock_cases):
        n_modes = 2 * n_photons
        batch_fock_points.append(
            GeneratorCase(
                name=f"batch_{batch_size}_fock_m{n_modes}_p{n_photons}",
                curve_name="batch_fock_curve",
                generator_count=args.generator_count,
                batch_size=batch_size,
                image_shape=image_shape,
                computation_space=ML.ComputationSpace.FOCK,
                n_modes=n_modes,
                n_photons=n_photons,
                x_value=batch_size,
                x_label=f"batch={batch_size}",
                **common_kwargs,
            )
        )

    generator_count_points = []
    for generator_count in _parse_int_list(args.generator_counts):
        generator_count_points.append(
            GeneratorCase(
                name=f"generators_{generator_count}",
                curve_name="generator_count_curve",
                generator_count=generator_count,
                batch_size=args.batch_size,
                image_shape=image_shape,
                computation_space=ML.ComputationSpace.FOCK,
                n_modes=args.n_modes,
                n_photons=args.n_photons,
                x_value=generator_count,
                x_label=f"N={generator_count}",
                **common_kwargs,
            )
        )

    space_points = []
    for computation_space, n_modes, n_photons in _parse_space_cases(args.space_cases):
        basis_size = _basis_size(computation_space, n_modes, n_photons)
        space_points.append(
            GeneratorCase(
                name=(f"{computation_space.name.lower()}_m{n_modes}_p{n_photons}"),
                curve_name="space_shape_curve",
                generator_count=args.generator_count,
                batch_size=args.batch_size,
                image_shape=image_shape,
                computation_space=computation_space,
                n_modes=n_modes,
                n_photons=n_photons,
                x_value=basis_size,
                x_label=f"{computation_space.name} system size = {basis_size}",
                **common_kwargs,
            )
        )

    output_shape_points = []
    if not args.skip_output_shape_sweep:
        for output_shape in [
            _parse_image_shape(shape) for shape in args.output_shapes.split(",")
        ]:
            image_features = _image_feature_count(output_shape)
            output_shape_points.append(
                GeneratorCase(
                    name=f"output_shape_{'x'.join(str(dim) for dim in output_shape)}",
                    curve_name="output_shape_curve",
                    generator_count=args.generator_count,
                    batch_size=args.batch_size,
                    image_shape=output_shape,
                    computation_space=ML.ComputationSpace.FOCK,
                    n_modes=args.n_modes,
                    n_photons=args.n_photons,
                    x_value=image_features,
                    x_label=f"features={image_features}",
                    **common_kwargs,
                )
            )

    curves = {
        "batch_fock_curve": {
            "description": (
                "Batch-size sweep where each point uses FOCK with "
                "n_photons=n and n_modes=2*n."
            ),
            "x_axis": "batch_size",
            "points": batch_fock_points,
        },
        "generator_count_curve": {
            "description": (
                "Generator-head-count sweep with fixed image and Fock space."
            ),
            "x_axis": "generator_count",
            "points": generator_count_points,
        },
        "space_shape_curve": {
            "description": (
                "FOCK and UNBUNCHED computation-space sweep with fixed image "
                "shape and generator count."
            ),
            "x_axis": "basis_size",
            "points": space_points,
        },
    }
    if output_shape_points:
        curves["output_shape_curve"] = {
            "description": (
                "Output-shape sweep. This measures adapter/runtime cost only; "
                "model quality is not guaranteed across shapes."
            ),
            "x_axis": "image_features",
            "model_quality_not_guaranteed": True,
            "points": output_shape_points,
        }
    return curves


def _git_value(args: list[str], repo: Path) -> str:
    """Return git metadata when available."""
    try:
        return subprocess.check_output(  # noqa: S603
            ["git", *args],  # noqa: S607
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_dirty_summary(repo: Path) -> dict[str, Any]:
    """Return compact dirty-worktree metadata."""
    status = _git_value(["status", "--short"], repo)
    if status == "unknown":
        return {"dirty": "unknown", "status_line_count": None}
    status_lines = [line for line in status.splitlines() if line.strip()]
    return {"dirty": bool(status_lines), "status_line_count": len(status_lines)}


def _device_metadata(device: torch.device) -> dict[str, Any]:
    """Return CUDA device metadata."""
    props = torch.cuda.get_device_properties(device)
    return {
        "device": str(device),
        "name": torch.cuda.get_device_name(device),
        "index": torch.cuda.current_device(),
        "total_memory_bytes": int(props.total_memory),
        "major": int(props.major),
        "minor": int(props.minor),
        "multi_processor_count": int(props.multi_processor_count),
    }


def _benchmark_metadata(
    args: argparse.Namespace, device: torch.device
) -> dict[str, Any]:
    """Return run metadata for the output JSON."""
    repo = Path(__file__).resolve().parents[1]
    return {
        "schema_version": 1,
        "benchmark": "photonic_generator_gpu",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "git": {
            "commit": _git_value(["rev-parse", "HEAD"], repo),
            "branch": _git_value(["branch", "--show-current"], repo),
            **_git_dirty_summary(repo),
        },
        "device": _device_metadata(device),
        "settings": {
            "dtype": args.dtype,
            "warmup_steps": args.warmup_steps,
            "repetitions": args.repetitions,
            "image_shape": list(_parse_image_shape(args.image_shape)),
            "batch_size": args.batch_size,
            "generator_count": args.generator_count,
            "n_modes": args.n_modes,
            "n_photons": args.n_photons,
            "latent_dim": args.latent_dim,
            "depth": args.depth,
            "batch_fock_cases": args.batch_fock_cases,
            "generator_counts": args.generator_counts,
            "space_cases": args.space_cases,
            "output_shapes": args.output_shapes,
            "skip_output_shape_sweep": args.skip_output_shape_sweep,
        },
    }


def _run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run all configured curves and return JSON-ready results."""
    _ensure_runtime_dependencies()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    requested_device = torch.device(args.device)
    if requested_device.type != "cuda":
        raise ValueError("This benchmark requires a CUDA device.")
    if requested_device.index is not None:
        torch.cuda.set_device(requested_device)
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = _dtype_from_name(args.dtype)
    curves = _build_cases(args)
    result = _benchmark_metadata(args, device)
    result["curves"] = {
        curve_name: {
            **{key: value for key, value in curve_config.items() if key != "points"},
            "points": [],
        }
        for curve_name, curve_config in curves.items()
    }
    if args.plot_dir is not None:
        result["plots"] = []

    for curve_name, curve_config in curves.items():
        curve_result = result["curves"][curve_name]
        for case in curve_config["points"]:
            print(f"Running {case.name}...", flush=True)
            curve_result["points"].append(
                _run_case(
                    case,
                    dtype=dtype,
                    device=device,
                    warmup_steps=args.warmup_steps,
                    repetitions=args.repetitions,
                )
            )
            if args.plot_dir is not None:
                _record_plot_path(
                    result,
                    _plot_metric_curve(result, args.plot_dir, curve_name),
                )
            _write_json(result, args.json_out)
    if args.plot_dir is not None:
        for plot_path in _plot_memory_graphs(
            result,
            args.plot_dir,
            bubble_scale=args.bubble_scale,
            min_bubble_area=args.min_bubble_area,
            max_bubble_area=args.max_bubble_area,
            annotate=not args.no_annotations,
            log_y=args.log_y,
        ):
            _record_plot_path(result, str(plot_path))
    return result


def _metric_series(points: list[dict[str, Any]], metric_name: str) -> list[float]:
    """Return mean metric values from curve points."""
    return [float(point[metric_name]["mean"]) for point in points]


def _memory_series_mib(points: list[dict[str, Any]]) -> list[float]:
    """Return peak forward/backward allocated delta in MiB."""
    values = []
    for point in points:
        forward_peak = point["forward_peak_delta_allocated_bytes"]["max"]
        backward_peak = point["backward_peak_delta_allocated_bytes"]["max"]
        values.append(max(forward_peak, backward_peak) / BYTES_PER_MIB)
    return values


def _sample_max_mib(point: dict[str, Any], field_name: str) -> float:
    """Return a benchmark sample maximum converted to MiB."""
    return float(point[field_name]["max"]) / BYTES_PER_MIB


def _bytes_mib(point: dict[str, Any], field_name: str) -> float:
    """Return a byte field converted to MiB."""
    return float(point[field_name]) / BYTES_PER_MIB


def _memory_point(curve_name: str, point: dict[str, Any]) -> MemoryPoint:
    """Build a memory point from one JSON point."""
    forward_peak_allocated = _sample_max_mib(point, "forward_peak_allocated_bytes")
    backward_peak_allocated = _sample_max_mib(point, "backward_peak_allocated_bytes")
    forward_delta_allocated = _sample_max_mib(
        point,
        "forward_peak_delta_allocated_bytes",
    )
    backward_delta_allocated = _sample_max_mib(
        point,
        "backward_peak_delta_allocated_bytes",
    )
    forward_peak_reserved = _sample_max_mib(point, "forward_peak_reserved_bytes")
    backward_peak_reserved = _sample_max_mib(point, "backward_peak_reserved_bytes")
    forward_delta_reserved = _sample_max_mib(
        point,
        "forward_peak_delta_reserved_bytes",
    )
    backward_delta_reserved = _sample_max_mib(
        point,
        "backward_peak_delta_reserved_bytes",
    )
    return MemoryPoint(
        curve_name=curve_name,
        case_name=point["case_name"],
        x_value=float(point["x_value"]),
        x_label=point["x_label"],
        computation_space=point["computation_space"],
        basis_size=int(point["basis_size"]),
        n_modes=int(point["n_modes"]),
        n_photons=int(point["n_photons"]),
        batch_size=int(point["batch_size"]),
        generator_count=int(point["generator_count"]),
        setup_allocated_mib=_bytes_mib(point, "allocated_after_setup_bytes"),
        peak_allocated_mib=max(forward_peak_allocated, backward_peak_allocated),
        peak_delta_allocated_mib=max(
            forward_delta_allocated,
            backward_delta_allocated,
        ),
        peak_reserved_mib=max(forward_peak_reserved, backward_peak_reserved),
        peak_delta_reserved_mib=max(forward_delta_reserved, backward_delta_reserved),
    )


def _memory_points(result: dict[str, Any]) -> list[MemoryPoint]:
    """Return memory points parsed from a benchmark result."""
    points = []
    for curve_name, curve in result["curves"].items():
        for point in curve["points"]:
            points.append(_memory_point(curve_name, point))
    return points


def _basis_transform(value: int, scale: str) -> float:
    """Transform a system size before mapping it to marker area."""
    if scale == "linear":
        return float(value)
    if scale == "log":
        return math.log10(max(1, value))
    raise ValueError(f"Unsupported bubble scale: {scale}.")


def _bubble_area(
    basis_size: int,
    basis_sizes: list[int],
    *,
    scale: str,
    min_area: float,
    max_area: float,
) -> float:
    """Map one system size to a scatter marker area."""
    transformed_values = [_basis_transform(value, scale) for value in basis_sizes]
    transformed = _basis_transform(basis_size, scale)
    low = min(transformed_values)
    high = max(transformed_values)
    if math.isclose(low, high):
        return (min_area + max_area) / 2
    ratio = (transformed - low) / (high - low)
    return min_area + ratio * (max_area - min_area)


def _bubble_areas(
    points: list[MemoryPoint],
    *,
    scale: str,
    min_area: float,
    max_area: float,
) -> list[float]:
    """Return marker areas for a sequence of memory points."""
    basis_sizes = [point.basis_size for point in points]
    return [
        _bubble_area(
            point.basis_size,
            basis_sizes,
            scale=scale,
            min_area=min_area,
            max_area=max_area,
        )
        for point in points
    ]


def _rgba_color(color: str, alpha: float) -> tuple[float, float, float, float]:
    """Return a Matplotlib color with the requested alpha channel."""
    from matplotlib.colors import to_rgba

    red, green, blue, _ = to_rgba(color)
    return red, green, blue, alpha


def _scatter_bubble_points(
    axis: Any,
    x_values: Sequence[float],
    y_values: Sequence[float],
    marker_areas: Sequence[float],
    color: str,
    *,
    label: str | None = None,
) -> None:
    """Draw translucent basis-size bubbles with centered point markers."""
    axis.scatter(
        x_values,
        y_values,
        s=marker_areas,
        facecolors=_rgba_color(color, BUBBLE_FACE_ALPHA),
        edgecolors=_rgba_color(color, BUBBLE_EDGE_ALPHA),
        linewidths=1.1,
        label=label,
        zorder=3,
    )
    axis.scatter(
        x_values,
        y_values,
        s=CENTER_POINT_AREA,
        facecolors=color,
        edgecolors="white",
        linewidths=0.5,
        zorder=4,
    )


def _curve_memory_points(
    points: list[MemoryPoint],
    curve_name: str,
) -> list[MemoryPoint]:
    """Return memory points for one curve sorted by x-axis value."""
    return sorted(
        [point for point in points if point.curve_name == curve_name],
        key=lambda point: point.x_value,
    )


def _group_memory_points(points: list[MemoryPoint]) -> dict[str, list[MemoryPoint]]:
    """Group memory points by computation space."""
    grouped: dict[str, list[MemoryPoint]] = {}
    for point in points:
        grouped.setdefault(point.computation_space, []).append(point)
    for space_name, space_points in grouped.items():
        grouped[space_name] = sorted(space_points, key=lambda point: point.x_value)
    return grouped


def _point_label(point: MemoryPoint) -> str:
    """Return a compact point annotation."""
    return f"{point.n_photons}p/{point.n_modes}m\nsystem size = {point.basis_size:,}"


def _legend_basis_values(points: list[MemoryPoint]) -> list[int]:
    """Return representative system sizes for the bubble legend."""
    basis_sizes = sorted({point.basis_size for point in points})
    if len(basis_sizes) <= 3:
        return basis_sizes
    return [
        basis_sizes[0],
        basis_sizes[len(basis_sizes) // 2],
        basis_sizes[-1],
    ]


def _add_bubble_legend(
    axis: Any,
    points: list[MemoryPoint],
    *,
    scale: str,
    min_area: float,
    max_area: float,
) -> None:
    """Add a marker-size legend for system size."""
    basis_sizes = [point.basis_size for point in points]
    handles = []
    labels = []
    for basis_size in _legend_basis_values(points):
        area = _bubble_area(
            basis_size,
            basis_sizes,
            scale=scale,
            min_area=min_area,
            max_area=max_area,
        )
        handle = axis.scatter(
            [],
            [],
            s=area,
            facecolors=_rgba_color("black", BUBBLE_FACE_ALPHA),
            edgecolors=_rgba_color("black", BUBBLE_EDGE_ALPHA),
            linewidths=1.1,
        )
        handles.append(handle)
        labels.append(f"system size = {basis_size:,}")

    title = "System size"
    if scale == "log":
        title = "System size (log area)"
    bubble_legend = axis.legend(
        handles,
        labels,
        title=title,
        loc="upper left",
        fontsize=8,
        title_fontsize=8,
        framealpha=0.9,
    )
    axis.add_artist(bubble_legend)


def _plot_memory_axis(
    axis: Any,
    points: list[MemoryPoint],
    metric_name: str,
    ylabel: str,
    *,
    bubble_scale: str,
    min_bubble_area: float,
    max_bubble_area: float,
    annotate: bool,
) -> None:
    """Plot one memory metric with basis-size bubble markers."""
    grouped = _group_memory_points(points)
    colors = {
        "FOCK": "tab:blue",
        "UNBUNCHED": "tab:orange",
        "DUAL_RAIL": "tab:green",
    }
    for space_name, space_points in grouped.items():
        x_values = [point.x_value for point in space_points]
        y_values = [float(getattr(point, metric_name)) for point in space_points]
        color = colors.get(space_name, "tab:gray")
        axis.plot(
            x_values,
            y_values,
            color=color,
            linewidth=1.4,
            alpha=0.65,
            label=space_name,
        )
        _scatter_bubble_points(
            axis,
            x_values,
            y_values,
            _bubble_areas(
                space_points,
                scale=bubble_scale,
                min_area=min_bubble_area,
                max_area=max_bubble_area,
            ),
            color,
        )
        if annotate:
            for x_value, y_value, point in zip(
                x_values,
                y_values,
                space_points,
                strict=True,
            ):
                axis.annotate(
                    _point_label(point),
                    (x_value, y_value),
                    textcoords="offset points",
                    xytext=(7, 7),
                    fontsize=8,
                )

    if len(grouped) > 1:
        axis.legend(loc="best", fontsize=8)
    _add_bubble_legend(
        axis,
        points,
        scale=bubble_scale,
        min_area=min_bubble_area,
        max_area=max_bubble_area,
    )
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)


def _plot_memory_curve(
    result: dict[str, Any],
    points: list[MemoryPoint],
    curve_name: str,
    plot_dir: Path,
    *,
    bubble_scale: str,
    min_bubble_area: float,
    max_bubble_area: float,
    annotate: bool,
    log_y: bool,
) -> Path:
    """Plot memory metrics for one benchmark curve."""
    import matplotlib.pyplot as plt

    curve_points = _curve_memory_points(points, curve_name)
    curve = result["curves"][curve_name]
    x_axis = curve.get("x_axis", curve_name)
    if x_axis == "basis_size":
        x_axis = "system size"
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8), constrained_layout=True)
    _plot_memory_axis(
        axes[0],
        curve_points,
        "peak_allocated_mib",
        "Peak allocated CUDA memory (MiB)",
        bubble_scale=bubble_scale,
        min_bubble_area=min_bubble_area,
        max_bubble_area=max_bubble_area,
        annotate=annotate,
    )
    _plot_memory_axis(
        axes[1],
        curve_points,
        "peak_delta_allocated_mib",
        "Peak allocated delta (MiB)",
        bubble_scale=bubble_scale,
        min_bubble_area=min_bubble_area,
        max_bubble_area=max_bubble_area,
        annotate=annotate,
    )
    for axis in axes:
        if log_y:
            axis.set_yscale("log")
        axis.set_xlabel(x_axis)

    fig.suptitle(curve_name.replace("_", " ").title())
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / f"{curve_name}_torch_allocated_memory.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_memory_overview(
    points: list[MemoryPoint],
    plot_dir: Path,
    *,
    bubble_scale: str,
    min_bubble_area: float,
    max_bubble_area: float,
    log_y: bool,
) -> Path:
    """Plot all benchmark points against computation-space system size."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8), constrained_layout=True)
    metric_specs = [
        ("peak_allocated_mib", "Peak allocated CUDA memory (MiB)"),
        ("peak_delta_allocated_mib", "Peak allocated delta (MiB)"),
    ]
    colors = {
        "batch_fock_curve": "tab:blue",
        "generator_count_curve": "tab:purple",
        "space_shape_curve": "tab:orange",
        "output_shape_curve": "tab:green",
    }
    for axis, (metric_name, ylabel) in zip(axes, metric_specs, strict=True):
        for curve_name in sorted({point.curve_name for point in points}):
            curve_points = sorted(
                [point for point in points if point.curve_name == curve_name],
                key=lambda point: point.basis_size,
            )
            _scatter_bubble_points(
                axis,
                [point.basis_size for point in curve_points],
                [float(getattr(point, metric_name)) for point in curve_points],
                _bubble_areas(
                    curve_points,
                    scale=bubble_scale,
                    min_area=min_bubble_area,
                    max_area=max_bubble_area,
                ),
                colors.get(curve_name, "tab:gray"),
                label=curve_name,
            )
        axis.set_xscale("log")
        if log_y:
            axis.set_yscale("log")
        axis.set_xlabel("system size")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best", fontsize=8)

    fig.suptitle("Torch CUDA Allocated Memory vs Computation Space")
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / "torch_allocated_memory_overview.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_memory_graphs(
    result: dict[str, Any],
    plot_dir: Path,
    *,
    bubble_scale: str,
    min_bubble_area: float,
    max_bubble_area: float,
    annotate: bool,
    log_y: bool,
) -> list[Path]:
    """Write QGAN allocated-memory graphs and return created paths."""
    points = _memory_points(result)
    if not points:
        raise ValueError("No benchmark points found in result.")

    written_paths = []
    for curve_name in result["curves"]:
        if _curve_memory_points(points, curve_name):
            written_paths.append(
                _plot_memory_curve(
                    result,
                    points,
                    curve_name,
                    plot_dir,
                    bubble_scale=bubble_scale,
                    min_bubble_area=min_bubble_area,
                    max_bubble_area=max_bubble_area,
                    annotate=annotate,
                    log_y=log_y,
                )
            )
    written_paths.append(
        _plot_memory_overview(
            points,
            plot_dir,
            bubble_scale=bubble_scale,
            min_bubble_area=min_bubble_area,
            max_bubble_area=max_bubble_area,
            log_y=log_y,
        )
    )
    return written_paths


def _format_mib(value: float) -> str:
    """Format a MiB value for console tables."""
    return f"{value:,.1f}"


def _print_memory_summary(points: list[MemoryPoint]) -> None:
    """Print a compact memory summary."""
    max_allocated = max(points, key=lambda point: point.peak_allocated_mib)
    max_delta = max(points, key=lambda point: point.peak_delta_allocated_mib)
    print(
        "Max absolute allocated: "
        f"{_format_mib(max_allocated.peak_allocated_mib)} MiB "
        f"({max_allocated.case_name}, system size = {max_allocated.basis_size:,})"
    )
    print(
        "Max allocated delta: "
        f"{_format_mib(max_delta.peak_delta_allocated_mib)} MiB "
        f"({max_delta.case_name}, system size = {max_delta.basis_size:,})"
    )
    batch_points = _curve_memory_points(points, "batch_fock_curve")
    if not batch_points:
        return

    print("\nbatch_fock_curve CUDA allocated memory:")
    print(
        "case                         batch   photons/modes   system size"
        "   peak_abs_MiB   peak_delta_MiB"
    )
    for point in batch_points:
        photon_mode_label = f"{point.n_photons}p/{point.n_modes}m"
        print(
            f"{point.case_name:<28} "
            f"{point.batch_size:>5}   "
            f"{photon_mode_label:>8}       "
            f"{point.basis_size:>10,}   "
            f"{_format_mib(point.peak_allocated_mib):>12}   "
            f"{_format_mib(point.peak_delta_allocated_mib):>14}"
        )


def _plot_metric_curves(result: dict[str, Any], plot_dir: Path) -> list[str]:
    """Write one three-panel plot per curve and return created paths."""
    plot_paths = []
    for curve_name in result["curves"]:
        path = _plot_metric_curve(result, plot_dir, curve_name)
        if path is not None:
            plot_paths.append(path)
    return plot_paths


def _plot_metric_curve(
    result: dict[str, Any],
    plot_dir: Path,
    curve_name: str,
) -> str | None:
    """Write one three-panel plot for a single curve."""
    import matplotlib.pyplot as plt

    curve = result["curves"][curve_name]
    points = curve["points"]
    x_values = [point["x_value"] for point in points]
    if not x_values:
        return None

    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), constrained_layout=True)
    _plot_metric_axis(
        axes[0],
        curve_name,
        points,
        x_values,
        _metric_series(points, "forward_time_ms"),
        "Forward time (ms)",
    )
    _plot_metric_axis(
        axes[1],
        curve_name,
        points,
        x_values,
        _metric_series(points, "backward_time_ms"),
        "Backward time (ms)",
    )
    _plot_metric_axis(
        axes[2],
        curve_name,
        points,
        x_values,
        _memory_series_mib(points),
        "Peak allocated delta (MiB)",
    )
    fig.suptitle(curve_name.replace("_", " ").title())
    path = plot_dir / f"{curve_name}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def _record_plot_path(result: dict[str, Any], path: str | None) -> None:
    """Record a generated plot path once in the JSON result."""
    if path is None:
        return
    plots = result.setdefault("plots", [])
    if path not in plots:
        plots.append(path)


def _plot_metric_axis(
    axis: Any,
    curve_name: str,
    points: list[dict[str, Any]],
    x_values: list[int],
    y_values: list[float],
    ylabel: str,
) -> None:
    """Plot one metric axis with curve-specific grouping and annotations."""
    if curve_name == "space_shape_curve":
        for space_name in sorted({point["computation_space"] for point in points}):
            selected = [
                (x, y, point)
                for x, y, point in zip(x_values, y_values, points, strict=True)
                if point["computation_space"] == space_name
            ]
            axis.plot(
                [row[0] for row in selected],
                [row[1] for row in selected],
                marker="o",
                label=space_name,
            )
        axis.legend()
    else:
        axis.plot(x_values, y_values, marker="o")

    if curve_name == "batch_fock_curve":
        axis.scatter(
            x_values,
            y_values,
            s=140,
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
        )
        for x_value, y_value, point in zip(
            x_values,
            y_values,
            points,
            strict=True,
        ):
            axis.annotate(
                (
                    f"{point['n_photons']}p/{point['n_modes']}m\n"
                    f"system size = {point['basis_size']}"
                ),
                (x_value, y_value),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
            )

    axis.set_xlabel(points[0]["curve_name"].replace("_", " "))
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)


def _write_json(result: dict[str, Any], path: Path) -> None:
    """Write benchmark JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


def _read_json(path: Path) -> dict[str, Any]:
    """Read a benchmark JSON document."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _render_plots_from_json(args: argparse.Namespace) -> int:
    """Render memory graphs from an existing benchmark JSON document."""
    if args.plot_dir is None:
        raise ValueError("plot_dir is required when json_in is supplied.")

    result = _read_json(args.json_in)
    points = _memory_points(result)
    if not points:
        raise ValueError("No benchmark points found in JSON.")

    written_paths = _plot_memory_graphs(
        result,
        args.plot_dir,
        bubble_scale=args.bubble_scale,
        min_bubble_area=args.min_bubble_area,
        max_bubble_area=args.max_bubble_area,
        annotate=not args.no_annotations,
        log_y=args.log_y,
    )
    _print_memory_summary(points)
    print("\nWrote plots:")
    for path in written_paths:
        print(f"  {path}")
    return 0


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark PhotonicGenerator GPU time and memory."
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("benchmarks/results/photonic_generator_gpu.json"),
        help="Path where JSON results are written.",
    )
    parser.add_argument(
        "--json-in",
        type=Path,
        default=None,
        help="Existing benchmark JSON to render memory graphs from without CUDA.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Optional directory where timing plots and memory graph PNGs are written.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="CUDA device string, for example cuda or cuda:0.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=("float32", "float64"),
        help="Floating dtype used by generator layers and latent tensors.",
    )
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--image-shape", default="1x4x4")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--generator-count", type=int, default=1)
    parser.add_argument("--n-modes", type=int, default=20)
    parser.add_argument("--n-photons", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument(
        "--batch-fock-cases",
        default=DEFAULT_BATCH_FOCK_CASES,
        help="Comma-separated batch_size:n_photons entries. n_modes is 2*n.",
    )
    parser.add_argument(
        "--generator-counts",
        default=DEFAULT_GENERATOR_COUNTS,
        help="Comma-separated generator counts for the N-generator curve.",
    )
    parser.add_argument(
        "--space-cases",
        default=DEFAULT_SPACE_CASES,
        help="Comma-separated SPACE:n_modes:n_photons entries.",
    )
    parser.add_argument(
        "--output-shapes",
        default=DEFAULT_OUTPUT_SHAPES,
        help="Comma-separated CxHxW or HxW shapes for output-shape sweep.",
    )
    parser.add_argument(
        "--skip-output-shape-sweep",
        action="store_true",
        help="Do not run the output-shape diagnostic sweep.",
    )
    parser.add_argument(
        "--bubble-scale",
        choices=("log", "linear"),
        default="log",
        help="Mapping from system size to memory-graph marker area.",
    )
    parser.add_argument(
        "--min-bubble-area",
        type=float,
        default=70.0,
        help="Smallest memory-graph scatter marker area.",
    )
    parser.add_argument(
        "--max-bubble-area",
        type=float,
        default=950.0,
        help="Largest memory-graph scatter marker area.",
    )
    parser.add_argument(
        "--no-annotations",
        action="store_true",
        help="Do not annotate memory-graph points with photon/mode and basis labels.",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use a logarithmic y-axis for memory graphs.",
    )
    args = parser.parse_args()
    if args.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative.")
    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.generator_count <= 0:
        raise ValueError("generator_count must be positive.")
    if args.n_modes <= 0 or args.n_photons <= 0:
        raise ValueError("n_modes and n_photons must be positive.")
    if args.latent_dim <= 0:
        raise ValueError("latent_dim must be positive.")
    if args.depth < 0:
        raise ValueError("depth must be non-negative.")
    if args.min_bubble_area <= 0:
        raise ValueError("min-bubble-area must be positive.")
    if args.max_bubble_area <= args.min_bubble_area:
        raise ValueError("max-bubble-area must be larger than min-bubble-area.")
    return args


def main() -> int:
    """Run the benchmark from the command line."""
    args = _parse_args()
    if args.json_in is not None:
        return _render_plots_from_json(args)

    result = _run_benchmark(args)
    _write_json(result, args.json_out)
    print(f"Wrote JSON results to {args.json_out}")
    if args.plot_dir is not None:
        print(f"Wrote plots to {args.plot_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
