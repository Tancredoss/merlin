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
    --plot-dir benchmarks/results/photonic_generator_gpu_plots
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch

import merlin as ML

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
    """Return the output basis size for the computation space."""
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

    layer = generator[0]
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
        "output_size_per_generator": layer.output_size,
        "parameter_count": sum(
            parameter.numel() for parameter in generator.parameters()
        ),
        "parameter_bytes": sum(
            parameter.numel() * parameter.element_size()
            for parameter in generator.parameters()
        ),
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
        basis_size = _basis_size(ML.ComputationSpace.FOCK, n_modes, n_photons)
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
                x_label=f"{computation_space.name} |S|={basis_size}",
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
                    _plot_curve(result, args.plot_dir, curve_name),
                )
            _write_json(result, args.json_out)
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


def _plot_curves(result: dict[str, Any], plot_dir: Path) -> list[str]:
    """Write one three-panel plot per curve and return created paths."""
    plot_paths = []
    for curve_name in result["curves"]:
        path = _plot_curve(result, plot_dir, curve_name)
        if path is not None:
            plot_paths.append(path)
    return plot_paths


def _plot_curve(
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
                f"{point['n_photons']}p/{point['n_modes']}m\n|S|={point['basis_size']}",
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
        "--plot-dir",
        type=Path,
        default=None,
        help="Optional directory where curve PNGs are written.",
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
    return args


def main() -> int:
    """Run the benchmark from the command line."""
    args = _parse_args()
    result = _run_benchmark(args)
    _write_json(result, args.json_out)
    print(f"Wrote JSON results to {args.json_out}")
    if args.plot_dir is not None:
        print(f"Wrote plots to {args.plot_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
