"""Benchmark superposition execution for sparse-support amplitude inputs.

The script builds Fock-space `QuantumLayer` instances with deterministic
interferometer circuits and feeds each layer a normalized amplitude vector. The
vector is represented as a dense tensor for broad compatibility, but only the
first `nnz` entries are non-zero. This gives a repeatable workload where the
logical input support is small while the surrounding Fock basis can be scaled by
changing `n_modes` and `n_photons`.

For each configured case the benchmark records:

* Forward-pass timings after optional warmup runs.
* Output shape and a simple output magnitude checksum.
* Process max RSS reported by `resource.getrusage`.
* Shapes and byte sizes of `torch.zeros` allocations made through
  `merlin.core.process`.

The allocation recorder is intentionally narrow. It does not replace a full
memory profiler; it captures allocation shapes that are useful when comparing
how different versions process superposition inputs. Results are printed as JSON
and can also be written to a file for later comparison.

Use `--include-output` when another tool should compare full complex outputs
between two benchmark runs.

Example:

    PYTHONPATH=$PWD python benchmarks/benchmark_superposition_streaming.py \
        --label local-run \
        --json-out benchmarks/results/local-run.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import resource
import subprocess  # noqa: S404
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import perceval as pcvl
import torch

import merlin.core.process as process_module
from merlin import ComputationSpace, MeasurementStrategy, QuantumLayer


@dataclass(frozen=True)
class Case:
    """One benchmark case.

    The Fock basis size is derived from `(n_modes, n_photons)`. `nnz` controls
    how many amplitudes in the input vector are non-zero, and `chunk_size` is
    forwarded to `QuantumLayer(..., simultaneous_processes=chunk_size)`.
    """

    name: str
    n_modes: int
    n_photons: int
    nnz: int
    chunk_size: int


# These cases are large enough to expose the old quadratic allocation but small
# enough to run quickly on a developer laptop or WSL VM.
DEFAULT_CASES = (
    Case("fock_m16_p3_16nz_chunk8", n_modes=16, n_photons=3, nnz=16, chunk_size=8),
    Case("fock_m28_p3_16nz_chunk8", n_modes=28, n_photons=3, nnz=16, chunk_size=8),
    Case("fock_m32_p3_16nz_chunk8", n_modes=32, n_photons=3, nnz=16, chunk_size=8),
)


def _git_value(args: list[str], repo: Path) -> str:
    """Return git metadata for the benchmark JSON, or ``unknown`` outside git.

    The benchmark can be run from a normal checkout, a detached worktree, or an
    exported source tree. Capturing commit/branch/dirty metadata makes later
    comparisons easier, but benchmark execution should not depend on git being
    available.
    """
    try:
        return subprocess.check_output(  # noqa: S603
            ["git", *args],  # noqa: S607
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _shape_from_zeros_call(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[int, ...] | None:
    shape = args[0] if args else kwargs.get("size")
    if shape is None:
        return None
    if isinstance(shape, int):
        return (shape,)
    try:
        return tuple(int(dim) for dim in shape)
    except TypeError:
        return None


def _dtype_from_zeros_call(kwargs: dict[str, Any]) -> torch.dtype:
    dtype = kwargs.get("dtype")
    return dtype if isinstance(dtype, torch.dtype) else torch.float32


def _element_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _bytes_for(shape: Iterable[int], dtype: torch.dtype) -> int:
    return math.prod(shape) * _element_size(dtype)


class ZeroAllocationRecorder:
    """Record `torch.zeros` allocations made by `merlin.core.process`.

    This is a targeted instrumentation hook, not a full memory profiler. It
    records allocation shapes that are relevant to superposition execution and
    marks whether any allocation spans both the full input basis and the full
    output basis.
    """

    def __init__(self, basis_size: int, output_size: int) -> None:
        self.basis_size = basis_size
        self.output_size = output_size
        self.records: list[dict[str, Any]] = []
        self._original = None

    def __enter__(self) -> ZeroAllocationRecorder:
        self._original = process_module.torch.zeros

        def tracked_zeros(*args, **kwargs):
            shape = _shape_from_zeros_call(args, kwargs)
            dtype = _dtype_from_zeros_call(kwargs)
            if shape is not None:
                allocation = {
                    "shape": list(shape),
                    "dtype": str(dtype).replace("torch.", ""),
                    "bytes": _bytes_for(shape, dtype),
                    "is_whole_support_table": self._is_whole_support_shape(shape),
                }
                self.records.append(allocation)
            return self._original(*args, **kwargs)

        process_module.torch.zeros = tracked_zeros
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._original is not None:
            process_module.torch.zeros = self._original

    def _is_whole_support_shape(self, shape: tuple[int, ...]) -> bool:
        """Return True for allocations spanning input basis by output basis."""
        if shape == (self.basis_size, self.output_size):
            return True
        return (
            len(shape) == 3
            and shape[1] == self.basis_size
            and shape[2] == self.output_size
        )

    def summary(self) -> dict[str, Any]:
        whole_support = [
            record for record in self.records if record["is_whole_support_table"]
        ]
        max_record = max(self.records, key=lambda row: row["bytes"], default=None)
        max_whole = max(whole_support, key=lambda row: row["bytes"], default=None)
        return {
            "zero_allocation_count": len(self.records),
            "max_zero_allocation": max_record,
            "whole_support_allocation_count": len(whole_support),
            "max_whole_support_allocation": max_whole,
            "whole_support_allocations": whole_support,
        }


def _make_layer(case: Case) -> QuantumLayer:
    """Build a deterministic interferometer layer for superposition execution."""
    circuit = pcvl.components.GenericInterferometer(
        case.n_modes,
        pcvl.components.catalog["mzi phase last"].generate,
        shape=pcvl.InterferometerShape.RECTANGLE,
    )
    return QuantumLayer(
        circuit=circuit,
        n_photons=case.n_photons,
        measurement_strategy=MeasurementStrategy.amplitudes(ComputationSpace.FOCK),
        amplitude_encoding=True,
        trainable_parameters=["phi"],
        input_parameters=[],
        dtype=torch.float32,
    )


def _make_dense_input(size: int, nnz: int) -> torch.Tensor:
    """Create a normalized dense vector with only `nnz` non-zero entries."""
    if nnz > size:
        raise ValueError(f"nnz={nnz} exceeds basis size {size}.")
    state = torch.zeros(size, dtype=torch.complex64)
    values = torch.arange(1, nnz + 1, dtype=torch.float32)
    state[:nnz] = torch.complex(values, torch.flip(values, dims=(0,)))
    return state / state.abs().pow(2).sum().sqrt()


def _output_values(output: torch.Tensor) -> dict[str, Any]:
    """Return a JSON-serializable complex output vector for comparison."""
    flat = output.detach().cpu().reshape(-1)
    return {
        "shape": list(output.shape),
        "real": [float(value) for value in flat.real.tolist()],
        "imag": [float(value) for value in flat.imag.tolist()],
    }


def _run_case(
    case: Case, runs: int, warmups: int, *, include_output: bool
) -> dict[str, Any]:
    """Run one case and return JSON-serializable benchmark data.

    Warmup runs are executed under the allocation recorder because they should
    exercise the same memory path as measured runs, but they are excluded from
    the reported timing statistics. A final forward pass outside the recorder
    checks output shape/value metadata after timings are collected.
    """
    torch.manual_seed(1234)
    layer = _make_layer(case)
    basis_size = len(layer.output_keys)
    amplitude_input = _make_dense_input(basis_size, case.nnz)

    times: list[float] = []
    output: torch.Tensor | None = None
    gc.collect()

    with ZeroAllocationRecorder(basis_size, layer.output_size) as recorder:
        for run_index in range(warmups + runs):
            start = time.perf_counter()
            output = layer(amplitude_input, simultaneous_processes=case.chunk_size)
            elapsed = time.perf_counter() - start
            if run_index >= warmups:
                times.append(elapsed)
            del output
            gc.collect()

    output = layer(amplitude_input, simultaneous_processes=case.chunk_size)
    rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    result = {
        **asdict(case),
        "basis_size": basis_size,
        "output_size": layer.output_size,
        "circuit": "generic_interferometer_mzi_phase_last_rectangle",
        "input_layout": "dense_with_sparse_support",
        "dtype": "complex64",
        "runs": runs,
        "warmups": warmups,
        "times_s": times,
        "mean_s": mean(times),
        "min_s": min(times),
        "max_s": max(times),
        "rss_max_kib": rss_kib,
        "output_shape": list(output.shape),
        "output_l1": float(output.abs().sum().item()),
        "allocations": recorder.summary(),
    }
    if include_output:
        result["output_values"] = _output_values(output)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="Human label for this run.")
    parser.add_argument("--json-out", type=Path, help="Optional JSON output path.")
    parser.add_argument(
        "--commit", help="Optional commit override for detached worktrees."
    )
    parser.add_argument("--branch", help="Optional branch/base label override.")
    parser.add_argument(
        "--dirty",
        choices=("true", "false"),
        help="Optional dirty-state override for detached worktrees.",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--include-output",
        action="store_true",
        help="Include full complex output values in JSON for result comparison.",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=[case.name for case in DEFAULT_CASES],
        help="Run one or more named cases. Defaults to all cases.",
    )
    return parser.parse_args()


def main() -> int:
    """Run selected cases and print/write one JSON payload."""
    args = parse_args()
    repo = Path.cwd()
    selected = set(args.case or [case.name for case in DEFAULT_CASES])
    cases = [case for case in DEFAULT_CASES if case.name in selected]
    status = _git_value(["status", "--porcelain"], repo)
    is_dirty = status not in ("", "unknown")
    if args.dirty is not None:
        is_dirty = args.dirty == "true"

    payload = {
        "label": args.label,
        "repo": str(repo),
        "commit": args.commit or _git_value(["rev-parse", "HEAD"], repo),
        "branch": args.branch
        or _git_value(["branch", "--show-current"], repo)
        or "detached",
        "is_dirty": is_dirty,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "pid": os.getpid(),
        "cases": [
            _run_case(
                case,
                args.runs,
                args.warmups,
                include_output=args.include_output,
            )
            for case in cases
        ],
    }

    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
