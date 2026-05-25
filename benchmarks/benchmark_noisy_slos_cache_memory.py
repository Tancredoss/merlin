"""Measure NoisySLOS input-cache memory growth.

This script builds a noisy :class:`merlin.QuantumLayer`, prepares an
amplitude-encoded equal superposition over the full Fock basis, and measures
how much memory is consumed when that forward pass populates
``NoisySLOSComputeGraph._slos_graph_per_input``.

Two complementary memory indicators are reported:

- ``cache_size_mb``: a recursive estimate of the cached object graph size,
  including tensor storage.
- ``tracemalloc_delta_mb``: Python allocation growth measured while populating
  the cache.

The recursive size estimate is the more relevant metric for the cached Merlin
objects. ``tracemalloc`` is included as a sanity check for Python-level growth.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import tracemalloc
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import perceval as pcvl
import torch

from merlin import ComputationSpace, MeasurementStrategy, QuantumLayer
from merlin.pcvl_pytorch.noisy_slos import NoisySLOSComputeGraph
from merlin.utils.combinadics import Combinadics


@dataclass
class CacheMemoryResult:
    """Memory measurements for one ``(modes, photons)`` configuration.

    Parameters
    ----------
    modes : int
        Number of optical modes.
    photons : int
        Number of photons.
    n_input_states : int
        Number of Fock input states used to populate the cache.
    cache_entries : int
        Number of cached entries in ``_slos_graph_per_input``.
    cache_size_mb : float
        Recursive size estimate for the cache, in MiB.
    graph_size_mb : float
        Recursive size estimate for the full noisy graph, in MiB.
    layer_size_mb : float
        Recursive size estimate for the entire QuantumLayer, in MiB.
    tracemalloc_delta_mb : float
        Python allocation growth while populating the cache, in MiB.
    tracemalloc_peak_mb : float
        Peak Python allocation observed while populating the cache, in MiB.
    cache_size_after_backward_mb : float
        Recursive cache size estimate after backward pass, in MiB.
    tracemalloc_delta_backward_mb : float
        Python allocation growth during backward pass, in MiB.
    """

    modes: int
    photons: int
    n_input_states: int
    cache_entries: int
    cache_size_mb: float
    graph_size_mb: float
    layer_size_mb: float
    tracemalloc_delta_mb: float
    tracemalloc_peak_mb: float
    cache_size_after_backward_mb: float = 0.0
    tracemalloc_delta_backward_mb: float = 0.0


@dataclass
class ExponentialFitResult:
    """Fitted coefficients and quality metrics for an exponential model.

    Parameters
    ----------
    intercept : float
        Constant term ``a`` in ``ln(y) = a + b*n + c*m + d*n*m``.
    coef_photons : float
        Coefficient ``b`` multiplying the photon count ``n``.
    coef_modes : float
        Coefficient ``c`` multiplying the mode count ``m``.
    coef_interaction : float
        Coefficient ``d`` multiplying the interaction term ``n*m``.
    r2_log : float
        Coefficient of determination on the log-transformed target.
    r2_original : float
        Coefficient of determination on the original target scale.
    """

    intercept: float
    coef_photons: float
    coef_modes: float
    coef_interaction: float
    r2_log: float
    r2_original: float


def _build_fixed_circuit(modes: int) -> pcvl.Circuit:
    """Build a parameter-free circuit for cache population.

    Parameters
    ----------
    modes : int
        Number of optical modes.

    Returns
    -------
    pcvl.Circuit
        A simple chain of Hadamard beam splitters.
    """

    circuit = pcvl.Circuit(modes)
    for mode in range(modes - 1):
        circuit.add((mode, mode + 1), pcvl.BS.H())
    return circuit


def _build_noisy_layer(modes: int, photons: int, dtype: torch.dtype) -> QuantumLayer:
    """Create a noisy ``QuantumLayer`` whose backend is ``NoisySLOSComputeGraph``.

    Parameters
    ----------
    modes : int
        Number of optical modes.
    photons : int
        Number of photons.
    dtype : torch.dtype
        Merlin real dtype.

    Returns
    -------
    QuantumLayer
        Layer configured with source noise, Fock-space probabilities, and
        amplitude encoding enabled.
    """

    return QuantumLayer(
        circuit=_build_fixed_circuit(modes),
        n_photons=photons,
        noise=pcvl.NoiseModel(indistinguishability=0.5),
        measurement_strategy=MeasurementStrategy.probs(
            computation_space=ComputationSpace.FOCK
        ),
        amplitude_encoding=True,
        dtype=dtype,
    )


def _get_noisy_graph(layer: QuantumLayer) -> NoisySLOSComputeGraph:
    """Extract the noisy SLOS graph from a layer.

    Parameters
    ----------
    layer : QuantumLayer
        Layer configured with source noise.

    Returns
    -------
    NoisySLOSComputeGraph
        Underlying noisy graph.

    Raises
    ------
    TypeError
        If the layer does not use ``NoisySLOSComputeGraph``.
    """

    graph = layer.computation_process.simulation_graph
    if not isinstance(graph, NoisySLOSComputeGraph):
        raise TypeError(
            "Expected QuantumLayer to use NoisySLOSComputeGraph, "
            f"got {type(graph).__name__}."
        )
    return graph


def _tensor_storage_nbytes(tensor: torch.Tensor) -> int:
    """Return tensor storage size in bytes.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor whose backing storage should be measured.

    Returns
    -------
    int
        Number of bytes owned by the tensor storage.
    """

    storage = tensor.untyped_storage()
    return int(storage.nbytes())


def _deep_size_bytes(
    obj: object,
    *,
    seen_ids: set[int] | None = None,
    seen_storages: set[int] | None = None,
) -> int:
    """Estimate the recursive size of a Python object graph.

    Parameters
    ----------
    obj : object
        Object to inspect.
    seen_ids : set[int] | None
        Visited Python object ids.
    seen_storages : set[int] | None
        Visited tensor storage pointers.

    Returns
    -------
    int
        Estimated recursive size in bytes.
    """

    if seen_ids is None:
        seen_ids = set()
    if seen_storages is None:
        seen_storages = set()

    obj_id = id(obj)
    if obj_id in seen_ids:
        return 0
    seen_ids.add(obj_id)

    size = sys.getsizeof(obj)

    if isinstance(obj, torch.Tensor):
        storage_ptr = obj.untyped_storage().data_ptr()
        if storage_ptr not in seen_storages:
            seen_storages.add(storage_ptr)
            size += _tensor_storage_nbytes(obj)
        return size

    if isinstance(obj, dict):
        for key, value in obj.items():
            size += _deep_size_bytes(
                key, seen_ids=seen_ids, seen_storages=seen_storages
            )
            size += _deep_size_bytes(
                value, seen_ids=seen_ids, seen_storages=seen_storages
            )
        return size

    if isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            size += _deep_size_bytes(
                item, seen_ids=seen_ids, seen_storages=seen_storages
            )
        return size

    if hasattr(obj, "__dict__"):
        size += _deep_size_bytes(
            vars(obj), seen_ids=seen_ids, seen_storages=seen_storages
        )
        return size

    return size


def _enumerate_fock_inputs(modes: int, photons: int) -> list[tuple[int, ...]]:
    """Enumerate all Fock basis states for a configuration.

    Parameters
    ----------
    modes : int
        Number of optical modes.
    photons : int
        Number of photons.

    Returns
    -------
    list[tuple[int, ...]]
        All Fock basis states in Merlin's descending lexicographic order.
    """

    return Combinadics("fock", n=photons, m=modes).enumerate_states()


def _build_equal_superposition_input(
    modes: int, photons: int, dtype: torch.dtype
) -> torch.Tensor:
    """Build a normalized equal-superposition amplitude input.

    Parameters
    ----------
    modes : int
        Number of optical modes.
    photons : int
        Number of photons.
    dtype : torch.dtype
        Real dtype for the amplitude vector.

    Returns
    -------
    torch.Tensor
        One-dimensional normalized amplitude vector covering the full Fock basis.
    """

    n_basis_states = len(_enumerate_fock_inputs(modes, photons))
    amplitude_input = torch.ones(n_basis_states, dtype=dtype)
    amplitude_input = amplitude_input / amplitude_input.norm(p=2)
    return amplitude_input


def measure_cache_memory(
    modes: int,
    photons: int,
    dtype: torch.dtype,
    *,
    verbose: bool = False,
    measure_backward: bool = False,
) -> CacheMemoryResult:
    """Populate the noisy cache and return memory measurements.

    Parameters
    ----------
    modes : int
        Number of optical modes.
    photons : int
        Number of photons.
    dtype : torch.dtype
        Merlin real dtype.
    verbose : bool
        If True, print the resulting cache keys after the amplitude-encoded
        forward pass.
    measure_backward : bool
        If True, also measure memory after backward pass through a simple loss.

    Returns
    -------
    CacheMemoryResult
        Memory usage summary for the configuration.
    """

    gc.collect()
    layer = _build_noisy_layer(modes, photons, dtype)
    graph = _get_noisy_graph(layer)
    amplitude_input = _build_equal_superposition_input(modes, photons, dtype)
    input_states = _enumerate_fock_inputs(modes, photons)

    tracemalloc.start()
    baseline_current, _ = tracemalloc.get_traced_memory()

    if measure_backward:
        amplitude_input.requires_grad = True

    output = layer(amplitude_input)

    if verbose:
        cache_keys = list(graph._slos_graph_per_input.keys())
        print(f"  cached_inputs={cache_keys}")
        print(
            f"  cache_entries={len(cache_keys)} "
            f"cache_size_mb={_deep_size_bytes(graph._slos_graph_per_input) / 1024**2:.4f}"
        )

    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    forward_delta_mb = (current_bytes - baseline_current) / 1024**2

    # Measure backward pass if requested
    cache_size_after_backward_mb = 0.0
    backward_delta_mb = 0.0

    if measure_backward:
        loss = output.sum()  # Simple loss: sum of all output probabilities

        tracemalloc.start()
        backward_baseline, _ = tracemalloc.get_traced_memory()

        loss.backward()

        backward_current, backward_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        cache_size_after_backward_mb = (
            _deep_size_bytes(graph._slos_graph_per_input) / 1024**2
        )
        backward_delta_mb = (backward_current - backward_baseline) / 1024**2

        if verbose:
            print(
                f"  after_backward: cache_size_mb={cache_size_after_backward_mb:.4f} "
                f"delta_mb={backward_delta_mb:.4f}"
            )

    return CacheMemoryResult(
        modes=modes,
        photons=photons,
        n_input_states=len(input_states),
        cache_entries=len(graph._slos_graph_per_input),
        cache_size_mb=_deep_size_bytes(graph._slos_graph_per_input) / 1024**2,
        graph_size_mb=_deep_size_bytes(graph) / 1024**2,
        layer_size_mb=_deep_size_bytes(layer) / 1024**2,
        tracemalloc_delta_mb=forward_delta_mb,
        tracemalloc_peak_mb=peak_bytes / 1024**2,
        cache_size_after_backward_mb=cache_size_after_backward_mb,
        tracemalloc_delta_backward_mb=backward_delta_mb,
    )


def _parse_range(values: Iterable[int] | None, *, default: list[int]) -> list[int]:
    """Normalize CLI ranges.

    Parameters
    ----------
    values : Iterable[int] | None
        CLI values.
    default : list[int]
        Default list to use when ``values`` is omitted.

    Returns
    -------
    list[int]
        Normalized integer list.
    """

    if values is None:
        return default
    return list(values)


def _plot_results(results: list[CacheMemoryResult], output_path: str) -> None:
    """Plot output dimension versus cache and graph memory.

    Parameters
    ----------
    results : list[CacheMemoryResult]
        Benchmark results to plot.
    output_path : str
        Destination image path.

    Raises
    ------
    RuntimeError
        If matplotlib is not installed.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib. Install it or rerun without --plot."
        ) from exc

    sorted_results = sorted(
        results,
        key=lambda result: (result.n_input_states, result.modes, result.photons),
    )
    output_dims = [result.n_input_states for result in sorted_results]
    cache_sizes = [result.cache_size_mb for result in sorted_results]
    graph_sizes = [result.graph_size_mb for result in sorted_results]
    labels = [f"m={result.modes}, n={result.photons}" for result in sorted_results]

    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(output_dims, cache_sizes, marker="o", linewidth=2, label="cache_mb")
    axis.plot(output_dims, graph_sizes, marker="s", linewidth=2, label="graph_mb")

    for output_dim, cache_size, label in zip(
        output_dims, cache_sizes, labels, strict=False
    ):
        axis.annotate(
            label,
            (output_dim, cache_size),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    axis.set_xlabel("Output dimension")
    axis.set_ylabel("Memory (MiB)")
    axis.set_title("NoisySLOS cache memory vs output dimension")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def _fit_exponential_model(
    results: list[CacheMemoryResult], target_attr: str
) -> ExponentialFitResult:
    r"""Fit an exponential rule against photons and modes.

    The fitted model is:

    .. math::
        y = \exp(a + b\,n + c\,m + d\,n m)

    where ``n`` is the number of photons and ``m`` is the number of modes.

    Parameters
    ----------
    results : list[CacheMemoryResult]
        Benchmark rows used for fitting.
    target_attr : str
        Name of the target attribute in ``CacheMemoryResult`` to fit.

    Returns
    -------
    ExponentialFitResult
        Estimated coefficients and fit quality.
    """

    if not results:
        raise ValueError("Cannot fit exponential model with no benchmark results.")

    photons = torch.tensor([result.photons for result in results], dtype=torch.float64)
    modes = torch.tensor([result.modes for result in results], dtype=torch.float64)
    targets = torch.tensor(
        [float(getattr(result, target_attr)) for result in results], dtype=torch.float64
    )

    design = torch.stack(
        [
            torch.ones_like(photons),
            photons,
            modes,
            photons * modes,
        ],
        dim=1,
    )

    log_targets = torch.log(targets.clamp_min(1e-12))
    coefficients = torch.linalg.lstsq(
        design, log_targets.unsqueeze(1)
    ).solution.squeeze(1)

    log_predictions = design @ coefficients
    predictions = torch.exp(log_predictions)

    sse_log = torch.sum((log_targets - log_predictions) ** 2)
    sst_log = torch.sum((log_targets - log_targets.mean()) ** 2)
    r2_log = 1.0 - (sse_log / sst_log)

    sse_original = torch.sum((targets - predictions) ** 2)
    sst_original = torch.sum((targets - targets.mean()) ** 2)
    r2_original = 1.0 - (sse_original / sst_original)

    return ExponentialFitResult(
        intercept=float(coefficients[0].item()),
        coef_photons=float(coefficients[1].item()),
        coef_modes=float(coefficients[2].item()),
        coef_interaction=float(coefficients[3].item()),
        r2_log=float(r2_log.item()),
        r2_original=float(r2_original.item()),
    )


def _format_exponential_formula(fit: ExponentialFitResult, target_name: str) -> str:
    """Format an exponential model formula as a printable string.

    Parameters
    ----------
    fit : ExponentialFitResult
        Fitted coefficients.
    target_name : str
        Name of the target variable.

    Returns
    -------
    str
        Formula string using ``m`` for modes and ``n`` for photons.
    """

    return (
        f"{target_name}(m,n) = exp("
        f"{fit.intercept:+.8f} "
        f"{fit.coef_photons:+.8f}*n "
        f"{fit.coef_modes:+.8f}*m "
        f"{fit.coef_interaction:+.8f}*n*m)"
    )


def main() -> None:
    """Run the cache-memory benchmark from the command line."""

    parser = argparse.ArgumentParser(
        description=(
            "Measure NoisySLOS `_slos_graph_per_input` memory growth for "
            "increasing numbers of modes and photons."
        )
    )
    parser.add_argument(
        "--modes",
        type=int,
        nargs="+",
        default=None,
        help="Mode counts to benchmark. Default: 2 4 6 8",
    )
    parser.add_argument(
        "--photons",
        type=int,
        nargs="+",
        default=None,
        help="Photon counts to benchmark. Default: 1 2 3 4",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Merlin float dtype.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print cache growth after each newly populated input state.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the results as JSON instead of a table.",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help=(
            "Generate a plot of output dimension versus cache_mb and graph_mb. "
            "Provide the output image path."
        ),
    )
    parser.add_argument(
        "--fit-exponential",
        action="store_true",
        help=(
            "Fit exponential rules for cache_mb and graph_mb as functions of "
            "photons (n) and modes (m), then print formulas and R^2."
        ),
    )
    parser.add_argument(
        "--backward",
        action="store_true",
        help="Also measure memory after backward pass through a simple loss.",
    )
    args = parser.parse_args()

    modes_values = _parse_range(args.modes, default=[2, 4, 6, 8])
    photon_values = _parse_range(args.photons, default=[1, 2, 3, 4])
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    results: list[CacheMemoryResult] = []
    for modes in modes_values:
        for photons in photon_values:
            if photons > modes:
                continue
            print(f"Benchmarking modes={modes}, photons={photons}...")
            result = measure_cache_memory(
                modes,
                photons,
                dtype,
                verbose=args.verbose,
                measure_backward=args.backward,
            )
            results.append(result)

    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
        return

    header = (
        "modes photons n_inputs cache_entries cache_mb graph_mb layer_mb "
        "tracemalloc_delta_mb tracemalloc_peak_mb"
    )
    print(header)
    for result in results:
        print(
            f"{result.modes:>5} {result.photons:>7} {result.n_input_states:>8} "
            f"{result.cache_entries:>13} {result.cache_size_mb:>8.4f} "
            f"{result.graph_size_mb:>8.4f} {result.layer_size_mb:>8.4f} "
            f"{result.tracemalloc_delta_mb:>20.4f} "
            f"{result.tracemalloc_peak_mb:>19.4f}"
        )

    if args.plot is not None:
        _plot_results(results, args.plot)
        print(f"Saved plot to {args.plot}")

    if args.backward:
        print("\nBackward pass memory measurements:")
        header_backward = "modes photons cache_after_backward_mb delta_backward_mb"
        print(header_backward)
        for result in results:
            print(
                f"{result.modes:>5} {result.photons:>7} "
                f"{result.cache_size_after_backward_mb:>25.4f} "
                f"{result.tracemalloc_delta_backward_mb:>21.4f}"
            )

    if args.fit_exponential:
        cache_fit = _fit_exponential_model(results, target_attr="cache_size_mb")
        graph_fit = _fit_exponential_model(results, target_attr="graph_size_mb")

        print("\nExponential fit (target = exp(a + b*n + c*m + d*n*m))")
        print(_format_exponential_formula(cache_fit, "cache_mb"))
        print(
            f"  R2_log={cache_fit.r2_log:.6f} "
            f"R2_original={cache_fit.r2_original:.6f}"
        )
        print(_format_exponential_formula(graph_fit, "graph_mb"))
        print(
            f"  R2_log={graph_fit.r2_log:.6f} "
            f"R2_original={graph_fit.r2_original:.6f}"
        )


if __name__ == "__main__":
    main()
