# MIT License
#
# Copyright (c) 2025 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Utilities and helpers for QuantumLayer initialization and configuration.
"""

# Usage outline (mirrors QuantumLayer.__init__ phases):
# 1) validate_and_resolve_circuit_source -> obtain prefixes/specs
# 2) validate_encoding_mode -> enforce amplitude/classical constraints
# 3) prepare_input_state -> normalize input state (incl. experiment override)
# 4) vet_experiment -> reject unsupported experiments
# 5) resolve_circuit -> build circuit/experiment wrapper
# 6) setup_noise_and_detectors -> extract noise/detectors + compatibility checks
# 7) Encoding/output utilities -> used during parameter prep & forward

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import exqalibur as xqlbr
import perceval as pcvl
import torch
from perceval.components import PS, AComponent

from ..builder.circuit_builder import CircuitBuilder
from ..core.computation_space import ComputationSpace
from ..core.partial_measurement import PartialMeasurement
from ..core.probability_distribution import ProbabilityDistribution
from ..core.state import StatePattern, _generate_default_input_state, generate_state
from ..core.state_vector import StateVector
from ..measurement.detectors import resolve_detectors
from ..measurement.photon_loss import resolve_photon_loss
from ..measurement.strategies import (
    MeasurementKind,
    MeasurementStrategyLike,
    _resolve_measurement_kind,
)

_CONSTRUCTOR_AMPLITUDE_ENCODING_REMOVAL_MESSAGE = (
    "amplitude_encoding=True was removed in 0.4. Pass amplitude data to "
    "forward(StateVector) or forward(complex_tensor) instead. Convert "
    "constructor tensors with StateVector.from_tensor() when a StateVector "
    "object is needed."
)

_CONSTRUCTOR_TENSOR_INPUT_STATE_REMOVAL_MESSAGE = (
    "torch.Tensor is no longer accepted as QuantumLayer input_state. Pass "
    "amplitude data to forward(StateVector) or forward(complex_tensor) instead. "
    "Convert constructor tensors with StateVector.from_tensor() when a "
    "StateVector object is needed."
)


@dataclass(frozen=True)
class EncodingModeConfig:
    """Store the validated encoding configuration.

    Parameters
    ----------
    amplitude_encoding : bool
        Whether amplitude encoding is enabled.
    input_size : int | None
        Resolved classical input size.
    n_photons : int | None
        Resolved photon count.
    input_parameters : list[str]
        Resolved list of input parameter prefixes.
    """

    amplitude_encoding: bool
    input_size: int | None
    n_photons: int | None
    input_parameters: list[str]


@dataclass(frozen=True)
class CircuitSource:
    """Store the resolved circuit source configuration.

    Parameters
    ----------
    source_type : Literal["builder", "circuit", "experiment"]
        Kind of source provided by the caller.
    builder : CircuitBuilder | None
        Builder instance when ``source_type == "builder"``.
    circuit : pcvl.Circuit | None
        Perceval circuit when ``source_type == "circuit"``.
    experiment : pcvl.Experiment | None
        Perceval experiment when ``source_type == "experiment"``.
    trainable_parameters : list[str]
        Resolved trainable parameter prefixes.
    input_parameters : list[str]
        Resolved input parameter prefixes.
    angle_encoding_specs : dict[str, dict[str, Any]]
        Stored angle encoding metadata extracted from the builder, if any.
    """

    source_type: Literal["builder", "circuit", "experiment"]
    builder: CircuitBuilder | None
    circuit: pcvl.Circuit | None
    experiment: pcvl.Experiment | None
    trainable_parameters: list[str]
    input_parameters: list[str]
    angle_encoding_specs: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ResolvedCircuit:
    """Store the resolved circuit and experiment pair.

    Parameters
    ----------
    circuit : pcvl.Circuit
        Resolved circuit instance.
    experiment : pcvl.Experiment
        Experiment wrapping the resolved circuit.
    noise : Any | None
        Attached experiment noise model, if present.
    has_custom_noise : bool
        Whether the experiment exposes a non-empty custom noise model.
    """

    circuit: pcvl.Circuit
    experiment: pcvl.Experiment
    noise: Any | None
    has_custom_noise: bool


@dataclass(frozen=True)
class NoiseAndDetectorConfig:
    """Store extracted noise and detector configuration.

    Parameters
    ----------
    photon_survival_probs : list[float]
        Photon survival probabilities derived from the experiment noise model.
    has_custom_noise : bool
        Whether a custom noise model is present.
    detectors : list[pcvl.Detector]
        Resolved detector list for every mode.
    has_custom_detectors : bool
        Whether the experiment defines non-default detectors.
    detector_warnings : list[str]
        Compatibility warnings emitted while resolving detector behavior.
    noise_groups: NoiseGroups | None
        The noise groups applied to the circuit to be ran.
    """

    photon_survival_probs: list[float]
    has_custom_noise: bool
    detectors: list[pcvl.Detector]
    has_custom_detectors: bool
    detector_warnings: list[str]
    noise_groups: NoiseGroups | None


@dataclass(frozen=True)
class NoiseGroups:
    """Stores the classified noise sources

    Parameters
    ----------
    source : dict[str, float | bool | None] | None
        Source noises e.g. {"indistinguishability": 0.9, "g2": 0.05, "g2_distinguishable": True},
        or ``None`` when no source noise is present.
    circuit : dict[str, float | None] | None
        Circuit noises e.g. {"phase_error": 0.05, "phase_imprecision": 0.1},
        or ``None`` when no circuit noise is present.
    post_measurement : dict[str, float | None] | None
        Post measurement noises e.g. {"brightness": 0.3, "transmittance": 0.9},
        or ``None`` when no post-measurement noise is present.
    """

    source: dict[str, float | bool | None] | None
    circuit: dict[str, float | None] | None
    post_measurement: dict[str, float | None] | None


def has_active_noise(noise_groups: NoiseGroups | None) -> bool:
    """Return whether any classified noise group is active.

    Parameters
    ----------
    noise_groups : NoiseGroups | None
        Classified noise groups returned by :func:`classify_noise`.

    Returns
    -------
    bool
        True when at least one source, circuit, or post-measurement noise group
        is active.
    """
    if noise_groups is None:
        return False
    return bool(
        noise_groups.source or noise_groups.circuit or noise_groups.post_measurement
    )


def has_source_noise(noise_groups: NoiseGroups | None) -> bool:
    """Return whether source noise is active.

    Parameters
    ----------
    noise_groups : NoiseGroups | None
        Classified noise groups returned by :func:`classify_noise`.

    Returns
    -------
    bool
        True when a source noise group is present.
    """
    return noise_groups is not None and bool(noise_groups.source)


def has_circuit_noise(noise_groups: NoiseGroups | None) -> bool:
    """Return whether circuit noise is active.

    Parameters
    ----------
    noise_groups : NoiseGroups | None
        Classified noise groups returned by :func:`classify_noise`.

    Returns
    -------
    bool
        True when a circuit noise group is present.
    """
    return noise_groups is not None and bool(noise_groups.circuit)


def has_phase_error(noise_groups: NoiseGroups | None) -> bool:
    """Return whether stochastic phase-error noise is active.

    Parameters
    ----------
    noise_groups : NoiseGroups | None
        Classified noise groups returned by :func:`classify_noise`.

    Returns
    -------
    bool
        True when ``phase_error`` is present in the circuit noise group. A
        ``None`` value means the stochastic half-width is stored locally on one
        or more Perceval phase shifters.
    """
    if not has_circuit_noise(noise_groups):
        return False
    circuit_noise = cast(dict[str, float | None], noise_groups.circuit)
    return "phase_error" in circuit_noise


def _phase_shifter_max_error(component: AComponent) -> float:
    """Return the local phase-error half-width stored on a Perceval phase shifter.

    Parameters
    ----------
    component : AComponent
        Component to inspect.

    Returns
    -------
    float
        Positive local phase-error half-width, or 0.0 when no local stochastic
        phase-error is configured.
    """
    if not isinstance(component, PS):
        return 0.0
    return float(getattr(component, "_max_error", 0.0))


def _circuit_has_phase_error(circuit: pcvl.Circuit) -> bool:
    """Return whether a circuit contains local phase-shifter error.

    Parameters
    ----------
    circuit : pcvl.Circuit
        Circuit whose components are inspected.

    Returns
    -------
    bool
        True when at least one :class:`pcvl.PS` component has a positive
        ``max_error`` value.
    """
    return any(_phase_shifter_max_error(component) > 0.0 for _, component in circuit)


def _with_component_phase_error(
    noise_groups: NoiseGroups | None,
) -> NoiseGroups:
    """Mark classified noise groups as containing local phase-error noise.

    Parameters
    ----------
    noise_groups : NoiseGroups | None
        Existing noise groups derived from a :class:`pcvl.NoiseModel`, if any.

    Returns
    -------
    NoiseGroups
        Noise groups with ``phase_error`` present in the circuit group. A
        ``None`` value is used when the stochastic width is provided by
        individual phase shifters rather than by :class:`pcvl.NoiseModel`.
    """
    if noise_groups is None:
        return NoiseGroups(
            source=None,
            circuit={"phase_error": None},
            post_measurement=None,
        )

    circuit_noise = dict(noise_groups.circuit or {})
    circuit_noise.setdefault("phase_error", None)
    return NoiseGroups(
        source=noise_groups.source,
        circuit=circuit_noise,
        post_measurement=noise_groups.post_measurement,
    )


@dataclass(frozen=True)
class InitializationContext:
    """Store immutable QuantumLayer initialization state.

    Parameters
    ----------
    device : torch.device | None
        Target device for the layer.
    dtype : torch.dtype
        Real dtype used by the layer.
    complex_dtype : torch.dtype
        Complex dtype paired with ``dtype``.
    amplitude_encoding : bool
        Whether amplitude encoding is enabled.
    input_size : int | None
        Resolved classical input size.
    circuit : pcvl.Circuit
        Resolved circuit.
    experiment : pcvl.Experiment
        Resolved experiment.
    noise : Any | None
        Attached noise model, if any.
    has_custom_noise : bool
        Whether the experiment defines custom noise.
    input_state : merlin.core.state_vector.StateVector | pcvl.BasicState | None
        Normalized input state.
    n_photons : int | None
        Resolved photon count.
    trainable_parameters : list[str]
        Trainable parameter prefixes.
    input_parameters : list[str]
        Classical input parameter prefixes.
    angle_encoding_specs : dict[str, dict[str, Any]]
        Angle encoding metadata extracted from the builder.
    photon_survival_probs : list[float]
        Photon survival probabilities derived from the experiment.
    detectors : list[pcvl.Detector]
        Resolved detector list.
    has_custom_detectors : bool
        Whether custom detectors are configured.
    computation_space : ComputationSpace
        Resolved computation space.
    measurement_strategy : :data:`~merlin.measurement.strategies.MeasurementStrategyLike`
        Measurement strategy used by the layer.
    warnings : list[str]
        Initialization warnings to surface to the caller.
    return_object : bool
        Whether the layer returns structured objects instead of tensors.
    noise_groups: NoiseGroups | None
        The noise groups applied to the circuit to be ran.
    """

    device: torch.device | None
    dtype: torch.dtype
    complex_dtype: torch.dtype
    amplitude_encoding: bool
    input_size: int | None
    circuit: pcvl.Circuit
    experiment: pcvl.Experiment
    noise: Any | None
    has_custom_noise: bool
    input_state: StateVector | pcvl.BasicState | None
    n_photons: int | None
    trainable_parameters: list[str]
    input_parameters: list[str]
    angle_encoding_specs: dict[str, dict[str, Any]]
    photon_survival_probs: list[float]
    detectors: list[pcvl.Detector]
    has_custom_detectors: bool
    computation_space: ComputationSpace
    measurement_strategy: MeasurementStrategyLike
    warnings: list[str]
    return_object: bool
    noise_groups: NoiseGroups | None
    n_phase_error_samples: int


def validate_encoding_mode(
    amplitude_encoding: bool,
    input_size: int | None,
    n_photons: int | None,
    input_parameters: list[str] | None,
) -> EncodingModeConfig:
    """Validate amplitude-encoding constraints.

    Parameters
    ----------
    amplitude_encoding : bool
        Removed compatibility flag. If True, a clear migration error is raised.
    input_size : int | None
        User-provided classical input size.
    n_photons : int | None
        User-provided photon count.
    input_parameters : list[str] | None
        User-provided classical input parameter prefixes.

    Returns
    -------
    EncodingModeConfig
        Validated and normalized encoding configuration.

    Raises
    ------
    ValueError
        If amplitude encoding is requested.
    """
    resolved_input_params = list(input_parameters) if input_parameters else []

    if amplitude_encoding:
        raise ValueError(_CONSTRUCTOR_AMPLITUDE_ENCODING_REMOVAL_MESSAGE)
    else:
        resolved_input_size = int(input_size) if input_size is not None else None

    return EncodingModeConfig(
        amplitude_encoding=amplitude_encoding,
        input_size=resolved_input_size,
        n_photons=n_photons,
        input_parameters=resolved_input_params,
    )


def prepare_input_state(
    input_state: (
        StateVector
        | pcvl.StateVector
        | pcvl.BasicState
        | list
        | tuple
        | torch.Tensor
        | None
    ),
    n_photons: int | None,
    computation_space: ComputationSpace,
    device: torch.device | None,
    complex_dtype: torch.dtype,
    experiment: pcvl.Experiment | None = None,
    circuit_m: int | None = None,
    amplitude_encoding: bool = False,
) -> tuple[StateVector | pcvl.BasicState | None, int | None]:
    """Normalize input_state to canonical form.

    Parameters
    ----------
    input_state : :class:`~merlin.core.state_vector.StateVector` | pcvl.StateVector | pcvl.BasicState | list | tuple | torch.Tensor | None
        The input state in various formats. :class:`~merlin.core.state_vector.StateVector` is the canonical type.
        Legacy tensor constructor inputs are rejected. Pass amplitude tensors to
        ``forward()`` or wrap them with ``StateVector.from_tensor()``.
    n_photons : int | None
        Number of photons (used for default state generation).
    computation_space : ComputationSpace
        The computation space configuration.
    device : torch.device | None
        Target device for tensors.
    complex_dtype : torch.dtype
        Complex dtype for tensor conversion.
    experiment : pcvl.Experiment | None
        Optional experiment whose input_state takes precedence.
    circuit_m : int | None
        Number of modes in the circuit (for default state generation).
    amplitude_encoding : bool
        Removed compatibility flag. If True, a clear migration error is raised.

    Returns
    -------
    tuple[merlin.core.state_vector.StateVector | pcvl.BasicState | None, int | None]
        The normalized input state and resolved photon count.

    Raises
    ------
    ValueError
        If amplitude encoding is requested, if ``torch.Tensor`` is passed as
        ``input_state``, if neither input_state nor n_photons is provided, or if
        StateVector is empty.

    Warns
    -----
    UserWarning
        When both experiment.input_state and input_state are provided.
    """
    if amplitude_encoding:
        raise ValueError(_CONSTRUCTOR_AMPLITUDE_ENCODING_REMOVAL_MESSAGE)

    # Experiment input_state takes precedence
    if experiment is not None and experiment.input_state is not None:
        if input_state is not None and experiment.input_state != input_state:
            warnings.warn(
                "Both 'experiment.input_state' and 'input_state' are provided. "
                "'experiment.input_state' will be used.",
                UserWarning,
                stacklevel=2,
            )
        input_state = experiment.input_state

    # === Handle StateVector (canonical, preferred) ===
    if isinstance(input_state, StateVector):
        return input_state, input_state.n_photons

    # === Reject removed tensor constructor state ===
    if isinstance(input_state, torch.Tensor):
        raise ValueError(_CONSTRUCTOR_TENSOR_INPUT_STATE_REMOVAL_MESSAGE)

    # === Handle tuple/list (convert to BasicState) ===
    if isinstance(input_state, tuple):
        input_state = list(input_state)

    # === Handle pcvl.BasicState ===
    if isinstance(input_state, pcvl.BasicState):
        if not isinstance(input_state, xqlbr.FockState):
            raise ValueError("BasicState with annotations is not supported")
        return input_state, n_photons

    # === Handle pcvl.StateVector ===
    elif isinstance(input_state, pcvl.StateVector):
        if len(input_state) == 0:
            raise ValueError("input_state StateVector cannot be empty")
        sv_n_photons = input_state.n.pop()
        if n_photons is not None and sv_n_photons != n_photons:
            raise ValueError(
                "Inconsistent number of photons between input_state and n_photons."
            )
        return StateVector.from_perceval(
            input_state,
            device=device,
            dtype=complex_dtype,
        ), sv_n_photons

    # === Validation: need either input_state or n_photons ===
    if input_state is None and n_photons is None:
        raise ValueError("Either input_state or n_photons must be provided")

    # === Generate default state from n_photons ===
    if input_state is None and n_photons is not None:
        if computation_space is ComputationSpace.DUAL_RAIL:
            return pcvl.BasicState(tuple([1, 0] * n_photons)), n_photons
        elif amplitude_encoding:
            if circuit_m is None:
                raise ValueError(
                    "circuit_m must be provided to generate default state for amplitude encoding."
                )
            input_state = _generate_default_input_state(
                circuit_m,
                n_photons,
                computation_space,
            )
        else:
            if circuit_m is None:
                raise ValueError(
                    "circuit_m must be provided to generate default state when input_state is omitted."
                )
            return generate_state(circuit_m, n_photons, StatePattern.SPACED), n_photons

    # === Handle list[int] (legacy) ===
    if isinstance(input_state, list):
        return pcvl.BasicState(tuple(cast(list[int], input_state))), n_photons

    return (
        cast(StateVector | pcvl.BasicState | None, input_state),
        n_photons,
    )


def validate_and_resolve_circuit_source(
    builder: CircuitBuilder | None,
    circuit: pcvl.Circuit | None,
    experiment: pcvl.Experiment | None,
    trainable_parameters: list[str] | None,
    input_parameters: list[str] | None,
) -> CircuitSource:
    """Validate and normalize the circuit source selection.

    Parameters
    ----------
    builder : CircuitBuilder | None
        Builder source, if provided.
    circuit : pcvl.Circuit | None
        Circuit source, if provided.
    experiment : pcvl.Experiment | None
        Experiment source, if provided.
    trainable_parameters : list[str] | None
        User-provided trainable parameter prefixes.
    input_parameters : list[str] | None
        User-provided input parameter prefixes.

    Returns
    -------
    CircuitSource
        Resolved circuit-source configuration.

    Raises
    ------
    ValueError
        If zero or multiple circuit sources are provided, or if builder-derived
        prefixes are mixed with explicit parameter prefixes.
    """
    if sum(x is not None for x in (circuit, builder, experiment)) != 1:
        raise ValueError(
            "Provide exactly one of 'circuit', 'builder', or 'experiment'."
        )

    if builder is not None and (
        trainable_parameters is not None or input_parameters is not None
    ):
        raise ValueError(
            "When providing a builder, do not also specify 'trainable_parameters' "
            "or 'input_parameters'. Those prefixes are derived from the builder."
        )

    if builder is not None:
        return CircuitSource(
            source_type="builder",
            builder=builder,
            circuit=None,
            experiment=None,
            trainable_parameters=list(builder.trainable_parameter_prefixes),
            input_parameters=list(builder.input_parameter_prefixes),
            angle_encoding_specs=builder.angle_encoding_specs,
        )

    resolved_trainable = list(trainable_parameters) if trainable_parameters else []
    resolved_input = list(input_parameters) if input_parameters else []
    source_type: Literal["circuit", "experiment"] = (
        "circuit" if circuit is not None else "experiment"
    )

    return CircuitSource(
        source_type=source_type,
        builder=None,
        circuit=circuit,
        experiment=experiment,
        trainable_parameters=resolved_trainable,
        input_parameters=resolved_input,
        angle_encoding_specs={},
    )


def vet_experiment(experiment: pcvl.Experiment) -> dict[str, bool]:
    """Check experiment constraints.

    Parameters
    ----------
    experiment : pcvl.Experiment
        Experiment to validate.

    Returns
    -------
    dict[str, bool]
        Summary of experiment properties relevant to QuantumLayer support.

    Raises
    ------
    ValueError
        If the experiment uses unsupported features such as post-selection,
        heralding, feed-forward, time dependence, or minimum-photon filters.
    """
    has_post_select = not experiment.post_select_fn == pcvl.PostSelect()
    _post_select_fn = experiment.post_select_fn
    has_post_select = (
        _post_select_fn is not None and _post_select_fn != pcvl.PostSelect()
    )
    has_heralding = bool(experiment.heralds) or bool(experiment.in_heralds)
    has_feedforward = bool(getattr(experiment, "has_feedforward", False))
    has_td_attr = getattr(experiment, "has_td", None)
    has_td = has_td_attr() if callable(has_td_attr) else bool(has_td_attr)
    has_min_photons_filter = bool(getattr(experiment, "min_photons_filter", False))
    has_noise = bool(getattr(experiment, "noise", None))

    if has_post_select or has_heralding:
        raise ValueError(
            "The provided experiment must not have post-selection or heralding."
        )
    if has_feedforward:
        raise ValueError(
            "Feed-forward components are not supported inside a QuantumLayer experiment."
        )
    if has_td:
        raise ValueError(
            "The provided experiment must be unitary, and must not have post-selection or heralding."
        )
    if has_min_photons_filter:
        raise ValueError("The provided experiment must not have a min_photons_filter.")

    return {
        "is_unitary": not has_td,
        "has_noise": has_noise,
        "has_post_select": has_post_select,
        "has_heralding": has_heralding,
        "has_feedforward": has_feedforward,
        "has_min_photons_filter": has_min_photons_filter,
    }


def resolve_circuit(
    circuit_source: CircuitSource,
    pcvl_module,
    noise: pcvl.NoiseModel | None = None,
) -> ResolvedCircuit:
    """Resolve a builder, circuit, or experiment into a unified circuit form.

    Parameters
    ----------
    circuit_source : CircuitSource
        Resolved circuit source configuration.
    pcvl_module : Any
        Perceval module used to instantiate experiments when needed.
    noise: pcvl.NoiseModel | None
        The resolved NoiseModel of the layer. Defaults to None (no noise).

    Returns
    -------
    ResolvedCircuit
        Unified circuit and experiment wrapper.

    Raises
    ------
    RuntimeError
        If the provided ``circuit_source`` is internally inconsistent.
    """
    if circuit_source.source_type == "builder":
        if circuit_source.builder is None:
            raise RuntimeError("Builder must be provided for builder source type.")
        circuit = circuit_source.builder.to_pcvl_circuit(pcvl_module)
        experiment = pcvl_module.Experiment(circuit)
        if noise is not None:
            experiment.noise = noise
    elif circuit_source.source_type == "circuit":
        if circuit_source.circuit is None:
            raise RuntimeError("Circuit must be provided for circuit source type.")
        circuit = circuit_source.circuit
        experiment = pcvl_module.Experiment(circuit)
        if noise is not None:
            experiment.noise = noise
    elif circuit_source.source_type == "experiment":
        if circuit_source.experiment is None:
            raise RuntimeError(
                "Experiment must be provided for experiment source type."
            )
        experiment = circuit_source.experiment
        noise = getattr(experiment, "noise", None) if noise is None else noise
        circuit = experiment.unitary_circuit()
    else:
        raise RuntimeError("Resolved circuit could not be determined.")

    return ResolvedCircuit(
        circuit=circuit,
        experiment=experiment,
        noise=noise,
        has_custom_noise=noise is not None,
    )


def setup_noise_and_detectors(
    experiment: pcvl.Experiment,
    circuit: pcvl.Circuit,
    computation_space: ComputationSpace,
    measurement_strategy: MeasurementStrategyLike,
    backend: str | None = None,
    noise: pcvl.NoiseModel | None = None,
    return_object: bool = False,
) -> NoiseAndDetectorConfig:
    """Extract and validate photon-loss and detector configuration.

    Parameters
    ----------
    experiment : pcvl.Experiment
        Experiment from which noise and detectors are extracted.
    circuit : pcvl.Circuit
        Resolved circuit used to determine the number of modes.
    computation_space : ComputationSpace
        Logical computation space requested by the layer.
    measurement_strategy : :data:`~merlin.measurement.strategies.MeasurementStrategyLike`
        Measurement strategy used to validate detector and noise compatibility.
    backend : str | None
        Backend identifier used when validating supported noisy measurement
        configurations. If omitted, no backend-specific validation is applied.
        Default value is None.
    noise : pcvl.NoiseModel | None
        Noise model to validate and classify. If omitted, the simulation is
        treated as noise-free unless detector configuration requires validation.
        Default value is None.
    return_object : bool
        Whether the layer returns structured objects instead of tensors.
        Default value is False.

    Returns
    -------
    NoiseAndDetectorConfig
        Extracted and validated noise/detector configuration including photon
        survival probabilities, detector settings, and classified noise groups.

    Raises
    ------
    RuntimeError
        If amplitude readout is requested together with custom detectors.
    NotImplementedError
        If a non-SLOS backend is specified.
    ValueError
        If measurement strategy is incompatible with noisy simulation, including
        amplitude readout with active noise.
    """
    # Measurement resolution
    measurement_kind = _resolve_measurement_kind(measurement_strategy)
    amplitude_readout = measurement_kind == MeasurementKind.AMPLITUDES

    # Checking and resolving the detectors
    detectors, empty_detectors = resolve_detectors(experiment, circuit.m)
    detector_warnings: list[str] = []
    has_custom_detectors = not empty_detectors
    if has_custom_detectors and computation_space is not ComputationSpace.FOCK:
        detectors = [pcvl.Detector.pnr()] * circuit.m
        detector_warnings.append(
            f"Detectors are ignored in favor of ComputationSpace: {computation_space}"
        )
    if amplitude_readout and has_custom_detectors:
        raise RuntimeError(
            "measurement_strategy=MeasurementStrategy.amplitudes() does not support experiments with detectors. "
            "Compute amplitudes without detectors and apply a Partial DetectorTransform manually if needed."
        )

    # Validating and sorting the noise model
    noise_groups = None if noise is None else classify_noise(noise)
    if _circuit_has_phase_error(circuit):
        noise_groups = _with_component_phase_error(noise_groups)
    validate_noisy_measurement_strategy(
        backend,
        output=measurement_kind.name.lower(),
        noise=noise,
        noise_groups=noise_groups,
        empty_detectors=empty_detectors,
        return_object=return_object,
        measurement_kind=measurement_kind,
    )

    # Post measurement error
    photon_survival_probs, no_post_measurement_noise = resolve_photon_loss(
        noise_groups, circuit.m
    )
    has_post_measurement_noise = not no_post_measurement_noise

    # Creating the noise config object
    return NoiseAndDetectorConfig(
        photon_survival_probs=photon_survival_probs,
        has_custom_noise=has_post_measurement_noise,
        detectors=detectors,
        has_custom_detectors=has_custom_detectors,
        detector_warnings=detector_warnings,
        noise_groups=noise_groups,
    )


def apply_angle_encoding(
    x: torch.Tensor,
    spec: dict[str, Any],
) -> torch.Tensor:
    """Apply custom angle encoding using stored metadata.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor to encode. May be one- or two-dimensional.
    spec : dict[str, Any]
        Angle encoding metadata containing feature combinations and scales.

    Returns
    -------
    torch.Tensor
        Encoded tensor matching the requested combinations.

    Raises
    ------
    ValueError
        If ``x`` has unsupported rank or does not provide enough features for a
        requested combination.
    """
    combos: list[tuple[int, ...]] = spec.get("combinations", [])
    scale_map: dict[int, float] = spec.get("scales", {})

    if x.dim() == 1:
        x_batch = x.unsqueeze(0)
        squeeze = True
    elif x.dim() == 2:
        x_batch = x
        squeeze = False
    else:
        raise ValueError(
            f"Angle encoding expects 1D or 2D tensors, got shape {tuple(x.shape)}"
        )

    if not combos:
        encoded = x_batch
        return encoded.squeeze(0) if squeeze else encoded

    encoded_cols: list[torch.Tensor] = []
    feature_dim = x_batch.shape[-1]

    for combo in combos:
        indices = list(combo)
        if any(idx >= feature_dim for idx in indices):
            raise ValueError(
                f"Input feature dimension {feature_dim} insufficient for angle encoding combination {combo}"
            )

        selected = x_batch[:, indices]
        scales = [scale_map.get(idx, 1.0) for idx in indices]
        scale_tensor = x_batch.new_tensor(scales)
        value = (selected * scale_tensor).sum(dim=1, keepdim=True)
        encoded_cols.append(value)

    encoded = (
        torch.cat(encoded_cols, dim=1)
        if encoded_cols
        else x_batch.new_zeros((x_batch.shape[0], 0))
    )

    return encoded.squeeze(0) if squeeze else encoded


def compute_new_memristive_ps_angles(
    memristive_metadata: list[dict],
    memristive_state: list[torch.Tensor],
    output: torch.Tensor | PartialMeasurement | StateVector | ProbabilityDistribution,
) -> list[torch.Tensor]:
    """
    Computes the new memristive phase shifter angles per the batch's output.

    Parameters
    ----------
    memristive_metadata: list[dict]
        The memristive metadata of all memristive phase shifters
    memristive_state: list[torch.Tensor],
        The current state of the memristive phase shifters
    output: torch.Tensor | PartialMeasurement | merlin.core.state_vector.StateVector | ProbabilityDistribution,
        The output of the quantum layers

    Returns
    -------
    list[torch.Tensor]
        The new states of all memristive phase shifters
    """
    new_memristive_states = []
    for metadata, state in zip(memristive_metadata, memristive_state, strict=True):
        try:
            new_memristive_states.append(metadata["update_rule"](state, output))
        except Exception as exc:
            raise ValueError(
                f"""The update rule of the following memristor does not follow the correct build or raises an error. Here is the expected signature:

                    Expected: update_rule(state: torch.Tensor,output: torch.Tensor | StateVector | ProbabilityDistribution | PartialMeasurement)-> torch.Tensor

                    Memristive phase-shifter analyzed: {metadata}
                    """
            ) from exc
    return new_memristive_states


def prepare_input_encoding(
    x: torch.Tensor,
    prefix: str | None = None,
    angle_encoding_specs: dict[str, dict[str, Any]] | None = None,
) -> torch.Tensor:
    """Prepare input encoding for a given parameter prefix.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor to encode.
    prefix : str | None
        Prefix identifying the relevant angle encoding specification.
    angle_encoding_specs : dict[str, dict[str, Any]] | None
        Available angle encoding specifications.

    Returns
    -------
    torch.Tensor
        Encoded tensor if a matching specification is found, otherwise the
        input tensor unchanged.
    """
    if not angle_encoding_specs:
        return x

    spec = None
    if prefix is not None:
        spec = angle_encoding_specs.get(prefix)
    elif len(angle_encoding_specs) == 1:
        spec = next(iter(angle_encoding_specs.values()))

    if spec:
        return apply_angle_encoding(x, spec)

    return x


def split_inputs_by_prefix(
    prefixes: list[str],
    tensor: torch.Tensor,
    angle_encoding_specs: dict[str, dict[str, Any]],
    spec_mappings: dict[str, list[str]] | None = None,
) -> list[torch.Tensor] | None:
    """Split a logical input tensor into per-prefix chunks when possible.

    Parameters
    ----------
    prefixes : list[str]
        Ordered parameter prefixes to split against.
    tensor : torch.Tensor
        Input tensor containing all logical features.
    angle_encoding_specs : dict[str, dict[str, Any]]
        Angle encoding specifications keyed by prefix.
    spec_mappings : dict[str, list[str]] | None
        Optional spec mappings used as a fallback for feature counting.

    Returns
    -------
    list[torch.Tensor] | None
        Per-prefix tensor slices when the split is possible, otherwise ``None``.
    """
    counts: list[int] = []
    for prefix in prefixes:
        count = feature_count_for_prefix(prefix, angle_encoding_specs, spec_mappings)
        if count is None:
            return None
        counts.append(count)

    total_required = sum(counts)
    feature_dim = tensor.shape[-1] if tensor.dim() > 1 else tensor.shape[0]
    if total_required != feature_dim:
        return None

    slices: list[torch.Tensor] = []
    offset = 0
    for count in counts:
        end = offset + count
        slices.append(
            tensor[..., offset:end] if tensor.dim() > 1 else tensor[offset:end]
        )
        offset = end
    return slices


def feature_count_for_prefix(
    prefix: str,
    angle_encoding_specs: dict[str, dict[str, Any]],
    spec_mappings: dict[str, list[str]] | None = None,
) -> int | None:
    """Infer the number of raw features associated with an encoding prefix.

    Parameters
    ----------
    prefix : str
        Encoding prefix to inspect.
    angle_encoding_specs : dict[str, dict[str, Any]]
        Angle encoding specifications keyed by prefix.
    spec_mappings : dict[str, list[str]] | None
        Optional spec mappings used as a fallback.

    Returns
    -------
    int | None
        Number of raw features associated with ``prefix``, or ``None`` if it
        cannot be inferred.
    """
    spec = angle_encoding_specs.get(prefix)
    if spec:
        combos = spec.get("combinations", [])
        feature_indices = {idx for combo in combos for idx in combo}
        if feature_indices:
            return len(feature_indices)

    mapping = (spec_mappings or {}).get(prefix, [])
    if mapping:
        return len(mapping)

    return None


def _build_simple_circuit(
    input_size: int,
    n_modes: int | None = None,
    angle_encoding_scale: float = 1.0,
) -> CircuitBuilder:
    """Build the canonical *simple* circuit topology for a given mode/input configuration.

    The layout is:

    1. A fully trainable entangling layer (``"LI_simple"``).
    2. An angle-encoding layer spanning ``range(input_size)`` (``"input"``).
    3. A fully trainable entangling layer (``"RI_simple"``).

    Both :meth:`~merlin.algorithms.layer.QuantumLayer.simple` and
    :meth:`~merlin.algorithms.kernels.FeatureMap.simple` delegate to this
    function so the circuit topology is defined in a single place.

    Parameters
    ----------
    input_size : int
        Number of classical features encoded by the angle-encoding layer.
        Must satisfy ``input_size <= n_modes``.
    n_modes : int | None
        Number of photonic modes for the circuit. If omitted, defaults to
        ``input_size + 1``.
    angle_encoding_scale : float
        Global multiplicative scale applied to angle-encoding features.
        Default is ``1.0``.

    Returns
    -------
    CircuitBuilder
        Configured builder ready to be consumed by the caller.
    """
    if n_modes is None:
        n_modes = input_size + 1
    builder = CircuitBuilder(n_modes=n_modes)

    # Trainable entangling layer before encoding
    builder.add_entangling_layer(trainable=True, name="LI_simple")

    # Angle encoding
    builder.add_angle_encoding(
        modes=list(range(input_size)),
        name="input",
        subset_combinations=False,
        scale=angle_encoding_scale,
    )

    # Trainable entangling layer after encoding
    builder.add_entangling_layer(trainable=True, name="RI_simple")

    return builder


def normalize_output_key(
    key: Iterable[int] | torch.Tensor | Sequence[int],
) -> tuple[int, ...]:
    """Normalize an output key to ``tuple[int, ...]``.

    Parameters
    ----------
    key : Iterable[int] | torch.Tensor | Sequence[int]
        Output key in iterable or tensor form.

    Returns
    -------
    tuple[int, ...]
        Normalized tuple representation of the output key.
    """
    if isinstance(key, torch.Tensor):
        return tuple(int(v) for v in key.tolist())
    return tuple(int(v) for v in key)


def classify_noise(noise: pcvl.NoiseModel | None) -> NoiseGroups | None:
    """From a noise model, create a NoiseGroups object splitting the noise inputs.

    Parameters
    ----------
    noise : pcvl.NoiseModel | None
        The noise model to apply to the circuit

    Returns
    -------
    NoiseGroups | None
        The NoiseGroups object that contains the noise sources per category.
    """

    # The classifying pipeline can be seen as following. For a specific noise source.
    # 1. Is the noise value None. If it is, don't add anything to its category dictionary.
    # 2. Is the value the default value (noiseless case). If it is, don't add anything to its category dictionary
    # 3. Otherwise, add the noise value its corresponding category.
    #
    # If a category does not have any noises, it should be set to None.

    if noise is None:
        return None
    # Source noise
    source = {}
    # indistinguishability
    if noise.indistinguishability is None:
        pass
    elif not noise.indistinguishability == 1.0:  # indistinguishability's default value
        source["indistinguishability"] = noise.indistinguishability
    g2_active = noise.g2 is not None and noise.g2 != 0.0
    # g2_distinguishable
    if noise.g2_distinguishable is None:
        pass
    elif noise.g2_distinguishable and g2_active:
        source["g2_distinguishable"] = True
    # g2
    if noise.g2 is None:
        pass
    elif g2_active:  # g2's default value
        source["g2"] = noise.g2
    if source == {}:
        source = None

    # Circuit noise
    circuit = {}
    # phase_imprecision
    if noise.phase_imprecision is None:
        pass
    elif not noise.phase_imprecision == 0.0:
        circuit["phase_imprecision"] = (
            noise.phase_imprecision
        )  # phase_imprecision's default value
    # phase_error
    if noise.phase_error is None:
        pass
    elif not noise.phase_error == 0.0:  # phase_error's default value
        circuit["phase_error"] = noise.phase_error
    if circuit == {}:
        circuit = None

    # Post-measurement noise
    post_measurement = {}
    # transmittance
    if noise.transmittance is None:
        pass
    elif not noise.transmittance == 1.0:  # transmittance's default value
        post_measurement["transmittance"] = noise.transmittance
    # brightness
    if noise.brightness is None:
        pass
    elif not noise.brightness == 1.0:  # brightness's default value
        post_measurement["brightness"] = noise.brightness

    if post_measurement == {}:
        post_measurement = None

    # Building the NoiseGroups object
    if (source is None) and (circuit is None) and (post_measurement is None):
        return None
    return NoiseGroups(
        source=source, circuit=circuit, post_measurement=post_measurement
    )


def validate_noisy_measurement_strategy(
    strategy: str | None,
    output: str,
    noise: pcvl.NoiseModel | None = None,
    noise_groups: NoiseGroups | None = None,
    empty_detectors: bool = False,
    return_object: bool = False,
    measurement_kind: MeasurementKind | None = None,
) -> pcvl.NoiseModel | None:
    """Validate the noise model and QuantumLayer configurations so that they match.

    Validates that the noise model, measurement strategy, backend, and output
    configuration are compatible.

    Parameters
    ----------
    strategy : str | None
        The simulation backend used (e.g., "slos"). If provided and not "slos",
        raises NotImplementedError. Default is None.
    output : str
        The measurement strategy output type (e.g., "amplitudes", "probabilities",
        "mode_expectations"). Must be "probabilities" when noise is present.
    noise : pcvl.NoiseModel | None
        The noise model to validate. Default is None.
    noise_groups : NoiseGroups | None
        Classified active noise groups. Neutral noise models should pass None
        here so they are not treated as active noise. Default is None.
    empty_detectors : bool
        Whether the circuit has no custom detectors. Used to determine early returns
        when there is no noise. Default is False.
    return_object : bool
        Whether the layer returns structured objects instead of tensors.
        Cannot be True when active noise is present. Default is False.
    measurement_kind : MeasurementKind | None
        Resolved measurement kind. Used to reject partial measurement with
        active noise. Default is None.

    Returns
    -------
    pcvl.NoiseModel | None
        The input noise model.

    Raises
    ------
    NotImplementedError
        If return_object=True and active noise is present, or if the backend
        strategy is not "slos".
    ValueError
        If output measurement strategy is incompatible with noisy simulation
        (e.g., "amplitudes" or "mode_expectations" with noise).
    """
    active_noise = has_active_noise(noise_groups)
    if not active_noise and empty_detectors:
        return noise

    if active_noise and return_object:
        raise NotImplementedError(
            "The noise computation with the return_object feature set at True is not yet implemented."
        )
    if active_noise and (
        measurement_kind is MeasurementKind.PARTIAL or output == "partial"
    ):
        raise ValueError(
            "Partial measurement is not supported with active noise. Use full measurement or disable noise."
        )
    if active_noise and output != "probabilities":
        raise ValueError(
            "When doing a noisy simulation, the probabilities measurement strategy must be used."
        )
    if strategy is not None and strategy.upper() != "SLOS":
        raise NotImplementedError(
            f"Backend '{strategy}' is not supported. "
            "Only 'slos' is currently available."
        )
    return noise


def normalize_noise(
    layer_noise: pcvl.NoiseModel | None,
    experiment_noise: pcvl.NoiseModel | None,
) -> pcvl.NoiseModel:
    """Normalize and resolve noise models from multiple QuantumLayer sources.

    Resolves conflicting noise model sources and validates g2_distinguishable
    settings for coherence consistency. When both sources provide a noise model,
    they must be identical. Automatically corrects g2_distinguishable when
    indistinguishability is 1.0 (fully indistinguishable photons).

    Parameters
    ----------
    layer_noise : pcvl.NoiseModel | None
        The noise model declared via the noise argument of QuantumLayer.
    experiment_noise : pcvl.NoiseModel | None
        The noise model declared via experiment.noise in QuantumLayer.

    Returns
    -------
    pcvl.NoiseModel
        The resolved and validated noise model with corrected g2_distinguishable
        and g2 settings as needed.

    Raises
    ------
    ValueError
        If both layer_noise and experiment_noise are None, or if both
        are provided but not identical.

    Warns
    -----
    UserWarning
        When g2_distinguishable is automatically corrected from True to False due
        to fully indistinguishable photons (indistinguishability == 1.0).
    """

    output_nm = None
    if layer_noise is None:
        output_nm = experiment_noise
        # Both noise models are None
        if output_nm is None:
            return None
    else:
        if experiment_noise is None:
            output_nm = layer_noise
        else:
            if layer_noise == experiment_noise:
                output_nm = layer_noise

    if output_nm is None:
        raise ValueError(
            "Conflicting noise models: specify via noise= or experiment.noise, not both"
        )

    # Warning if trying to use g2 distinguishable photons if the photons are completly indistiguishable
    if (
        output_nm.indistinguishability == 1.0
        and (not output_nm.g2 == 0.0)
        and output_nm.g2_distinguishable
    ):
        warnings.warn(
            "When indistinguishability is 1.0 (fully indistinguishable photons) and g2 noise is present, "
            "g2_distinguishable must be False since indistinguishable g2 photons (indistinguishability=1.0) cannot be distinguished. "
            "Setting g2_distinguishable=False automatically.",
            UserWarning,
            stacklevel=2,
        )

    # If there is no application possible for g2_distinguishable (no g2 or indistiguishable photons), set it to False as it has no impact of the simulation
    if output_nm.indistinguishability == 1.0 or output_nm.g2 == 0.0:
        output_nm.g2_distinguishable = False

    return output_nm


def _normalize_sector_keys(
    keys: list[tuple[int, ...]] | list[list[tuple[int, ...]]],
) -> tuple[tuple[int, ...], ...]:
    if keys and isinstance(keys[0], list):
        nested_keys = cast(list[list[tuple[int, ...]]], keys)
        return tuple(tuple(k) for key_list in nested_keys for k in key_list)
    flat_keys = cast(list[tuple[int, ...]], keys)
    return tuple(tuple(k) for k in flat_keys)
