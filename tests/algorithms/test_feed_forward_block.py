# MIT License
#
# Copyright (c)
#
# Tests for FeedForwardBlock API.

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import perceval as pcvl
import pytest
import torch
from exqalibur import FSArray
from perceval import BasicState, Circuit, Matrix, Unitary
from perceval.algorithm import Sampler
from perceval.components import PERM
from perceval.utils import NoiseModel

from merlin.algorithms.feed_forward import FeedForwardBlock
from merlin.algorithms.layer import QuantumLayer
from merlin.core.computation_space import ComputationSpace
from merlin.core.state_vector import StateVector
from merlin.measurement.strategies import MeasurementStrategy

_BASIS_CACHE: dict[tuple[int, int], list[tuple[int, ...]]] = {}


def _basis_states(n_modes: int, n_photons: int) -> list[tuple[int, ...]]:
    cache_key = (n_modes, n_photons)
    if cache_key not in _BASIS_CACHE:
        layer = QuantumLayer(
            input_size=0,
            circuit=pcvl.Circuit(n_modes),
            n_photons=n_photons,
            measurement_strategy=MeasurementStrategy.probs(
                computation_space=ComputationSpace.FOCK
            ),
        )
        _BASIS_CACHE[cache_key] = layer.computation_process.simulation_graph.mapped_keys
    return _BASIS_CACHE[cache_key]


def _as_keyed_tensors(block: FeedForwardBlock, tensor: torch.Tensor):
    keys = block.output_keys
    mapped: dict[tuple[int, ...], torch.Tensor] = {}
    for idx, key in enumerate(keys):
        entry = tensor[:, idx]
        if entry.shape[0] == 1:
            entry = entry.squeeze(0)
        size = block.output_state_sizes[key]
        if entry.ndim > 1 and entry.shape[-1] > size:
            entry = entry[..., :size]
        mapped[key] = entry
    return mapped


def _build_balanced_feedforward_experiment():
    """Construct a small experiment with one detector and a feed-forward provider."""
    exp = pcvl.Experiment()
    root = pcvl.Circuit(3)
    root.add(0, pcvl.BS())
    exp.add(0, root)

    exp.add(0, pcvl.Detector.pnr())

    reflective = pcvl.Circuit(2)
    reflective.add(0, PERM([1, 0]))

    transmissive = pcvl.Circuit(2)
    transmissive.add(0, pcvl.BS())

    provider = pcvl.FFCircuitProvider(1, 0, reflective)
    provider.add_configuration([1], transmissive)
    exp.add(0, provider)
    return exp


def _build_two_stage_experiment():
    """Approximate the multi-level experiment from ff_perceval.py."""
    exp = pcvl.Experiment()
    root = pcvl.Circuit(4)
    root.add(0, pcvl.BS())
    exp.add(0, root)

    exp.add(0, pcvl.Detector.pnr())
    v0 = pcvl.Circuit(3) // pcvl.BS()
    v1 = pcvl.Circuit(3) // pcvl.BS()
    v2 = pcvl.Circuit(3) // pcvl.BS()
    provider1 = pcvl.FFCircuitProvider(1, 0, v0)
    provider1.add_configuration([1], v1)
    provider1.add_configuration([2], v2)
    exp.add(0, provider1)

    exp.add(3, pcvl.Detector.threshold())
    provider2 = pcvl.FFCircuitProvider(1, -1, pcvl.Circuit(2))
    provider2.add_configuration([1], pcvl.Circuit(2) // pcvl.BS())
    exp.add(3, provider2)

    for mode in (1, 2):
        exp.add(mode, pcvl.Detector.pnr())

    return exp


def test_feedforward_block2_balanced_split():
    exp = _build_balanced_feedforward_experiment()
    block = FeedForwardBlock(
        exp,
        input_state=[2, 0, 0],
    )

    outputs = block()
    distribution_map = _as_keyed_tensors(block, outputs)

    total_prob = 0.0
    measurement_probs = defaultdict(float)
    for key, probability in distribution_map.items():
        if probability.ndim:
            prob_value = probability.squeeze().item()
        else:
            prob_value = probability.item()
        total_prob += prob_value
        measurement_probs[key[0]] += prob_value

    assert math.isclose(total_prob, 1.0, rel_tol=1e-5)
    assert len(measurement_probs) == 3


def test_feedforward_block2_parses_multiple_stages():
    exp = _build_two_stage_experiment()
    block = FeedForwardBlock(exp, input_state=[1, 1, 0, 0])

    assert len(block.stages) == 2
    assert block.stages[0].measured_modes == (0,)
    assert block.stages[1].measured_modes == (3,)
    desc = block.describe()
    assert "Stage 1" in desc and "Stage 2" in desc


def test_feedforward_block_uses_experiment_input_state():
    exp_with_state = _build_balanced_feedforward_experiment()
    exp_with_state.with_input(BasicState([2, 0, 0]))
    block_from_experiment = FeedForwardBlock(exp_with_state)

    exp_reference = _build_balanced_feedforward_experiment()
    block_reference = FeedForwardBlock(exp_reference, input_state=[2, 0, 0])

    assert torch.allclose(block_from_experiment(), block_reference())


def test_feedforward_block_warns_on_conflicting_input_state():
    exp = _build_balanced_feedforward_experiment()
    exp.with_input(BasicState([2, 0, 0]))
    with pytest.warns(UserWarning):
        FeedForwardBlock(exp, input_state=[1, 1, 0])


def test_feedforward_block_rejects_noisy_experiment():
    exp = _build_balanced_feedforward_experiment()
    exp.noise = NoiseModel(brightness=0.9)
    with pytest.raises(NotImplementedError):
        FeedForwardBlock(exp, input_state=[2, 0, 0])


def test_feedforward_block2_matches_perceval_two_stage():
    exp = _build_two_stage_experiment()
    input_state = [1, 1, 0, 0]
    block = FeedForwardBlock(exp, input_state=input_state)

    block_outputs = block()
    distribution_map = _as_keyed_tensors(block, block_outputs)
    block_probs = {
        key: float(value.item() if value.ndim == 0 else value.squeeze().item())
        for key, value in distribution_map.items()
    }

    exp.with_input(pcvl.BasicState(input_state))
    processor = pcvl.Processor("SLOS", exp)
    sampler = Sampler(processor)
    perceval_results = dict(sampler.probs()["results"])
    perceval_probs = {
        tuple(int(v) for v in state): float(prob)
        for state, prob in perceval_results.items()
    }
    assert set(block_probs) == set(perceval_probs)
    for key, value in block_probs.items():
        assert math.isclose(value, perceval_probs[key], rel_tol=1e-5, abs_tol=1e-5)


def test_feedforward_block2_amplitude_strategy_matches_probabilities():
    exp = _build_balanced_feedforward_experiment()
    input_state = [2, 0, 0]
    block_prob = FeedForwardBlock(exp, input_state=input_state)
    block_amp = FeedForwardBlock(
        exp,
        input_state=input_state,
        measurement_strategy=MeasurementStrategy.NONE,
    )

    prob_outputs = block_prob()
    prob_map = _as_keyed_tensors(block_prob, prob_outputs)
    amp_outputs = block_amp()
    assert isinstance(amp_outputs, list)

    full_probabilities = {
        key: float(value.item() if value.ndim == 0 else value.squeeze().item())
        for key, value in prob_map.items()
    }
    reconstructed: defaultdict[tuple[int, ...], float] = defaultdict(float)
    for measurement_key, branch_prob, remaining_n, amp_tensor in amp_outputs:
        assert torch.is_complex(amp_tensor)
        prob = branch_prob
        while prob.ndim < amp_tensor.ndim:
            prob = prob.unsqueeze(-1)
        distribution = amp_tensor.abs().pow(2) * prob
        unmeasured = [idx for idx, value in enumerate(measurement_key) if value is None]
        basis = _basis_states(len(unmeasured), remaining_n)
        states = basis if basis else ((),)
        flat_distribution = distribution.reshape(-1).tolist()
        for state, prob_value in zip(states, flat_distribution, strict=False):
            full_key = list(measurement_key)
            for mode_idx, value in zip(unmeasured, state, strict=False):
                full_key[mode_idx] = value
            reconstructed[tuple(full_key)] += prob_value

    assert set(full_probabilities.keys()) == set(reconstructed.keys())
    for key, value in full_probabilities.items():
        assert math.isclose(value, reconstructed[key], rel_tol=1e-6, abs_tol=1e-6)


def test_feedforward_block2_mode_expectations():
    exp = _build_balanced_feedforward_experiment()
    input_state = [2, 0, 0]
    block_prob = FeedForwardBlock(exp, input_state=input_state)
    block_expect = FeedForwardBlock(
        exp,
        input_state=input_state,
        measurement_strategy=MeasurementStrategy.mode_expectations(
            ComputationSpace.UNBUNCHED
        ),
    )

    prob_outputs = block_prob()
    expect_outputs = block_expect()
    prob_map = _as_keyed_tensors(block_prob, prob_outputs)
    prob_scalars = {
        key: float(value.item() if value.ndim == 0 else value.squeeze().item())
        for key, value in prob_map.items()
    }
    expectation = expect_outputs.squeeze(0)
    manual = torch.zeros_like(expectation)
    for state, probability in prob_scalars.items():
        state_tensor = torch.tensor(
            state, dtype=expectation.dtype, device=expectation.device
        )
        manual += probability * state_tensor
    assert torch.allclose(manual, expectation, atol=1e-6, rtol=1e-6)


def test_feedforward_block2_rejects_tensor_input_state():
    exp = _build_balanced_feedforward_experiment()
    basis = _basis_states(3, 2)
    amplitudes = torch.zeros(len(basis), dtype=torch.complex64)
    amplitudes[basis.index((2, 0, 0))] = 1.0

    with pytest.raises(ValueError) as exc_info:
        FeedForwardBlock(exp, input_state=amplitudes)

    message = str(exc_info.value)
    assert "torch.Tensor" in message
    assert "FeedForwardBlock input_state" in message
    assert "StateVector.from_tensor()" in message


def test_feedforward_block2_accepts_merlin_state_vector_input():
    exp = _build_balanced_feedforward_experiment()
    block_basic = FeedForwardBlock(exp, input_state=[2, 0, 0])
    basis = _basis_states(3, 2)
    amplitudes = torch.zeros(len(basis), dtype=torch.complex64)
    amplitudes[basis.index((2, 0, 0))] = 1.0
    state_vector = StateVector.from_tensor(amplitudes, n_modes=3, n_photons=2)
    block_state_vector = FeedForwardBlock(exp, input_state=state_vector)

    ref_outputs = _as_keyed_tensors(block_basic, block_basic())
    state_vector_outputs = _as_keyed_tensors(block_state_vector, block_state_vector())
    for key in block_basic.output_keys:
        assert torch.allclose(ref_outputs[key], state_vector_outputs[key], atol=1e-6)


def test_feedforward_block2_accepts_state_vector_input():
    exp = _build_balanced_feedforward_experiment()
    block_basic = FeedForwardBlock(exp, input_state=[2, 0, 0])
    state_vector = pcvl.StateVector()
    state_vector += pcvl.StateVector(pcvl.BasicState([2, 0, 0])) * 1.0
    block_sv = FeedForwardBlock(exp, input_state=state_vector)

    ref_outputs = _as_keyed_tensors(block_basic, block_basic())
    sv_outputs = _as_keyed_tensors(block_sv, block_sv())
    for key in block_basic.output_keys:
        assert torch.allclose(ref_outputs[key], sv_outputs[key], atol=1e-6)


def test_feedforward_block2_input_and_trainable_parameters_backward():
    exp = pcvl.Experiment()
    root = pcvl.Circuit(2)
    root.add(0, pcvl.PS(pcvl.P("phi")))
    root.add((0, 1), pcvl.BS(theta=pcvl.P("theta_1")))
    exp.add(0, root)
    exp.add(0, pcvl.Detector.pnr())

    conditional = pcvl.Circuit(1)
    conditional.add(0, pcvl.PS(pcvl.P("theta_2")))
    provider = pcvl.FFCircuitProvider(1, 0, conditional)
    exp.add(0, provider)

    block = FeedForwardBlock(
        exp,
        input_state=[1, 0],
        input_parameters=["phi"],
        trainable_parameters=["theta"],
    )

    x = torch.tensor([[0.1]], dtype=torch.float32, requires_grad=True)
    outputs = block(x)
    target_index = next(idx for idx, key in enumerate(block.output_keys) if key[0] == 1)
    loss = outputs[:, target_index].real.sum()
    loss.backward()

    assert x.grad is not None
    # assert torch.any(x.grad.abs() > 0)
    # assert any(
    #    parameter.grad is not None and torch.any(parameter.grad != 0)
    #    for parameter in block.parameters()
    # )


def test_feedforward_block2_forward_without_inputs_matches_explicit_tensor():
    exp = _build_balanced_feedforward_experiment()
    block = FeedForwardBlock(exp, input_state=[2, 0, 0])

    automatic = block()
    explicit = block(torch.zeros((1, 0)))
    assert torch.allclose(automatic, explicit)


def test_feedforward_block2_requires_classical_features_when_needed():
    exp = pcvl.Experiment()
    circuit = pcvl.Circuit(2)
    circuit.add(0, pcvl.PS(pcvl.P("phi")))
    exp.add(0, circuit)
    exp.add(0, pcvl.Detector.pnr())
    provider = pcvl.FFCircuitProvider(1, 0, pcvl.Circuit(1))
    exp.add(0, provider)

    block = FeedForwardBlock(exp, input_state=[1, 0], input_parameters=["phi"])

    with pytest.raises(ValueError, match="provide a feature tensor"):
        block()

    one_d = torch.tensor([0.2], dtype=torch.float32)
    block(one_d.unsqueeze(0))
    block(one_d)


def _fourier_unitary(dim: int) -> Unitary:
    omega = np.exp(2j * np.pi / dim)
    matrix = np.empty((dim, dim), dtype=np.complex128)
    scale = 1 / math.sqrt(dim)
    for row in range(dim):
        for col in range(dim):
            matrix[row, col] = omega ** (row * col) * scale
    return Unitary(Matrix(matrix))


def _build_feedforward_experiment(detector) -> tuple[pcvl.Experiment, list[int]]:
    m = 4
    input_state = [1, 1, 0, 0]

    exp = pcvl.Experiment()
    root = Circuit(m)
    root.add(0, _fourier_unitary(m))
    root.add((0, 1), pcvl.BS())
    exp.add(0, root)

    exp.add(0, detector)

    default_branch = Circuit(m - 1)
    default_branch.add(0, _fourier_unitary(m - 1))

    adaptive_branch = Circuit(m - 1)
    adaptive_branch.add(0, PERM([2, 1, 0]))
    adaptive_branch.add(0, _fourier_unitary(m - 1))

    provider = pcvl.FFCircuitProvider(1, 0, default_branch)
    provider.add_configuration([1], adaptive_branch)
    exp.add(0, provider)

    exp.with_input(BasicState(input_state))

    return exp


def _perceval_probabilities(exp: pcvl.Experiment) -> dict[tuple[int, ...], float]:
    processor = pcvl.Processor("SLOS", exp)
    sampler = Sampler(processor)
    results = sampler.probs()["results"]
    return {
        tuple(int(value) for value in state): float(prob)
        for state, prob in results.items()
    }


def _prune_probabilities(
    distribution: dict[tuple[int, ...], float], *, atol: float = 1e-12
) -> dict[tuple[int, ...], float]:
    """Remove numerically empty entries from a probability map."""
    return {key: value for key, value in distribution.items() if abs(value) > atol}


def _block_probabilities(
    block: FeedForwardBlock, outputs: torch.Tensor
) -> dict[tuple[int, ...], float]:
    if outputs.shape[0] != 1:
        raise AssertionError("Test expects a single batch item.")
    batch = outputs.squeeze(0)
    return {block.output_keys[idx]: float(batch[idx]) for idx in range(batch.shape[0])}


def test_feedforward_block_matches_perceval_distribution():
    experiment = _build_feedforward_experiment(pcvl.Detector.pnr())
    block = FeedForwardBlock(experiment)

    classical_inputs = torch.zeros((1, 0))
    outputs = block(classical_inputs)
    block_probs = _prune_probabilities(_block_probabilities(block, outputs))

    perceval_probs = _prune_probabilities(_perceval_probabilities(experiment))

    assert set(block_probs.keys()) == set(perceval_probs.keys())
    for key, prob in block_probs.items():
        assert math.isclose(prob, perceval_probs[key], rel_tol=1e-5, abs_tol=1e-5), (
            f"Mismatch for key {key}: Merlin={prob}, Perceval={perceval_probs[key]}"
        )


def test_feedforwardblock_params_only_in_branches():
    """Verify that trainable parameters are correctly assigned to stages and their layers."""
    m = 6
    k = 4
    n = 2

    possible_measurements = list(FSArray(k, n - 1))

    def gi_func(idx):
        return (
            Circuit(2)
            // pcvl.BS()
            // pcvl.PS(pcvl.P(f"x{idx}"))
            // pcvl.BS()
            // pcvl.PS(pcvl.P(f"A{2 * idx}"))
            // pcvl.BS()
            // pcvl.PS(pcvl.P(f"A{2 * idx + 1}"))
            // pcvl.BS()
        )

    def g(measurement):
        return possible_measurements.index(measurement) + 1

    def adaptive_mzi_stage1(measurement):
        g_val = g(measurement)
        return Circuit(2) // pcvl.BS() // pcvl.PS(pcvl.P(f"B{g_val - 1}")) // pcvl.BS()

    gi = pcvl.GenericInterferometer(m, gi_func)

    # Stage 1: after first k detectors
    feedforward_config_1 = pcvl.FFCircuitProvider(k, 0, Circuit(2))
    for measurement in possible_measurements:
        feedforward_config_1.add_configuration(
            measurement, adaptive_mzi_stage1(measurement)
        )

    experiment = pcvl.Experiment(m)
    experiment.add(0, gi)

    for i in range(k):
        experiment.add(i, pcvl.Detector.pnr())

    experiment.add(0, feedforward_config_1)

    ff_block = FeedForwardBlock(
        experiment,
        input_state=BasicState([1] * n + [0] * (m - n)),
        trainable_parameters=["A", "B"],
        input_parameters=["x"],
    )

    # Verify single stage was created
    assert len(ff_block._stage_runtimes) == 1
    stage_0 = ff_block._stage_runtimes[0]

    # Verify trainable parameters include both unitary ("A") and provider ("B") parameters
    assert stage_0.trainable_parameters is not None
    trainable_set = set(stage_0.trainable_parameters)
    assert "A" in trainable_set, "Stage should have 'A' parameters from unitary"
    assert "B" in trainable_set, (
        "Stage should have 'B' parameters from conditional branches"
    )

    # Verify pre_layer exists and has correct parameters
    assert stage_0.pre_layer is not None
    pre_layer_trainable = set(stage_0.pre_layer.trainable_parameters or [])
    assert "A" in pre_layer_trainable, "Pre-layer should have 'A' from unitary"
    # Note: "x" is included in the first-stage unitary, so it should be treated as a first-stage input parameter.
    assert "B" not in pre_layer_trainable, (
        "Pre-layer should not have 'B' (only in conditional branches)"
    )

    # Verify conditional circuits exist and contain "B" parameters
    assert stage_0.conditional_circuits is not None
    assert len(stage_0.conditional_circuits) > 0, (
        "Stage should have conditional circuits"
    )
    for circuit in stage_0.conditional_circuits.values():
        circuit_params = circuit.params
        # At least some conditional circuits should have B parameters
        if any(p.startswith("B") for p in circuit_params):
            break
    else:
        pytest.fail("No conditional circuit found with 'B' parameters")

    # Verify input parameters are handled by the FeedForwardBlock
    # Note: "x" was specified as an input_parameter and exists in provider circuits.
    # However, the first-stage pre_layer is built from the unitary circuit where "x" doesn't exist,
    # so "x" won't appear in pre_layer.input_parameters. FeedForwardBlock currently consumes
    # classical inputs in the first stage only; this assertion only verifies prefix mapping.

    # Verify the FeedForwardBlock's input parameter mapping
    assert ff_block._input_params_to_prefix_mapping is not None
    assert "x" in ff_block._prefix_to_params_mapping
    assert any(
        prefix == "x" for prefix in ff_block._input_params_to_prefix_mapping.values()
    ), "Input parameter 'x' should be recognized by FeedForwardBlock"

    # Verifying no forward bug
    ff_block.forward(torch.rand([1, 15]))


def test_feedforwardblock_input_at_send_layer_fails():
    m = 6
    k = 4
    n = 4

    possible_measurements = list(FSArray(k, n - 1))

    def gi_func(idx):
        return (
            Circuit(2)
            // pcvl.BS()
            // pcvl.PS(pcvl.P(f"A{2 * idx}"))
            // pcvl.BS()
            // pcvl.PS(pcvl.P(f"A{2 * idx + 1}"))
            // pcvl.BS()
        )

    def g(measurement):
        return possible_measurements.index(measurement) + 1

    def adaptive_mzi_stage1(measurement):
        g_val = g(measurement)
        return (
            Circuit(2)
            // pcvl.BS()
            // pcvl.PS(g_val * pcvl.P("x"))
            // pcvl.BS()
            // pcvl.PS(pcvl.P(f"B{g_val - 1}"))
            // pcvl.BS()
        )

    gi = pcvl.GenericInterferometer(m, gi_func)

    # Stage 1: after first k detectors
    feedforward_config_1 = pcvl.FFCircuitProvider(k, 0, Circuit(2))
    for measurement in possible_measurements:
        feedforward_config_1.add_configuration(
            measurement, adaptive_mzi_stage1(measurement)
        )

    experiment = pcvl.Experiment(m)
    experiment.add(0, gi)

    for i in range(k):
        experiment.add(i, pcvl.Detector.pnr())

    experiment.add(0, feedforward_config_1)
    with pytest.raises(
        ValueError, match="The first stage must use all of the input parameters"
    ):
        _ = FeedForwardBlock(
            experiment,
            input_state=BasicState([1] * n + [0] * (m - n)),
            trainable_parameters=["A", "B"],
            input_parameters=["x"],
        )


def test_feedforwardblock_params_multi_stage():
    """Verify parameter assignment works correctly across multiple stages."""
    # Build a 2-stage experiment with distinct parameters per stage
    # Following the structure of _build_two_stage_experiment()
    exp = pcvl.Experiment()

    # Stage 0: 4-mode root circuit, measure mode 0, provider with A parameters
    root = pcvl.Circuit(4)
    root.add(0, pcvl.BS())
    exp.add(0, root)

    exp.add(0, pcvl.Detector.pnr())
    v0_stage0 = Circuit(3) // pcvl.PS(pcvl.P("A0")) // pcvl.BS()
    v1_stage0 = Circuit(3) // pcvl.PS(pcvl.P("A1")) // pcvl.BS()
    provider0 = pcvl.FFCircuitProvider(1, 0, v0_stage0)
    provider0.add_configuration([1], v1_stage0)
    exp.add(0, provider0)

    # Stage 1: measure mode 3 (remaining after mode 0 measurement), provider with C parameters
    exp.add(3, pcvl.Detector.pnr())
    v0_stage1 = Circuit(2) // pcvl.PS(pcvl.P("C0")) // pcvl.BS(theta=pcvl.P("C1"))
    provider1 = pcvl.FFCircuitProvider(1, -1, v0_stage1)
    exp.add(3, provider1)

    # Add passive detectors on modes 1, 2 (these don't create feed-forward stages)
    for mode in (1, 2):
        exp.add(mode, pcvl.Detector.pnr())

    ff_block = FeedForwardBlock(
        exp,
        input_state=[1, 1, 0, 0],
        trainable_parameters=["A", "C"],
    )

    # Verify two stages were created
    assert len(ff_block._stage_runtimes) == 2, (
        f"Expected 2 stages, got {len(ff_block._stage_runtimes)}. Stages: {[s.measured_modes for s in ff_block._stage_runtimes]}"
    )
    stage_0 = ff_block._stage_runtimes[0]
    stage_1 = ff_block._stage_runtimes[1]

    # Stage 0: should have only "A" parameters
    assert stage_0.trainable_parameters is not None
    stage_0_params = set(stage_0.trainable_parameters)
    assert "A" in stage_0_params, "Stage 0 should have 'A' parameters"
    assert "C" not in stage_0_params, (
        "Stage 0 should not have 'C' parameters (belongs to stage 1)"
    )

    # Stage 1: should have only "C" parameters
    assert stage_1.trainable_parameters is not None
    stage_1_params = set(stage_1.trainable_parameters)
    assert "C" in stage_1_params, "Stage 1 should have 'C' parameters"
    assert "A" not in stage_1_params, (
        "Stage 1 should not have 'A' parameters (belongs to stage 0)"
    )

    # Verify each stage has its runtime objects (pre_layer may be None if root circuit has no parameters)
    assert stage_0.pre_layer is not None, "Stage 0 should have pre_layer"
    # Note: stage_1.pre_layer can be None if the root circuit for stage 1 has no parameters

    # Verify each stage has conditional circuits with their respective parameters
    assert stage_0.conditional_circuits is not None
    assert len(stage_0.conditional_circuits) > 0, (
        "Stage 0 should have conditional circuits"
    )
    assert any(
        any(p.startswith("A") for p in circuit.params)
        for circuit in stage_0.conditional_circuits.values()
    ), "Stage 0 conditional circuits should contain 'A' parameters"

    assert stage_1.conditional_circuits is not None
    assert len(stage_1.conditional_circuits) > 0, (
        "Stage 1 should have conditional circuits"
    )
    assert any(
        any(p.startswith("C") for p in circuit.params)
        for circuit in stage_1.conditional_circuits.values()
    ), "Stage 1 conditional circuits should contain 'C' parameters"
