import re

import perceval as pcvl
import pytest

import merlin as ml
from merlin.core import StateVector
from merlin.algorithms.layer_utils import (
    classify_noise,
    normalize_noise,
)
import numpy as np
import torch
from copy import deepcopy


@pytest.fixture
def noise_groups() -> ml.QuantumLayer:
    return classify_noise(
        pcvl.NoiseModel(
            brightness=0.1,
            indistinguishability=0.2,
            g2=0.3,
            g2_distinguishable=True,
            transmittance=0.4,
            phase_imprecision=0.5,
            phase_error=0.6,
        ),
    )


# Classification tests
def test_brightness_classified_as_post_measurement(
    noise_groups,
):
    noise_groups = noise_groups
    assert noise_groups.post_measurement["brightness"] == 0.1
    assert "brightness" not in noise_groups.source.keys()
    assert "brightness" not in noise_groups.circuit.keys()


def test_g2_classified_as_source(noise_groups):
    noise_groups = noise_groups
    assert noise_groups.source["g2"] == 0.3
    assert "g2" not in noise_groups.post_measurement.keys()
    assert "g2" not in noise_groups.circuit.keys()


def test_indistinguishability_and_g2_distinguishable_classified_as_source(
    noise_groups,
):
    noise_groups = noise_groups
    assert noise_groups.source["indistinguishability"] == 0.2
    assert "indistinguishability" not in noise_groups.post_measurement.keys()
    assert "indistinguishability" not in noise_groups.circuit.keys()

    assert noise_groups.source["g2_distinguishable"]
    assert "g2_distinguishable" not in noise_groups.post_measurement.keys()
    assert "g2_distinguishable" not in noise_groups.circuit.keys()


def test_phase_imprecision_and_phase_error_classified_as_circuit(
    noise_groups,
):
    noise_groups = noise_groups
    assert noise_groups.circuit["phase_imprecision"] == 0.5
    assert "phase_imprecision" not in noise_groups.post_measurement.keys()
    assert "phase_imprecision" not in noise_groups.source.keys()

    assert noise_groups.circuit["phase_error"] == 0.6
    assert "phase_error" not in noise_groups.post_measurement.keys()
    assert "phase_error" not in noise_groups.source.keys()


def test_transmittance_classified_as_post_measurement(noise_groups):
    noise_groups = noise_groups
    assert noise_groups.post_measurement["transmittance"] == 0.4
    assert "transmittance" not in noise_groups.source.keys()
    assert "transmittance" not in noise_groups.circuit.keys()


# Validation tests
def test_noisy_layer_with_amplitudes_strategy_raises_value_error():
    circ = ml.CircuitBuilder(n_modes=5)
    circ.add_entangling_layer()
    circ.add_angle_encoding()
    circ.add_entangling_layer()

    with pytest.raises(
        ValueError,
        match="When doing a noisy simulation, the probabilities measurement strategy must be used.",
    ):
        _ = ml.QuantumLayer(
            n_photons=2,
            input_size=5,
            builder=circ,
            noise=pcvl.NoiseModel(
                brightness=0.1,
                indistinguishability=0.2,
                g2=0.3,
                g2_distinguishable=False,
                transmittance=0.4,
                phase_imprecision=0.5,
                phase_error=0.6,
            ),
            measurement_strategy=ml.MeasurementStrategy.amplitudes(
                computation_space=ml.ComputationSpace.FOCK
            ),
        )
    with pytest.raises(
        ValueError,
        match="When doing a noisy simulation, the probabilities measurement strategy must be used.",
    ):
        exp = pcvl.Experiment(pcvl.Circuit(5))
        exp.noise = pcvl.NoiseModel(brightness=0.5)
        _ = ml.QuantumLayer(
            input_size=5,
            experiment=exp,
            input_state=[1, 0, 0, 0, 0],
            measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
        )


def test_noisy_layer_with_detectors_with_other_computation_spaces():
    circ = ml.CircuitBuilder(n_modes=5)
    circ.add_entangling_layer()
    circ.add_angle_encoding()
    circ.add_entangling_layer()
    circuit = circ.to_pcvl_circuit()

    exp = pcvl.Experiment(circuit)
    exp._add_detector(mode=0, detector=pcvl.Detector.threshold())

    with pytest.warns(
        UserWarning, match="Detectors are ignored in favor of ComputationSpace"
    ):
        _ = ml.QuantumLayer(
            n_photons=1,
            input_size=5,
            experiment=exp,
            input_state=[1, 0, 0, 0, 0],
            trainable_parameters=list(circ.trainable_parameter_prefixes),
            input_parameters=list(circ.input_parameter_prefixes),
            measurement_strategy=ml.MeasurementStrategy.probs(
                computation_space=ml.ComputationSpace.UNBUNCHED
            ),
        )

    circ2 = ml.CircuitBuilder(n_modes=6)
    circ2.add_entangling_layer()
    circ2.add_angle_encoding()
    circ.add_entangling_layer()
    circuit2 = circ2.to_pcvl_circuit()

    exp2 = pcvl.Experiment(circuit2)
    exp2._add_detector(mode=0, detector=pcvl.Detector.threshold())

    with pytest.warns(
        UserWarning, match="Detectors are ignored in favor of ComputationSpace"
    ):
        _ = ml.QuantumLayer(
            n_photons=3,
            input_size=6,
            experiment=exp2,
            input_state=[1, 0, 1, 0, 1, 0],
            trainable_parameters=list(circ2.trainable_parameter_prefixes),
            input_parameters=list(circ2.input_parameter_prefixes),
            measurement_strategy=ml.MeasurementStrategy.probs(
                computation_space=ml.ComputationSpace.DUAL_RAIL
            ),
        )


def test_noisy_layer_with_mode_expectations_strategy_raises_value_error():
    circ = ml.CircuitBuilder(n_modes=5)
    circ.add_entangling_layer()
    circ.add_angle_encoding()
    circ.add_entangling_layer()

    with pytest.raises(
        ValueError,
        match="When doing a noisy simulation, the probabilities measurement strategy must be used.",
    ):
        _ = ml.QuantumLayer(
            n_photons=2,
            input_size=5,
            builder=circ,
            noise=pcvl.NoiseModel(
                brightness=0.1,
                indistinguishability=0.2,
                g2=0.3,
                g2_distinguishable=False,
                transmittance=0.4,
                phase_imprecision=0.5,
                phase_error=0.6,
            ),
            measurement_strategy=ml.MeasurementStrategy.mode_expectations(
                computation_space=ml.ComputationSpace.FOCK
            ),
        )
    # No error: amplitudes with no noise model
    _ = ml.QuantumLayer(
        n_photons=2,
        input_size=5,
        builder=circ,
        measurement_strategy=ml.MeasurementStrategy.amplitudes(),
    )


def test_noisy_layer_with_probs_strategy_raises_not_implemented():
    circ = ml.CircuitBuilder(n_modes=5)
    circ.add_entangling_layer()
    circ.add_angle_encoding()
    circ.add_entangling_layer()

    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "The following noises are not implemented yet for the QuantumLayer. Circuit noises: ['phase_imprecision']."
        ),
    ):
        _ = ml.QuantumLayer(
            n_photons=2,
            input_size=5,
            builder=circ,
            noise=pcvl.NoiseModel(
                phase_imprecision=0.5,
            ),
        )

    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "The following noises are not implemented yet for the QuantumLayer. Circuit noises: ['phase_error']."
        ),
    ):
        _ = ml.QuantumLayer(
            n_photons=2,
            input_size=5,
            builder=circ,
            noise=pcvl.NoiseModel(
                phase_error=0.6,
            ),
        )


# Entry-point normalization
def test_noise_via_experiment_raises_not_implemented():
    circ = ml.CircuitBuilder(n_modes=5)
    circ.add_entangling_layer()
    circ.add_angle_encoding()
    circ.add_entangling_layer()
    circuit = circ.to_pcvl_circuit()

    experiment = pcvl.Experiment(circuit)
    experiment.noise = pcvl.NoiseModel(
        brightness=0.1,
        indistinguishability=0.2,
        g2=0.3,
        g2_distinguishable=True,
        transmittance=0.4,
        phase_imprecision=0.5,
        phase_error=0.6,
    )

    with pytest.raises(
        NotImplementedError,
    ):
        _ = ml.QuantumLayer(
            n_photons=2,
            input_size=5,
            experiment=experiment,
            input_state=[1, 0, 0, 0, 0],
            measurement_strategy=ml.MeasurementStrategy.probs(
                computation_space=ml.ComputationSpace.FOCK
            ),
        )


def test_noise_via_direct_parameter_raises_not_implemented():
    circ = ml.CircuitBuilder(n_modes=5)
    circ.add_entangling_layer()
    circ.add_angle_encoding()
    circ.add_entangling_layer()

    with pytest.raises(
        NotImplementedError,
    ):
        _ = ml.QuantumLayer(
            n_photons=2,
            input_size=5,
            builder=circ,
            noise=pcvl.NoiseModel(
                brightness=0.1,
                indistinguishability=0.2,
                g2=0.3,
                g2_distinguishable=True,
                transmittance=0.4,
                phase_imprecision=0.5,
                phase_error=0.6,
            ),
            measurement_strategy=ml.MeasurementStrategy.probs(
                computation_space=ml.ComputationSpace.FOCK
            ),
        )


def test_normalise_noise():

    noise = pcvl.NoiseModel(
        brightness=0.1,
        indistinguishability=0.2,
        g2=0.3,
        g2_distinguishable=True,
        transmittance=0.4,
        phase_imprecision=0.5,
        phase_error=0.6,
    )
    output = normalize_noise(noise, noise)
    assert output == noise

    output = normalize_noise(noise, None)
    assert output == noise

    output = normalize_noise(None, noise)
    assert output == noise

    output = normalize_noise(None, None)
    assert output == None

    with pytest.raises(
        ValueError,
        match="Conflicting noise models: specify via noise= or experiment.noise, not both",
    ):
        output = normalize_noise(pcvl.NoiseModel(brightness=0.9), noise)


# Error message content
def test_impossible_noise():
    circ = ml.CircuitBuilder(n_modes=5)
    circ.add_entangling_layer()
    circ.add_angle_encoding()
    circ.add_entangling_layer()

    noise = pcvl.NoiseModel(indistinguishability=1.0, g2_distinguishable=True, g2=0.2)

    # When indistinguishability is 1.0 and g2_distinguishable is True with g2 noise,
    # a warning should be emitted and g2_distinguishable auto-corrected to False
    with pytest.warns(
        UserWarning,
        match=r"g2_distinguishable must be False since indistinguishable g2 photons \(indistinguishability=1\.0\) cannot be distinguished\.",
    ):
        _ = ml.QuantumLayer(
            n_photons=2,
            input_size=5,
            builder=circ,
            noise=noise,
            computation_space=ml.ComputationSpace.FOCK,
        )

    # Verify that g2_distinguishable was auto-corrected
    assert noise.g2_distinguishable is False


def _builder(n_modes: int = 4) -> ml.CircuitBuilder:
    """Create a small parameterized builder used only for construction tests."""
    builder = ml.CircuitBuilder(n_modes=n_modes)
    builder.add_entangling_layer()
    builder.add_angle_encoding()
    builder.add_entangling_layer()
    return builder


def _is_empty_group(group: dict | None) -> bool:
    return group is None or group == {}


def test_direct_noise_brightness_feeds_photon_loss_transform():
    layer = ml.QuantumLayer(
        n_photons=2,
        input_size=4,
        builder=_builder(),
        noise=pcvl.NoiseModel(brightness=0.3),
    )

    assert layer._photon_survival_probs == pytest.approx([0.3, 0.3, 0.3, 0.3])


def test_direct_noise_transmittance_feeds_photon_loss_transform():
    layer = ml.QuantumLayer(
        n_photons=2,
        input_size=4,
        builder=_builder(),
        noise=pcvl.NoiseModel(transmittance=0.4),
    )

    assert layer._photon_survival_probs == pytest.approx([0.4, 0.4, 0.4, 0.4])


def test_photon_survival_on_simple_circuits():
    # Empty_circuit
    layer = ml.QuantumLayer(
        input_size=0,
        circuit=pcvl.Circuit(m=4),
        noise=pcvl.NoiseModel(transmittance=0.1, brightness=0.2),
        n_photons=2,
    )
    assert np.allclose(layer._photon_survival_probs, [0.02, 0.02, 0.02, 0.02])

    # HOM
    circuit_hom = pcvl.Circuit(m=2).add([0, 1], pcvl.BS(convention=pcvl.BSConvention.H))
    layer = ml.QuantumLayer(
        circuit=circuit_hom,
        noise=pcvl.NoiseModel(transmittance=0.1, brightness=0.2),
        n_photons=2,
        measurement_strategy=ml.MeasurementStrategy.probs(
            computation_space=ml.ComputationSpace.FOCK
        ),
    )
    assert np.allclose(layer._photon_survival_probs, [0.02, 0.02])


def test_noise_groups_are_passed_to_computation_process():
    layer = ml.QuantumLayer(
        n_photons=2,
        input_size=4,
        builder=_builder(),
        noise=pcvl.NoiseModel(brightness=0.3),
    )

    assert layer.computation_process.noise_groups is not None
    assert layer.computation_process.noise_groups.post_measurement == {
        "brightness": 0.3
    }


def test_empty_noise_has_no_active_groups():
    # Normalizing for the non-trivial g2_distinguishable handling in Perceval
    groups = classify_noise(normalize_noise(pcvl.NoiseModel(), None))

    assert groups is None or (
        _is_empty_group(groups.source)
        and _is_empty_group(groups.circuit)
        and _is_empty_group(groups.post_measurement)
    )


def test_brightness_only_classification_has_no_source_or_circuit_groups():
    groups = classify_noise(normalize_noise(pcvl.NoiseModel(brightness=0.3), None))

    assert groups is not None
    assert _is_empty_group(groups.source)
    assert _is_empty_group(groups.circuit)
    assert groups.post_measurement == {"brightness": 0.3}


def test_not_implemented_error_lists_classified_groups():
    with pytest.warns(
        UserWarning,
        match=r"g2_distinguishable must be False since indistinguishable g2 photons \(indistinguishability=1\.0\) cannot be distinguished\.",
    ):
        with pytest.raises(NotImplementedError) as exc_info:
            ml.QuantumLayer(
                n_photons=2,
                input_size=4,
                builder=_builder(),
                noise=pcvl.NoiseModel(g2=0.05, phase_error=0.1),
                measurement_strategy=ml.MeasurementStrategy.probs(
                    computation_space=ml.ComputationSpace.FOCK
                ),
            )

    message = str(exc_info.value)
    assert "Circuit" in message
    assert "phase_error" in message


def test_experiment_noise_amplitudes_uses_value_error_contract():
    experiment = pcvl.Experiment(pcvl.Circuit(4))
    experiment.noise = pcvl.NoiseModel(brightness=0.5)

    with pytest.raises(ValueError, match="probabilities measurement strategy"):
        ml.QuantumLayer(
            input_size=4,
            experiment=experiment,
            input_state=[1, 0, 1, 0],
            measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
        )


def test_return_object_with_noise_fails_fast():
    with pytest.raises(
        NotImplementedError,
        match="The noise computation with the return_object feature set at True is not yet implemented.",
    ):
        ml.QuantumLayer(
            n_photons=2,
            input_size=4,
            builder=_builder(),
            noise=pcvl.NoiseModel(brightness=0.3),
            return_object=True,
        )


def test_indistinguishability_noise_no_longer_raises_not_implemented():
    layer = ml.QuantumLayer(
        n_photons=2,
        input_size=4,
        builder=_builder(),
        noise=pcvl.NoiseModel(indistinguishability=0.9),
        measurement_strategy=ml.MeasurementStrategy.probs(
            computation_space=ml.ComputationSpace.FOCK
        ),
    )

    out = layer(torch.rand(1, 4))
    assert out.shape[0] == 1


def test_brightness_noise_with_return_object_still_raises_not_implemented():
    with pytest.raises(
        NotImplementedError,
        match="The noise computation with the return_object feature set at True is not yet implemented.",
    ):
        ml.QuantumLayer(
            n_photons=2,
            input_size=4,
            builder=_builder(),
            noise=pcvl.NoiseModel(brightness=0.8),
            return_object=True,
        )


def test_indistinguishability_noise_backward_populates_thetas_grad():
    layer = ml.QuantumLayer(
        n_photons=2,
        input_size=4,
        builder=_builder(),
        noise=pcvl.NoiseModel(indistinguishability=0.9),
        measurement_strategy=ml.MeasurementStrategy.probs(
            computation_space=ml.ComputationSpace.FOCK
        ),
    )

    loss = ((layer(torch.rand(1, 4)) - torch.rand(1, 10)) ** 2).sum()
    loss.backward()

    grads = [p.grad for p in layer.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
    assert any(g.grad is not None for g in layer.thetas)
    assert len(layer.thetas) > 0


def test_indistinguishability_with_bunched_input_state_no_longer_raises():
    layer = ml.QuantumLayer(
        n_photons=2,
        input_size=4,
        builder=_builder(),
        input_state=[2, 0, 0, 0],
        noise=pcvl.NoiseModel(indistinguishability=0.9),
        measurement_strategy=ml.MeasurementStrategy.probs(
            computation_space=ml.ComputationSpace.FOCK
        ),
    )

    out = layer(torch.rand(1, 4))
    assert out.shape[0] == 1


def test_noisy_quantumlayer_batched_forward_matches_single_forwards():
    layer = ml.QuantumLayer(
        n_photons=3,
        input_size=5,
        builder=_builder(n_modes=5),
        noise=pcvl.NoiseModel(indistinguishability=0.4),
        measurement_strategy=ml.MeasurementStrategy.probs(
            computation_space=ml.ComputationSpace.FOCK
        ),
        dtype=torch.float64,
    )

    x_batch = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0], [1.57, 1.57, 1.57, 1.57, 1.57]],
        dtype=torch.float64,
    )

    batched = layer(x_batch)
    single_0 = layer(x_batch[0].unsqueeze(0))
    single_1 = layer(x_batch[1].unsqueeze(0))

    assert batched.shape[0] == 2
    assert torch.allclose(batched[0], single_0[0], atol=1e-6)
    assert torch.allclose(batched[1], single_1[0], atol=1e-6)
    assert torch.allclose(
        batched.sum(dim=1),
        torch.ones(2, dtype=batched.dtype, device=batched.device),
        atol=1e-6,
    )


def test_indistiguishable_layer_against_perceval_unitary():
    """Validate QuantumLayer with indistinguishability matches Perceval reference."""
    # Create a simple 2-mode circuit for testing
    circuit = pcvl.Circuit(2)
    circuit.add((0, 1), pcvl.BS.H())

    noise = pcvl.NoiseModel(indistinguishability=0.5)
    source = pcvl.Source.from_noise_model(noise)
    backend = pcvl.BackendFactory.get_backend("SLOS")
    sim = pcvl.Simulator(backend)
    sim.set_circuit(deepcopy(circuit))

    # Create QuantumLayer with equivalent circuit
    layer = ml.QuantumLayer(
        n_photons=2,
        circuit=circuit,
        noise=noise,
        measurement_strategy=ml.MeasurementStrategy.probs(
            computation_space=ml.ComputationSpace.FOCK
        ),
        amplitude_encoding=True,
        dtype=torch.float64,
    )

    # Test input states: enumerate Fock states for 2 photons in 2 modes.
    test_states_perceval = [[2, 0], [1, 1], [0, 2]]
    test_states_merlin = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    batched_input = torch.tensor(test_states_merlin, dtype=torch.float32)

    # Single batched call for all three input states.
    layer_output = layer(batched_input)
    all_output_states = ml.Combinadics(scheme="fock", n=2, m=2).enumerate_states()

    for batch_idx, input_state_tuple_perceval in enumerate(test_states_perceval):
        input_state = pcvl.BasicState(input_state_tuple_perceval)
        layer_probs = layer_output[batch_idx].detach().cpu().numpy()
        perceval_probs = sim.probs_svd((source, input_state))["results"]

        # Compare each probability between QuantumLayer and Perceval.
        for i, output_state in enumerate(all_output_states):
            state = pcvl.FockState(output_state)
            assert np.isclose(
                layer_probs[i],
                perceval_probs[state],
                atol=1e-4,
            ), (
                f"Probability mismatch for input {tuple(input_state_tuple_perceval)}, "
                f"output {tuple(output_state)}: "
                f"QuantumLayer={layer_probs[i]}, Perceval={perceval_probs[state]}"
            )

        assert np.isclose(layer_probs.sum(), 1.0, atol=1e-6)
        assert np.isclose(sum(perceval_probs.values()), 1.0, atol=1e-6)


def test_indistiguishable_layer_against_perceval_unitary_statevector_input():
    """Validate QuantumLayer with StateVector input against Perceval under indistinguishability noise."""
    circuit = pcvl.Circuit(2)
    circuit.add((0, 1), pcvl.BS.H())

    noise = pcvl.NoiseModel(indistinguishability=0.5)
    source = pcvl.Source.from_noise_model(noise)
    backend = pcvl.BackendFactory.get_backend("SLOS")
    sim = pcvl.Simulator(backend)
    sim.set_circuit(deepcopy(circuit))

    # Create layer without amplitude_encoding flag, without fixed input_state
    layer = ml.QuantumLayer(
        n_photons=2,
        circuit=deepcopy(circuit),
        noise=noise,
        measurement_strategy=ml.MeasurementStrategy.probs(
            computation_space=ml.ComputationSpace.FOCK
        ),
        dtype=torch.float64,
    )

    # Test with StateVector inputs
    test_states_perceval = [[2, 0], [1, 1], [0, 2]]
    output_states = ml.Combinadics(scheme="fock", n=2, m=2).enumerate_states()

    for input_state_tuple in test_states_perceval:
        # Create StateVector from basic state
        sv = StateVector.from_basic_state(
            input_state_tuple,
            device=layer.device,
            dtype=layer.complex_dtype,
        )

        # Forward pass with StateVector
        layer_output = layer(sv)
        layer_probs = layer_output[0].detach().cpu().numpy()

        # Compare against Perceval
        perceval_probs = sim.probs_svd((source, pcvl.BasicState(input_state_tuple)))[
            "results"
        ]

        for i, output_state in enumerate(output_states):
            state = pcvl.FockState(output_state)
            assert np.isclose(
                layer_probs[i],
                perceval_probs[state],
                atol=1e-4,
            ), (
                f"Probability mismatch for StateVector input {tuple(input_state_tuple)}, "
                f"output {tuple(output_state)}: "
                f"QuantumLayer={layer_probs[i]}, Perceval={perceval_probs[state]}"
            )

        assert np.isclose(layer_probs.sum(), 1.0, atol=1e-6)
        assert np.isclose(sum(perceval_probs.values()), 1.0, atol=1e-6)


def test_indistiguishable_layer_against_perceval_unitary_complex_tensor_input():
    """Validate QuantumLayer with complex tensor input against Perceval under indistinguishability noise."""
    circuit = pcvl.Circuit(2)
    circuit.add((0, 1), pcvl.BS.H())

    noise = pcvl.NoiseModel(indistinguishability=0.5)
    source = pcvl.Source.from_noise_model(noise)
    backend = pcvl.BackendFactory.get_backend("SLOS")
    sim = pcvl.Simulator(backend)
    sim.set_circuit(deepcopy(circuit))

    # Create layer without amplitude_encoding flag, without fixed input_state
    layer = ml.QuantumLayer(
        n_photons=2,
        circuit=deepcopy(circuit),
        noise=noise,
        measurement_strategy=ml.MeasurementStrategy.probs(
            computation_space=ml.ComputationSpace.FOCK
        ),
        dtype=torch.float64,
    )

    # Test with complex tensor inputs (one-hot encoded basis states)
    test_states_perceval = [[2, 0], [1, 1], [0, 2]]
    output_states = ml.Combinadics(scheme="fock", n=2, m=2).enumerate_states()

    # Get mapping of Fock states to their index in the layer's output space
    all_fock_states = ml.Combinadics(scheme="fock", n=2, m=2).enumerate_states()

    for input_state_tuple in test_states_perceval:
        # Find the index of this input state in the Fock basis
        input_idx = next(
            i
            for i, state in enumerate(all_fock_states)
            if tuple(state) == tuple(input_state_tuple)
        )

        # Create one-hot complex tensor
        num_fock_states = len(all_fock_states)
        complex_tensor = torch.zeros(num_fock_states, dtype=torch.complex128)
        complex_tensor[input_idx] = 1.0 + 0.0j

        # Forward pass with complex tensor
        layer_output = layer(complex_tensor)
        layer_probs = layer_output[0].detach().cpu().numpy()

        # Compare against Perceval
        perceval_probs = sim.probs_svd((source, pcvl.BasicState(input_state_tuple)))[
            "results"
        ]

        for i, output_state in enumerate(output_states):
            state = pcvl.FockState(output_state)
            assert np.isclose(
                layer_probs[i],
                perceval_probs[state],
                atol=1e-4,
            ), (
                f"Probability mismatch for complex tensor input {tuple(input_state_tuple)}, "
                f"output {tuple(output_state)}: "
                f"QuantumLayer={layer_probs[i]}, Perceval={perceval_probs[state]}"
            )

        assert np.isclose(layer_probs.sum(), 1.0, atol=1e-6)
        assert np.isclose(sum(perceval_probs.values()), 1.0, atol=1e-6)


def test_indistiguishable_layer_against_perceval_unitary_no_amplitude_encoding():
    """Validate QuantumLayer (non-amplitude path) against Perceval under indistinguishability noise."""
    circuit = pcvl.Circuit(2)
    circuit.add((0, 1), pcvl.BS.H())

    noise = pcvl.NoiseModel(indistinguishability=0.5)
    source = pcvl.Source.from_noise_model(noise)
    backend = pcvl.BackendFactory.get_backend("SLOS")
    sim = pcvl.Simulator(backend)
    sim.set_circuit(deepcopy(circuit))

    input_states = [[2, 0], [1, 1], [0, 2]]
    output_states = ml.Combinadics(scheme="fock", n=2, m=2).enumerate_states()

    for input_state_tuple in input_states:
        layer = ml.QuantumLayer(
            n_photons=2,
            circuit=deepcopy(circuit),
            input_state=input_state_tuple,
            noise=noise,
            measurement_strategy=ml.MeasurementStrategy.probs(
                computation_space=ml.ComputationSpace.FOCK
            ),
            dtype=torch.float64,
        )

        # Non-amplitude path: fixed constructor input_state, no amplitude_encoding input.
        layer_output = layer()
        layer_probs = layer_output[0].detach().cpu().numpy()

        perceval_probs = sim.probs_svd((source, pcvl.BasicState(input_state_tuple)))[
            "results"
        ]

        for i, output_state in enumerate(output_states):
            state = pcvl.FockState(output_state)
            assert np.isclose(
                layer_probs[i],
                perceval_probs[state],
                atol=1e-4,
            ), (
                f"Probability mismatch for input {tuple(input_state_tuple)}, "
                f"output {tuple(output_state)}: "
                f"QuantumLayer={layer_probs[i]}, Perceval={perceval_probs[state]}"
            )

        assert np.isclose(layer_probs.sum(), 1.0, atol=1e-6)
        assert np.isclose(sum(perceval_probs.values()), 1.0, atol=1e-6)


def test_no_noise():
    layer = ml.QuantumLayer(
        n_photons=2,
        input_size=4,
        builder=_builder(),
    )
    assert layer.noise is None
    assert layer._noise_groups is None
    assert not layer.has_custom_noise_model
    assert not layer.has_custom_detectors

    # No errors should raise
    layer(torch.rand(4))


def test_computation_space_changed():
    noise = pcvl.NoiseModel(indistinguishability=0.2)
    builder = ml.CircuitBuilder(n_modes=5)

    with pytest.raises(
        UserWarning,
        match="Noisy simulations with source noise currently use ComputationSpace.FOCK. Other computation spaces are not yet supported for noise models.",
    ):
        layer = ml.QuantumLayer(
            builder=builder,
            noise=noise,
            n_photons=2,
            computation_space=ml.ComputationSpace.UNBUNCHED,
        )
        assert layer.computation_space == ml.ComputationSpace.FOCK
        assert layer.output_size == 15
        assert layer().size(0) == 15

    with pytest.raises(
        UserWarning,
        match="Noisy simulations with source noise currently use ComputationSpace.FOCK. Other computation spaces are not yet supported for noise models.",
    ):
        layer = ml.QuantumLayer(
            builder=builder,
            noise=noise,
            n_photons=2,
            computation_space=ml.ComputationSpace.DUAL_RAIL,
        )
        assert layer.computation_space == ml.ComputationSpace.FOCK
        assert layer.output_size == 15
        assert layer().size(0) == 15
    with pytest.raises(
        UserWarning,
        match="Noisy simulations with source noise currently use ComputationSpace.FOCK. Other computation spaces are not yet supported for noise models.",
    ):
        layer = ml.QuantumLayer(
            builder=builder,
            noise=noise,
            n_photons=2,
            computation_space=ml.ComputationSpace.UNBUNCHED,
            amplitude_encoding=True,
        )
        assert layer.computation_space == ml.ComputationSpace.FOCK
        assert layer.output_size == 15
        assert layer.input_size == 15
        assert layer(torch.ones(15)).size(0) == 15


# Regression tests for g2 implementation (PML-286)


def test_g2_with_probs_no_longer_raises_not_implemented():
    """Regression: NoiseModel(g2=0.05) with probs output no longer raises NotImplementedError."""
    circ = ml.CircuitBuilder(n_modes=3)
    circ.add_entangling_layer()
    circ.add_angle_encoding(modes=[0, 1])
    circ.add_entangling_layer()

    # Should not raise NotImplementedError anymore
    layer = ml.QuantumLayer(
        n_photons=2,
        input_size=2,
        builder=circ,
        noise=pcvl.NoiseModel(indistinguishability=0.3, g2=0.05),
        measurement_strategy=ml.MeasurementStrategy.probs(
            computation_space=ml.ComputationSpace.FOCK
        ),
    )

    # Should be able to forward pass without error
    x = torch.randn(1, 2)
    output = layer(x)
    assert output is not None


def test_g2_indistinguishable_with_probs_no_longer_raises():
    """Regression: NoiseModel(g2=0.05, g2_distinguishable=False) with probs no longer raises."""
    circ = ml.CircuitBuilder(n_modes=3)
    circ.add_entangling_layer()
    circ.add_angle_encoding(modes=[0, 1])
    circ.add_entangling_layer()

    # Should not raise NotImplementedError anymore
    layer = ml.QuantumLayer(
        n_photons=2,
        input_size=2,
        builder=circ,
        noise=pcvl.NoiseModel(g2=0.05, g2_distinguishable=False),
        measurement_strategy=ml.MeasurementStrategy.probs(
            computation_space=ml.ComputationSpace.FOCK
        ),
    )

    # Should be able to forward pass without error
    x = torch.randn(1, 2)
    output = layer(x)
    assert output is not None

    # Should not raise NotImplementedError anymore
    layer = ml.QuantumLayer(
        n_photons=2,
        input_size=2,
        builder=circ,
        noise=pcvl.NoiseModel(
            indistinguishability=0.7, g2=0.05, g2_distinguishable=True
        ),
        measurement_strategy=ml.MeasurementStrategy.probs(
            computation_space=ml.ComputationSpace.FOCK
        ),
    )

    # Should be able to forward pass without error
    x = torch.randn(1, 2)
    output = layer(x)
    assert output is not None


def test_brightness_still_uses_post_measurement_approximation():
    """Regression: NoiseModel(brightness=0.8) still uses post-measurement approximation path."""
    circ = ml.CircuitBuilder(n_modes=3)
    circ.add_entangling_layer()
    circ.add_angle_encoding(modes=[0, 1])

    layer = ml.QuantumLayer(
        n_photons=2,
        input_size=2,
        builder=circ,
        noise=pcvl.NoiseModel(brightness=0.8),
        measurement_strategy=ml.MeasurementStrategy.probs(
            computation_space=ml.ComputationSpace.FOCK
        ),
    )

    # Verify brightness is classified as post_measurement
    assert "brightness" in layer._noise_groups.post_measurement
    assert layer._noise_groups.post_measurement["brightness"] == 0.8

    # Should forward without error
    x = torch.randn(1, 2)
    output = layer(x)
    assert output is not None


def test_g2_layer_forward_returns_tensor():
    """Regression: layer(x) returns SectoredDistribution when g2 > 0."""
    circ = ml.CircuitBuilder(n_modes=3)
    circ.add_entangling_layer()
    circ.add_angle_encoding(modes=[0, 1])
    circ.add_entangling_layer()

    layer = ml.QuantumLayer(
        n_photons=2,
        input_size=2,
        builder=circ,
        noise=pcvl.NoiseModel(g2=0.1, g2_distinguishable=False),
        measurement_strategy=ml.MeasurementStrategy.probs(
            computation_space=ml.ComputationSpace.FOCK
        ),
    )

    x = torch.randn(1, 2)
    output = layer(x)

    assert isinstance(output, torch.Tensor)
    assert max(output.shape) == 31


def test_g2_gradient_regression():
    """Regression: loss.backward() completes and layer.thetas.grad is not None for g2 > 0."""
    circ = ml.CircuitBuilder(n_modes=3)
    circ.add_entangling_layer(trainable=True)
    circ.add_angle_encoding(modes=[0, 1])
    circ.add_entangling_layer(trainable=True)

    layer = ml.QuantumLayer(
        n_photons=2,
        input_size=2,
        builder=circ,
        noise=pcvl.NoiseModel(g2=0.05, g2_distinguishable=False),
        measurement_strategy=ml.MeasurementStrategy.probs(
            computation_space=ml.ComputationSpace.FOCK
        ),
    )

    x = torch.randn(1, 2, requires_grad=True)
    output = layer(x)
    state = torch.zeros_like(output)
    state[0] = 1
    loss = ((output - state) ** 2).mean()
    loss.backward()

    # # Verify gradients are computed
    assert x.grad is not None
    assert torch.any(x.grad != 0)

    # Verify layer parameters have gradients
    for param in layer.parameters():
        if param.requires_grad:
            assert param.grad is not None

    for param in layer.thetas:
        assert param.grad is not None


def test_g2_output_keys_match_tensor_order_with_forward():
    """Verify that g2 noise (with and without photon loss) produces correct output_keys matching tensor order.

    Also validates output key coverage against Perceval simulations using fixed circuits.
    """

    # Test 1: G2 noise only (no photon loss)
    circ_g2_only = ml.CircuitBuilder(n_modes=2)
    circ_g2_only.add_entangling_layer()
    circ_g2_only.add_angle_encoding()

    layer_g2 = ml.QuantumLayer(
        n_photons=1,
        input_size=2,
        builder=circ_g2_only,
        noise=pcvl.NoiseModel(g2=0.1, g2_distinguishable=False),
        measurement_strategy=ml.MeasurementStrategy.probs(
            computation_space=ml.ComputationSpace.FOCK
        ),
    )

    # Forward pass with g2 only
    x_g2 = torch.zeros(1, 2)
    output_g2 = layer_g2(x_g2)
    keys_g2 = layer_g2.output_keys

    # Verify keys are flat (not nested) for g2 case
    assert isinstance(keys_g2, list)
    assert all(
        isinstance(k, tuple) for k in keys_g2
    ), "G2 output keys should be flat tuples"

    # Verify no duplicates in keys
    assert len(keys_g2) == len(
        set(keys_g2)
    ), f"Duplicate keys found in g2 case: {keys_g2}"

    # Verify output size matches keys
    assert output_g2.shape[-1] == len(
        keys_g2
    ), f"Output tensor size {output_g2.shape[-1]} doesn't match keys count {len(keys_g2)}"

    # Verify keys are valid Fock states for 1 photon in 2 modes
    expected_keys_g2 = {(0, 1), (1, 1), (2, 0), (0, 2), (1, 0)}
    assert (
        set(keys_g2) == expected_keys_g2
    ), f"Expected {expected_keys_g2}, got {set(keys_g2)}"

    # Verify probabilities sum to 1
    assert torch.isclose(
        output_g2.sum(), torch.tensor(1.0), atol=1e-5
    ), f"G2 output probabilities don't sum to 1: {output_g2.sum()}"

    # Compare with Perceval for g2 only case using fixed circuit (no variable params)
    fixed_circuit_g2 = pcvl.Circuit(2)
    fixed_circuit_g2.add((0, 1), pcvl.BS.H())

    noise_g2 = pcvl.NoiseModel(g2=0.1, g2_distinguishable=False)
    source_g2 = pcvl.Source.from_noise_model(noise_g2)
    backend = pcvl.BackendFactory.get_backend("SLOS")
    sim = pcvl.Simulator(backend)
    sim.set_circuit(deepcopy(fixed_circuit_g2))

    # Input state [1,0] for Perceval - compare key structure at least
    perceval_result_g2 = sim.probs_svd((source_g2, pcvl.BasicState([1, 0])))["results"]

    # Verify that all Perceval output states are represented in our keys
    perceval_states = {tuple(state) for state in perceval_result_g2.keys()}
    assert perceval_states.issubset(
        set(keys_g2)
    ), f"Some Perceval states not in Merlin keys. Perceval: {perceval_states}, Merlin: {set(keys_g2)}"

    # Test 2: G2 + Photon loss (brightness)
    circ_g2_pl = ml.CircuitBuilder(n_modes=2)
    circ_g2_pl.add_entangling_layer()
    circ_g2_pl.add_angle_encoding()

    layer_g2_pl = ml.QuantumLayer(
        n_photons=1,
        input_size=2,
        builder=circ_g2_pl,
        noise=pcvl.NoiseModel(g2=0.1, g2_distinguishable=False, brightness=0.2),
        measurement_strategy=ml.MeasurementStrategy.probs(
            computation_space=ml.ComputationSpace.FOCK
        ),
    )

    # Forward pass with g2 + photon loss
    x_g2_pl = torch.zeros(1, 2)
    output_g2_pl = layer_g2_pl(x_g2_pl)
    keys_g2_pl = layer_g2_pl.output_keys

    # Verify keys are flat (not nested) for g2 + photon loss case
    assert isinstance(keys_g2_pl, list)
    assert all(
        isinstance(k, tuple) for k in keys_g2_pl
    ), "G2+PL output keys should be flat tuples"

    # Verify no duplicates in keys
    assert len(keys_g2_pl) == len(
        set(keys_g2_pl)
    ), f"Duplicate keys found in g2+PL case: {keys_g2_pl}"

    # Verify output size matches keys
    assert output_g2_pl.shape[-1] == len(
        keys_g2_pl
    ), f"Output tensor size {output_g2_pl.shape[-1]} doesn't match keys count {len(keys_g2_pl)}"

    # With photon loss, we can have 0 or 1 photon states
    # Valid states: (0,0), (1,0), (0,1)
    expected_keys_g2_pl = {(0, 1), (0, 0), (1, 1), (2, 0), (0, 2), (1, 0)}
    assert (
        set(keys_g2_pl) == expected_keys_g2_pl
    ), f"Expected {expected_keys_g2_pl}, got {set(keys_g2_pl)}"

    # Verify probabilities sum to 1
    assert torch.isclose(
        output_g2_pl.sum(), torch.tensor(1.0), atol=1e-5
    ), f"G2+PL output probabilities don't sum to 1: {output_g2_pl.sum()}"

    # Compare with Perceval for g2 + photon loss case using fixed circuit
    sim_pl = pcvl.Simulator(backend)
    sim_pl.set_circuit(deepcopy(fixed_circuit_g2))

    noise_g2_pl = pcvl.NoiseModel(g2=0.1, g2_distinguishable=False, brightness=0.2)
    source_g2_pl = pcvl.Source.from_noise_model(noise_g2_pl)

    perceval_result_g2_pl = sim_pl.probs_svd((source_g2_pl, pcvl.BasicState([1, 0])))[
        "results"
    ]

    # Verify that all Perceval output states are represented in our keys
    perceval_states_pl = {tuple(state) for state in perceval_result_g2_pl.keys()}
    assert perceval_states_pl.issubset(
        set(keys_g2_pl)
    ), f"Some Perceval states not in Merlin keys. Perceval: {perceval_states_pl}, Merlin: {set(keys_g2_pl)}"
