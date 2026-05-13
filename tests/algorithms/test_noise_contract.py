import re

import perceval as pcvl
import pytest

import merlin as ml
from merlin.algorithms.layer_utils import (
    classify_noise_model,
    normalize_noise_model,
)
import numpy as np
import torch


@pytest.fixture
def noise_groups() -> ml.QuantumLayer:
    return classify_noise_model(
        pcvl.NoiseModel(
            brightness=0.1,
            indistinguishability=0.2,
            g2=0.3,
            g2_distinguishable=False,
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

    assert not noise_groups.source["g2_distinguishable"]
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

    with pytest.raises(
        ValueError,
        match="When doing a noisy simulation, the probabilities measurement strategy must be used.",
    ):
        _ = ml.QuantumLayer(
            n_photons=2,
            input_size=5,
            builder=circ,
            noise_model=pcvl.NoiseModel(
                brightness=0.1,
                indistinguishability=0.2,
                g2=0.3,
                g2_distinguishable=True,
                transmittance=0.4,
                phase_imprecision=0.5,
                phase_error=0.6,
            ),
            measurement_strategy=ml.MeasurementStrategy.AMPLITUDES,
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
    circuit = circ.to_pcvl_circuit()

    exp = pcvl.Experiment(circuit)
    exp._add_detector(mode=0, detector=pcvl.Detector.threshold())

    with pytest.warns(
        UserWarning, match="Detectors are ignored in favor of ComputationSpace"
    ):
        _ = ml.QuantumLayer(
            n_photons=2,
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

    with pytest.raises(
        ValueError,
        match="When doing a noisy simulation, the probabilities measurement strategy must be used.",
    ):
        _ = ml.QuantumLayer(
            n_photons=2,
            input_size=5,
            builder=circ,
            noise_model=pcvl.NoiseModel(
                brightness=0.1,
                indistinguishability=0.2,
                g2=0.3,
                g2_distinguishable=True,
                transmittance=0.4,
                phase_imprecision=0.5,
                phase_error=0.6,
            ),
            measurement_strategy=ml.MeasurementStrategy.MODE_EXPECTATIONS,
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

    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "The following noises are not implemented yet for the QuantumLayer. Source noises: ['indistinguishability']."
        ),
    ):
        _ = ml.QuantumLayer(
            n_photons=2,
            input_size=5,
            builder=circ,
            noise_model=pcvl.NoiseModel(
                indistinguishability=0.2,
            ),
        )
    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "The following noises are not implemented yet for the QuantumLayer. Source noises: ['g2']."
        ),
    ):
        _ = ml.QuantumLayer(
            n_photons=2,
            input_size=5,
            builder=circ,
            noise_model=pcvl.NoiseModel(
                g2=0.3,
            ),
        )
    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "The following noises are not implemented yet for the QuantumLayer. Source noises: ['g2_distinguishable', 'indistinguishability']."
        ),
    ):
        _ = ml.QuantumLayer(
            n_photons=2,
            input_size=5,
            builder=circ,
            noise_model=pcvl.NoiseModel(
                indistinguishability=0.3,
                g2_distinguishable=False,
            ),
        )
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
            noise_model=pcvl.NoiseModel(
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
            noise_model=pcvl.NoiseModel(
                phase_error=0.6,
            ),
        )


# Entry-point normalization
def test_noise_via_experiment_raises_not_implemented():
    circ = ml.CircuitBuilder(n_modes=5)
    circ.add_entangling_layer()
    circ.add_angle_encoding()
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
        )


def test_noise_via_direct_parameter_raises_not_implemented():
    circ = ml.CircuitBuilder(n_modes=5)
    circ.add_entangling_layer()
    circ.add_angle_encoding()

    with pytest.raises(
        NotImplementedError,
    ):
        _ = ml.QuantumLayer(
            n_photons=2,
            input_size=5,
            builder=circ,
            noise_model=pcvl.NoiseModel(
                brightness=0.1,
                indistinguishability=0.2,
                g2=0.3,
                g2_distinguishable=True,
                transmittance=0.4,
                phase_imprecision=0.5,
                phase_error=0.6,
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
    output = normalize_noise_model(noise, noise)
    assert output == noise

    output = normalize_noise_model(noise, None)
    assert output == noise

    output = normalize_noise_model(None, noise)
    assert output == noise

    output = normalize_noise_model(None, None)
    assert output == None

    with pytest.raises(
        ValueError,
        match="Conflicting noise models: specify via noise_model= or experiment.noise, not both",
    ):
        output = normalize_noise_model(pcvl.NoiseModel(brightness=0.9), noise)


# Error message content
def test_impossible_noise():
    circ = ml.CircuitBuilder(n_modes=5)
    circ.add_entangling_layer()
    circ.add_angle_encoding()

    noise = pcvl.NoiseModel(
        indistinguishability=1.0,
        g2_distinguishable=False,
    )

    with pytest.raises(
        ValueError,
        match="g2_distinguishable noise can not be False",
    ):
        _ = ml.QuantumLayer(n_photons=2, input_size=5, builder=circ, noise_model=noise)


def _builder(n_modes: int = 4) -> ml.CircuitBuilder:
    """Create a small parameterized builder used only for construction tests."""
    builder = ml.CircuitBuilder(n_modes=n_modes)
    builder.add_entangling_layer()
    builder.add_angle_encoding()
    return builder


def _is_empty_group(group: dict | None) -> bool:
    return group is None or group == {}


def test_direct_noise_model_brightness_feeds_photon_loss_transform():
    layer = ml.QuantumLayer(
        n_photons=2,
        input_size=4,
        builder=_builder(),
        noise_model=pcvl.NoiseModel(brightness=0.3),
    )

    assert layer._photon_survival_probs == pytest.approx([0.3, 0.3, 0.3, 0.3])


def test_direct_noise_model_transmittance_feeds_photon_loss_transform():
    layer = ml.QuantumLayer(
        n_photons=2,
        input_size=4,
        builder=_builder(),
        noise_model=pcvl.NoiseModel(transmittance=0.4),
    )

    assert layer._photon_survival_probs == pytest.approx([0.4, 0.4, 0.4, 0.4])


def test_photon_survival_on_simple_circuits():
    # Empty_circuit
    layer = ml.QuantumLayer(
        input_size=0,
        circuit=pcvl.Circuit(m=4),
        noise_model=pcvl.NoiseModel(transmittance=0.1, brightness=0.2),
        n_photons=2,
    )
    assert np.allclose(layer._photon_survival_probs, [0.02, 0.02, 0.02, 0.02])

    # HOM
    circuit_hom = pcvl.Circuit(m=2).add([0, 1], pcvl.BS(convention=pcvl.BSConvention.H))
    layer = ml.QuantumLayer(
        circuit=circuit_hom,
        noise_model=pcvl.NoiseModel(transmittance=0.1, brightness=0.2),
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
        noise_model=pcvl.NoiseModel(brightness=0.3),
    )

    assert layer.computation_process.noise_groups is not None
    assert layer.computation_process.noise_groups.post_measurement == {
        "brightness": 0.3
    }


def test_empty_noise_model_has_no_active_groups():
    groups = classify_noise_model(pcvl.NoiseModel())

    assert groups is None or (
        _is_empty_group(groups.source)
        and _is_empty_group(groups.circuit)
        and _is_empty_group(groups.post_measurement)
    )


def test_brightness_only_classification_has_no_source_or_circuit_groups():
    groups = classify_noise_model(pcvl.NoiseModel(brightness=0.3))

    assert groups is not None
    assert _is_empty_group(groups.source)
    assert _is_empty_group(groups.circuit)
    assert groups.post_measurement == {"brightness": 0.3}


def test_not_implemented_error_lists_classified_groups():
    with pytest.raises(NotImplementedError) as exc_info:
        ml.QuantumLayer(
            n_photons=2,
            input_size=4,
            builder=_builder(),
            noise_model=pcvl.NoiseModel(g2=0.05, phase_error=0.1),
        )

    message = str(exc_info.value)
    assert "Source" in message
    assert "Circuit" in message
    assert "g2" in message
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


def test_return_object_with_noise_model_fails_fast():
    with pytest.raises(
        NotImplementedError,
        match="The noise computation with the return_object feature set at True is not yet implemented.",
    ):
        ml.QuantumLayer(
            n_photons=2,
            input_size=4,
            builder=_builder(),
            noise_model=pcvl.NoiseModel(brightness=0.3),
            return_object=True,
        )


def test_no_noise():
    layer = ml.QuantumLayer(
        n_photons=2,
        input_size=4,
        builder=_builder(),
    )
    assert layer.noise_model is None
    assert layer._noise_groups is None
    assert not layer.has_custom_noise_model
    assert not layer.has_custom_detectors

    # No errors should raise
    layer(torch.rand(4))
