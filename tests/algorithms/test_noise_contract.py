import pytest
import perceval as pcvl
import merlin as ml
from merlin.algorithms.layer_utils import (
    classify_noise_model,
    normalize_noise_model,
)


@pytest.fixture
def noise_groups() -> ml.QuantumLayer:
    return classify_noise_model(
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

    assert noise_groups.source["g2_distinguishable"] == True
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
        RuntimeError,
        match="measurement_strategy=MeasurementStrategy.AMPLITUDES cannot be used when the experiment defines a NoiseModel.",
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
            measurement_strategy=ml.MeasurementStrategy.probs(
                computation_space=ml.ComputationSpace.UNBUNCHED
            ),
        )

    circ2 = ml.CircuitBuilder(n_modes=5)
    circ2.add_entangling_layer()
    circ2.add_angle_encoding()
    circuit2 = circ2.to_pcvl_circuit()

    exp2 = pcvl.Experiment(circuit2)
    exp2._add_detector(mode=0, detector=pcvl.Detector.threshold())

    with pytest.warns(
        UserWarning, match="Detectors are ignored in favor of ComputationSpace"
    ):
        _ = ml.QuantumLayer(
            n_photons=2,
            input_size=5,
            experiment=exp2,
            input_state=[1, 0, 0, 0, 0],
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
        match="The indistinguishability error is not implement yet",
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
        match="The g2 error is not implement yet",
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
        match="The g2_distinguishable error is not implement yet",
    ):
        _ = ml.QuantumLayer(
            n_photons=2,
            input_size=5,
            builder=circ,
            noise_model=pcvl.NoiseModel(
                g2=0.3,
                g2_distinguishable=True,
            ),
        )
    with pytest.raises(
        NotImplementedError,
        match="The phase_imprecision error is not implement yet",
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
        match="The phase_error error is not implement yet",
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
        g2_distinguishable=True,
    )

    with pytest.raises(
        ValueError,
        match="g2_distinguishable noise can not be True",
    ):
        _ = ml.QuantumLayer(n_photons=2, input_size=5, builder=circ, noise_model=noise)
