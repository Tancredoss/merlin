import re

import perceval as pcvl
import pytest

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
    """Direct ``noise_model=`` must feed the existing post-measurement loss path.

    PML-284 says ``brightness`` is classified as post-measurement noise and must
    remain handled by the existing ``PhotonLossTransform`` plumbing. The direct
    API should therefore produce per-mode survival probabilities of ``0.3``.

    Current bug: for builder/circuit sources, the resolved direct noise model is
    not attached to the synthetic experiment used by ``resolve_photon_loss()``,
    so survival probabilities stay at ``1.0``.
    """
    layer = ml.QuantumLayer(
        n_photons=2,
        input_size=4,
        builder=_builder(),
        noise_model=pcvl.NoiseModel(brightness=0.3),
    )

    assert layer._photon_survival_probs == pytest.approx([0.3, 0.3, 0.3, 0.3])


def test_direct_noise_model_transmittance_feeds_photon_loss_transform():
    """Direct ``transmittance`` must also stay in the photon-loss post-process.

    This is the sibling case to ``brightness``. The ticket explicitly says
    ``brightness`` and ``transmittance`` are post-measurement values consumed by
    ``resolve_photon_loss()`` and the existing ``PhotonLossTransform``.

    Current bug: direct ``noise_model=NoiseModel(transmittance=...)`` is
    classified but not consumed by photon-loss resolution for builder/circuit
    sources.
    """
    layer = ml.QuantumLayer(
        n_photons=2,
        input_size=4,
        builder=_builder(),
        noise_model=pcvl.NoiseModel(transmittance=0.4),
    )

    assert layer._photon_survival_probs == pytest.approx([0.4, 0.4, 0.4, 0.4])


def test_noise_groups_are_passed_to_computation_process():
    """The PR plumbing must actually pass ``noise_groups`` into the process.

    PML-284 asks for ``NoiseGroups`` to propagate through initialization so that
    ``ComputationProcessFactory.create()`` can route to the correct noisy process
    subclass in follow-up work.

    Current bug: ``NoiseGroups`` is stored on the initialization context, and the
    factory accepts a ``noise_groups`` keyword, but ``QuantumLayer`` does not pass
    it at the call site. The resulting ``layer.computation_process.noise_groups``
    is ``None``.
    """
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
    """A default Perceval ``NoiseModel()`` should not classify active noise.

    The ticket says default/no-op values are excluded because they represent
    "no noise on this axis." Perceval exposes defaults as concrete identity
    values such as ``brightness=1``, ``g2=0`` and ``phase_error=0`` rather than
    as ``None``.

    Current bug: classification checks only for ``None``, so an empty model is
    classified as having source, circuit and post-measurement groups.
    """
    groups = classify_noise_model(pcvl.NoiseModel())

    assert groups is None or (
        _is_empty_group(groups.source)
        and _is_empty_group(groups.circuit)
        and _is_empty_group(groups.post_measurement)
    )


def test_brightness_only_classification_has_no_source_or_circuit_groups():
    """A brightness-only model should produce only post-measurement groups.

    This test makes the misclassification easy to see in pytest output. The
    expected group is only ``{"brightness": 0.3}``; source and circuit groups
    should be empty because all their values are Perceval identity defaults.
    """
    groups = classify_noise_model(pcvl.NoiseModel(brightness=0.3))

    assert groups is not None
    assert _is_empty_group(groups.source)
    assert _is_empty_group(groups.circuit)
    assert groups.post_measurement == {"brightness": 0.3}


def test_not_implemented_error_lists_classified_groups():
    """The construction-time ``NotImplementedError`` should summarize groups.

    PML-284 asks for a clear error, after classification and validation, that
    identifies the source/circuit/post-measurement groups found and states that
    differentiable noisy SLOS is not implemented yet.

    Current bug: the implementation raises field-specific messages such as
    ``The g2 error is not implemented yet`` and does not include the group
    summary reviewers and follow-up implementers need.
    """
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
    """Both noise entry points should use the same wrong-strategy error type.

    The ticket defines error ordering as: wrong ``measurement_strategy`` raises
    ``ValueError`` before noisy execution reaches the ``NotImplementedError``
    gate.

    Current bug: the direct ``noise_model=`` path raises ``ValueError``, but the
    ``experiment.noise`` path still hits the older amplitude/noise check and
    raises ``RuntimeError``.
    """
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
    """``return_object=True`` with noise is listed as out of scope for PML-284.

    The ticket explicitly says supporting ``return_object=True`` with a noise
    model is out of scope. Construction should therefore fail fast with a clear
    validation error instead of silently accepting a contract the ticket does not
    support.

    Current bug: a post-measurement-only direct noise model can be constructed
    with ``return_object=True``.
    """
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
