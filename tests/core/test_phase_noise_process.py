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

"""Tests for phase-noise handling in ComputationProcess."""

from __future__ import annotations

import perceval as pcvl
import pytest
import torch

from merlin.algorithms.layer_utils import NoiseGroups
from merlin.core.computation_space import ComputationSpace
from merlin.core.process import ComputationProcess, ComputationProcessFactory
from merlin.core.sectored_distribution import SectoredDistribution
from merlin.utils.normalization import (
    normalize_probabilities,
    probabilities_from_amplitudes,
)


def _mzi_circuit() -> pcvl.Circuit:
    circuit = pcvl.Circuit(2)
    circuit.add((0, 1), pcvl.BS.H())
    circuit.add(0, pcvl.PS(pcvl.P("phi")))
    circuit.add((0, 1), pcvl.BS.H())
    return circuit


def _process(
    noise_groups: NoiseGroups | None = None,
    *,
    input_state: list[int] | torch.Tensor | None = None,
    n_phase_error_samples: int = 3,
) -> ComputationProcess:
    if input_state is None:
        input_state = [1, 0]
    return ComputationProcess(
        circuit=_mzi_circuit(),
        input_state=input_state,
        trainable_parameters=["phi"],
        input_parameters=[],
        n_photons=1,
        dtype=torch.float64,
        computation_space=ComputationSpace.FOCK,
        noise_groups=noise_groups,
        n_phase_error_samples=n_phase_error_samples,
    )


def _phase_parameter(value: float = 0.37) -> list[torch.Tensor]:
    return [torch.tensor([value], dtype=torch.float64, requires_grad=True)]


def _manual_phase_error_average(
    process: ComputationProcess,
    parameters: list[torch.Tensor],
    *,
    amplitude_encoding: bool = False,
) -> torch.Tensor | SectoredDistribution:
    accumulated: torch.Tensor | SectoredDistribution | None = None
    for _sample_index in range(process._n_phase_error_samples):
        unitary = process.converter.to_tensor(*parameters, apply_phase_error=True)
        probabilities = process._compute_probabilities_for_unitary(
            unitary, amplitude_encoding=amplitude_encoding
        )
        if accumulated is None:
            accumulated = probabilities
            continue
        if isinstance(accumulated, SectoredDistribution):
            assert isinstance(probabilities, SectoredDistribution)
            accumulated = process._add_sectored_distributions(
                accumulated, probabilities
            )
        else:
            assert isinstance(probabilities, torch.Tensor)
            accumulated = accumulated + probabilities

    assert accumulated is not None
    if isinstance(accumulated, SectoredDistribution):
        return process._divide_sectored_distribution(
            accumulated, process._n_phase_error_samples
        )
    return accumulated / process._n_phase_error_samples


def _manual_coherent_phase_error_average(
    process: ComputationProcess,
    parameters: list[torch.Tensor],
) -> torch.Tensor:
    accumulated: torch.Tensor | None = None
    for _sample_index in range(process._n_phase_error_samples):
        unitary = process.converter.to_tensor(*parameters, apply_phase_error=True)
        amplitudes = process._compute_superposition_amplitudes_for_unitary(unitary)
        probabilities = probabilities_from_amplitudes(amplitudes)
        probabilities = normalize_probabilities(
            probabilities, process.computation_space
        )
        accumulated = (
            probabilities if accumulated is None else accumulated + probabilities
        )

    assert accumulated is not None
    return accumulated / process._n_phase_error_samples


def _manual_incoherent_phase_error_average(
    process: ComputationProcess,
    parameters: list[torch.Tensor],
) -> torch.Tensor:
    accumulated: torch.Tensor | None = None
    for _sample_index in range(process._n_phase_error_samples):
        unitary = process.converter.to_tensor(*parameters, apply_phase_error=True)
        probabilities = process._compute_source_probabilities_for_unitary(
            unitary, amplitude_encoding=True
        )
        assert isinstance(probabilities, torch.Tensor)
        accumulated = (
            probabilities if accumulated is None else accumulated + probabilities
        )

    assert accumulated is not None
    return accumulated / process._n_phase_error_samples


def test_no_noise_compute_returns_amplitudes():
    process = _process()

    output = process.compute(_phase_parameter())

    assert isinstance(output, torch.Tensor)
    assert output.is_complex()


def test_phase_imprecision_only_compute_still_returns_amplitudes():
    process = _process(
        NoiseGroups(
            source=None,
            circuit={"phase_imprecision": 0.5},
            post_measurement=None,
        )
    )

    output = process.compute(_phase_parameter())

    assert isinstance(output, torch.Tensor)
    assert output.is_complex()
    assert process.converter._phase_imprecision == pytest.approx(0.5)


def test_phase_error_compute_returns_probabilities():
    process = _process(
        NoiseGroups(source=None, circuit={"phase_error": 0.2}, post_measurement=None),
        n_phase_error_samples=4,
    )

    torch.manual_seed(12)
    output = process.compute(_phase_parameter())

    assert isinstance(output, torch.Tensor)
    assert not output.is_complex()
    assert torch.allclose(output.sum(dim=-1), torch.tensor(1.0, dtype=output.dtype))


def test_phase_error_matches_manual_probability_average():
    process = _process(
        NoiseGroups(source=None, circuit={"phase_error": 0.2}, post_measurement=None),
        n_phase_error_samples=4,
    )
    parameters = _phase_parameter()

    torch.manual_seed(123)
    output = process.compute(parameters)
    torch.manual_seed(123)
    expected = _manual_phase_error_average(process, parameters)

    assert isinstance(output, torch.Tensor)
    assert isinstance(expected, torch.Tensor)
    assert torch.allclose(output, expected)


def test_phase_error_with_tensor_superposition_averages_probabilities():
    input_state = torch.tensor(
        [1.0, 1.0j],
        dtype=torch.complex128,
    )
    process = _process(
        NoiseGroups(source=None, circuit={"phase_error": 0.2}, post_measurement=None),
        input_state=input_state,
        n_phase_error_samples=4,
    )
    parameters = _phase_parameter()

    torch.manual_seed(123)
    output = process.compute(parameters, amplitude_encoding=True)
    torch.manual_seed(123)
    expected = _manual_coherent_phase_error_average(process, parameters)
    torch.manual_seed(123)
    incoherent_mixture = _manual_incoherent_phase_error_average(process, parameters)

    assert isinstance(output, torch.Tensor)
    assert isinstance(expected, torch.Tensor)
    assert not output.is_complex()
    assert torch.allclose(output, expected)
    assert not torch.allclose(expected, incoherent_mixture)
    assert torch.allclose(
        output.sum(dim=-1),
        torch.ones(output.shape[:-1], dtype=output.dtype, device=output.device),
    )


def test_phase_error_with_source_noise_averages_noisy_probabilities():
    process = _process(
        NoiseGroups(
            source={"indistinguishability": 0.8},
            circuit={"phase_error": 0.2},
            post_measurement=None,
        ),
        n_phase_error_samples=4,
    )
    parameters = _phase_parameter()

    torch.manual_seed(123)
    output = process.compute(parameters)
    torch.manual_seed(123)
    expected = _manual_phase_error_average(process, parameters)

    assert isinstance(output, torch.Tensor)
    assert isinstance(expected, torch.Tensor)
    assert torch.allclose(output, expected)


def test_phase_error_with_g2_averages_sectored_distributions():
    process = _process(
        NoiseGroups(
            source={"g2": 0.05},
            circuit={"phase_error": 0.2},
            post_measurement=None,
        ),
        n_phase_error_samples=3,
    )
    parameters = _phase_parameter()

    torch.manual_seed(123)
    output = process.compute(parameters)
    torch.manual_seed(123)
    expected = _manual_phase_error_average(process, parameters)

    assert isinstance(output, SectoredDistribution)
    assert isinstance(expected, SectoredDistribution)
    assert {sector.n_photons for sector in output.sectors} == {
        sector.n_photons for sector in expected.sectors
    }
    for sector in output.sectors:
        expected_sector = expected.get_sector(sector.n_photons)
        assert torch.allclose(sector.tensor, expected_sector.tensor)


def test_n_phase_error_samples_must_be_integer():
    with pytest.raises(TypeError, match="n_phase_error_samples must be an integer."):
        _process(n_phase_error_samples=1.5)  # type: ignore[arg-type]


def test_n_phase_error_samples_must_be_at_least_one():
    with pytest.raises(ValueError, match="n_phase_error_samples must be at least 1."):
        _process(n_phase_error_samples=0)


def test_compute_superposition_state_raises_with_phase_error():
    process = _process(
        NoiseGroups(source=None, circuit={"phase_error": 0.2}, post_measurement=None)
    )

    with pytest.raises(RuntimeError, match="phase_error"):
        process.compute_superposition_state(_phase_parameter())


def test_compute_ebs_simultaneously_raises_with_phase_error():
    input_state = torch.tensor([1.0, 0.0], dtype=torch.complex128)
    process = _process(
        NoiseGroups(source=None, circuit={"phase_error": 0.2}, post_measurement=None),
        input_state=input_state,
    )

    with pytest.raises(RuntimeError, match="phase_error"):
        process.compute_ebs_simultaneously(_phase_parameter())


def test_compute_with_keys_returns_phase_error_probabilities():
    process = _process(
        NoiseGroups(source=None, circuit={"phase_error": 0.2}, post_measurement=None),
        n_phase_error_samples=4,
    )

    torch.manual_seed(123)
    keys, output = process.compute_with_keys(_phase_parameter())

    assert keys == process.simulation_graph.mapped_keys
    assert isinstance(output, torch.Tensor)
    assert not output.is_complex()


def test_factory_forwards_n_phase_error_samples():
    process = ComputationProcessFactory.create(
        circuit=_mzi_circuit(),
        input_state=[1, 0],
        trainable_parameters=["phi"],
        input_parameters=[],
        n_photons=1,
        dtype=torch.float64,
        computation_space=ComputationSpace.FOCK,
        noise_groups=NoiseGroups(
            source=None,
            circuit={"phase_error": 0.2},
            post_measurement=None,
        ),
        n_phase_error_samples=7,
    )

    assert process._n_phase_error_samples == 7
