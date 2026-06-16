# MIT License
#
# Copyright (c) 2026 Quandela
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

"""CUDA coverage for ``QuantumLayer`` noise models."""

from __future__ import annotations

from collections.abc import Callable

import perceval as pcvl
import pytest
import torch

import merlin as ml
from merlin.core.sectored_distribution import SectoredDistribution

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


NoiseFactory = Callable[[], pcvl.NoiseModel]


NOISE_MODEL_FACTORIES: list[tuple[str, NoiseFactory]] = [
    ("brightness", lambda: pcvl.NoiseModel(brightness=0.8)),
    ("transmittance", lambda: pcvl.NoiseModel(transmittance=0.7)),
    ("indistinguishability", lambda: pcvl.NoiseModel(indistinguishability=0.75)),
    ("g2", lambda: pcvl.NoiseModel(g2=0.05, g2_distinguishable=False)),
    (
        "g2_distinguishable",
        lambda: pcvl.NoiseModel(
            g2=0.05,
            g2_distinguishable=True,
            indistinguishability=0.8,
        ),
    ),
    ("phase_imprecision", lambda: pcvl.NoiseModel(phase_imprecision=0.5)),
    ("phase_error", lambda: pcvl.NoiseModel(phase_error=0.15)),
]


def _phase_sensitive_circuit() -> pcvl.Circuit:
    """Create a two-mode circuit whose probabilities depend on one phase."""
    circuit = pcvl.Circuit(2)
    circuit.add((0, 1), pcvl.BS.H())
    circuit.add(0, pcvl.PS(pcvl.P("phi")))
    circuit.add((0, 1), pcvl.BS.H())
    return circuit


def _cuda_noise_layer(noise: pcvl.NoiseModel) -> ml.QuantumLayer:
    """Create a small noisy layer that should keep all tensors on CUDA."""
    layer = ml.QuantumLayer(
        input_size=0,
        circuit=_phase_sensitive_circuit(),
        input_state=[1, 0],
        n_photons=1,
        trainable_parameters=["phi"],
        noise=noise,
        n_phase_error_samples=3,
        measurement_strategy=ml.MeasurementStrategy.probs(
            computation_space=ml.ComputationSpace.FOCK
        ),
        dtype=torch.float64,
        device=torch.device("cuda"),
    )
    with torch.no_grad():
        layer.thetas[0].fill_(0.37)
    return layer


def _assert_output_is_cuda_probability(
    output: torch.Tensor | SectoredDistribution,
) -> None:
    """Assert a probability output is normalized and resident on CUDA."""
    if isinstance(output, SectoredDistribution):
        assert output.sectors
        for sector in output.sectors:
            assert sector.tensor.device.type == "cuda"
            assert not sector.tensor.is_complex()
            assert torch.all(sector.tensor >= 0.0)
        assert torch.allclose(
            output.total_probability(),
            output.total_probability().new_tensor(1.0),
            atol=1e-6,
        )
        return

    assert output.device.type == "cuda"
    assert not output.is_complex()
    assert torch.all(output >= 0.0)
    assert torch.allclose(
        output.sum(dim=-1),
        output.new_ones(output.shape[:-1]),
        atol=1e-6,
    )


def _weighted_probability_loss(
    output: torch.Tensor | SectoredDistribution,
) -> torch.Tensor:
    """Return a phase-sensitive scalar loss for CUDA backward checks."""
    if isinstance(output, SectoredDistribution):
        loss: torch.Tensor | None = None
        for sector in output.sectors:
            weights = torch.linspace(
                0.25,
                1.0,
                sector.tensor.shape[-1],
                dtype=sector.tensor.dtype,
                device=sector.tensor.device,
            )
            sector_loss = (sector.tensor * weights).sum()
            loss = sector_loss if loss is None else loss + sector_loss
        if loss is None:
            raise RuntimeError("No sectors were available for the CUDA loss.")
        return loss

    weights = torch.linspace(
        0.25,
        1.0,
        output.shape[-1],
        dtype=output.dtype,
        device=output.device,
    )
    return (output * weights).sum()


def _assert_weighted_backward_keeps_gradients_on_cuda(
    layer: ml.QuantumLayer,
    output: torch.Tensor | SectoredDistribution,
) -> None:
    """Assert noisy CUDA outputs can backpropagate to trainable phase settings."""
    _weighted_probability_loss(output).backward()
    trainable_grads = [
        parameter.grad for parameter in layer.parameters() if parameter.requires_grad
    ]

    assert trainable_grads
    assert any(grad is not None for grad in trainable_grads)
    for grad in trainable_grads:
        if grad is None:
            continue
        assert grad.device.type == "cuda"
        assert torch.isfinite(grad).all()


@pytest.mark.parametrize(
    ("noise_name", "noise_factory"),
    NOISE_MODEL_FACTORIES,
    ids=[noise_name for noise_name, _noise_factory in NOISE_MODEL_FACTORIES],
)
def test_separate_noise_model_runs_forward_and_backward_on_cuda(
    noise_name: str,
    noise_factory: NoiseFactory,
) -> None:
    """Check each active noise field independently on CUDA."""
    del noise_name
    layer = _cuda_noise_layer(noise_factory())

    torch.manual_seed(1234)
    output = layer()

    _assert_output_is_cuda_probability(output)
    _assert_weighted_backward_keeps_gradients_on_cuda(layer, output)


def test_all_noise_model_fields_run_forward_and_backward_on_cuda() -> None:
    """Check the combined source, circuit, and post-measurement noise path on CUDA."""
    layer = _cuda_noise_layer(
        pcvl.NoiseModel(
            brightness=0.85,
            transmittance=0.9,
            indistinguishability=0.8,
            g2=0.05,
            g2_distinguishable=True,
            phase_imprecision=0.5,
            phase_error=0.15,
        )
    )

    torch.manual_seed(1234)
    output = layer()

    _assert_output_is_cuda_probability(output)
    _assert_weighted_backward_keeps_gradients_on_cuda(layer, output)
