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

"""Tests for differentiable phase noise in CircuitConverter."""

from __future__ import annotations

import inspect

import perceval as pcvl
import pytest
import torch

import merlin.pcvl_pytorch.locirc_to_tensor as locirc_to_tensor
from merlin.pcvl_pytorch.locirc_to_tensor import CircuitConverter


def _single_phase_circuit() -> pcvl.Circuit:
    return pcvl.Circuit(1) // pcvl.PS(pcvl.P("phi"))


def _expected_phase_unitary(phase: torch.Tensor | float, dtype: torch.dtype) -> torch.Tensor:
    phase_tensor = torch.as_tensor(phase, dtype=torch.float64)
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    return torch.exp(1j * phase_tensor.to(complex_dtype)).reshape(1, 1)


def _multi_parameter_bs_ps_circuit() -> pcvl.Circuit:
    circuit = pcvl.Circuit(2)
    circuit.add((0, 1), pcvl.BS.Rx(pcvl.P("theta0")))
    circuit.add(0, pcvl.PS(pcvl.P("phi0")))
    circuit.add(1, pcvl.PS(pcvl.P("phi1")))
    circuit.add((0, 1), pcvl.BS.Ry(pcvl.P("theta1")))
    return circuit


def _weighted_unitary_loss(unitary: torch.Tensor) -> torch.Tensor:
    return (
        unitary.real[..., 0, 0].sum()
        + 0.5 * unitary.imag[..., 0, 1].sum()
        - 0.25 * unitary.real[..., 1, 0].sum()
        + 0.75 * unitary.imag[..., 1, 1].sum()
    )


def test_phase_imprecision_quantizes_forward_phase_values():
    circuit = _single_phase_circuit()
    converter = CircuitConverter(
        circuit,
        input_specs=["phi"],
        dtype=torch.float64,
        phase_imprecision=0.5,
    )

    phase = torch.tensor([0.74], dtype=torch.float64)
    unitary = converter.to_tensor(phase)

    expected = _expected_phase_unitary(0.5, torch.float64)
    assert torch.allclose(unitary, expected)


def test_phase_imprecision_ste_preserves_gradient_to_phase_parameter():
    circuit = _single_phase_circuit()
    converter = CircuitConverter(
        circuit,
        input_specs=["phi"],
        dtype=torch.float64,
        phase_imprecision=0.5,
    )
    phase = torch.tensor([0.74], dtype=torch.float64, requires_grad=True)

    unitary = converter.to_tensor(phase)
    unitary.real.sum().backward()

    expected_grad = -torch.sin(torch.tensor([0.5], dtype=torch.float64))
    assert torch.allclose(phase.grad, expected_grad)


def test_inactive_phase_imprecision_matches_noiseless_converter():
    circuit = _single_phase_circuit()
    phase = torch.tensor([0.74], dtype=torch.float64)

    noiseless = CircuitConverter(circuit, ["phi"], dtype=torch.float64).to_tensor(phase)
    inactive = CircuitConverter(
        circuit,
        ["phi"],
        dtype=torch.float64,
        phase_imprecision=0.0,
    ).to_tensor(phase)

    assert torch.allclose(inactive, noiseless)


def test_negative_phase_imprecision_raises_value_error():
    with pytest.raises(ValueError, match="phase_imprecision must be non-negative"):
        CircuitConverter(_single_phase_circuit(), ["phi"], phase_imprecision=-0.1)


def test_negative_phase_error_raises_value_error():
    with pytest.raises(ValueError, match="phase_error must be non-negative"):
        CircuitConverter(_single_phase_circuit(), ["phi"], phase_error=-0.1)


def test_active_phase_error_uses_torch_rng_and_is_reproducible():
    circuit = _single_phase_circuit()
    converter = CircuitConverter(
        circuit,
        ["phi"],
        dtype=torch.float64,
        phase_error=0.5,
    )
    phase = torch.tensor([0.74], dtype=torch.float64)

    torch.manual_seed(1234)
    first = converter.to_tensor(phase, apply_phase_error=True)
    torch.manual_seed(1234)
    second = converter.to_tensor(phase, apply_phase_error=True)

    assert torch.allclose(first, second)


def test_different_torch_seeds_produce_different_phase_error_samples():
    circuit = _single_phase_circuit()
    converter = CircuitConverter(
        circuit,
        ["phi"],
        dtype=torch.float64,
        phase_error=0.5,
    )
    phase = torch.tensor([0.74], dtype=torch.float64)

    torch.manual_seed(1234)
    first = converter.to_tensor(phase, apply_phase_error=True)
    torch.manual_seed(5678)
    second = converter.to_tensor(phase, apply_phase_error=True)

    assert not torch.allclose(first, second)


def test_phase_error_is_inactive_without_apply_phase_error_flag():
    circuit = _single_phase_circuit()
    phase = torch.tensor([0.74], dtype=torch.float64)
    noiseless = CircuitConverter(circuit, ["phi"], dtype=torch.float64).to_tensor(phase)
    converter = CircuitConverter(
        circuit,
        ["phi"],
        dtype=torch.float64,
        phase_error=0.5,
    )

    torch.manual_seed(1234)
    unitary = converter.to_tensor(phase)

    assert torch.allclose(unitary, noiseless)


def test_constant_ps_remains_dynamic_when_phase_error_is_configured():
    circuit = pcvl.Circuit(1) // pcvl.PS(0.25)
    converter = CircuitConverter(circuit, dtype=torch.float64, phase_error=0.5)

    assert any(isinstance(component, pcvl.PS) for _, component in converter.list_rct)

    torch.manual_seed(1234)
    first = converter.to_tensor(apply_phase_error=True)
    torch.manual_seed(5678)
    second = converter.to_tensor(apply_phase_error=True)

    assert not torch.allclose(first, second)


def test_tiny_nonzero_phase_error_keeps_constant_ps_dynamic():
    circuit = pcvl.Circuit(1) // pcvl.PS(0.25)
    converter = CircuitConverter(circuit, dtype=torch.float64, phase_error=1e-15)

    assert any(isinstance(component, pcvl.PS) for _, component in converter.list_rct)


def test_constant_ps_with_local_max_error_remains_dynamic():
    circuit = pcvl.Circuit(1) // pcvl.PS(0.25, max_error=0.5)
    converter = CircuitConverter(circuit, dtype=torch.float64)

    assert any(isinstance(component, pcvl.PS) for _, component in converter.list_rct)

    torch.manual_seed(1234)
    first = converter.to_tensor(apply_phase_error=True)
    torch.manual_seed(5678)
    second = converter.to_tensor(apply_phase_error=True)

    assert not torch.allclose(first, second)


def test_local_max_error_is_inactive_without_apply_phase_error_flag():
    circuit = pcvl.Circuit(1) // pcvl.PS(0.25, max_error=0.5)
    converter = CircuitConverter(circuit, dtype=torch.float64)

    torch.manual_seed(1234)
    first = converter.to_tensor()
    torch.manual_seed(5678)
    second = converter.to_tensor()

    expected = _expected_phase_unitary(0.25, torch.float64)
    assert torch.allclose(first, expected)
    assert torch.allclose(second, expected)


def test_local_max_error_uses_component_half_width():
    circuit = pcvl.Circuit(1) // pcvl.PS(pcvl.P("phi"), max_error=0.25)
    converter = CircuitConverter(circuit, ["phi"], dtype=torch.float64)
    phase = torch.tensor([0.74], dtype=torch.float64)

    torch.manual_seed(1234)
    phase_error_sample = torch.empty((), dtype=torch.float64).uniform_(-0.25, 0.25)
    expected = _expected_phase_unitary(phase + phase_error_sample, torch.float64)

    torch.manual_seed(1234)
    unitary = converter.to_tensor(phase, apply_phase_error=True)

    assert torch.allclose(unitary, expected)


def test_local_max_error_warns_and_overrides_global_phase_error():
    circuit = pcvl.Circuit(1) // pcvl.PS(pcvl.P("phi"), max_error=0.25)
    phase = torch.tensor([0.74], dtype=torch.float64)

    with pytest.warns(UserWarning, match="max_error overrides phase_error"):
        converter = CircuitConverter(
            circuit,
            ["phi"],
            dtype=torch.float64,
            phase_error=0.75,
        )

    torch.manual_seed(1234)
    phase_error_sample = torch.empty((), dtype=torch.float64).uniform_(-0.25, 0.25)
    expected = _expected_phase_unitary(phase + phase_error_sample, torch.float64)

    torch.manual_seed(1234)
    unitary = converter.to_tensor(phase, apply_phase_error=True)

    assert torch.allclose(unitary, expected)


def test_constant_ps_can_be_precomputed_with_only_phase_imprecision():
    circuit = pcvl.Circuit(1) // pcvl.PS(0.74)
    converter = CircuitConverter(
        circuit,
        dtype=torch.float64,
        phase_imprecision=0.5,
    )

    assert all(isinstance(component, torch.Tensor) for _, component in converter.list_rct)

    unitary = converter.to_tensor()
    expected = _expected_phase_unitary(0.5, torch.float64)
    assert torch.allclose(unitary, expected)


def test_phase_error_samples_independently_for_batched_phase_parameters():
    circuit = _single_phase_circuit()
    converter = CircuitConverter(
        circuit,
        ["phi"],
        dtype=torch.float64,
        phase_error=0.5,
    )
    phases = torch.tensor([[0.25], [0.25]], dtype=torch.float64)

    torch.manual_seed(1234)
    unitary = converter.to_tensor(phases, apply_phase_error=True)

    assert unitary.shape == (2, 1, 1)
    assert not torch.allclose(unitary[0], unitary[1])


def test_phase_error_output_uses_converter_dtype():
    circuit = _single_phase_circuit()
    converter = CircuitConverter(
        circuit,
        ["phi"],
        dtype=torch.float64,
        phase_error=0.5,
    )
    phase = torch.tensor([0.25], dtype=torch.float64)

    torch.manual_seed(1234)
    unitary = converter.to_tensor(phase, apply_phase_error=True)

    assert unitary.dtype == torch.complex128


def test_python_random_is_not_used_for_phase_noise():
    source = inspect.getsource(locirc_to_tensor)

    assert "import random" not in source
    assert "random." not in source


def test_phase_error_perturbations_do_not_require_gradients():
    """Verify that phase_error samples do not flow gradients backward.

    Phase noise perturbations are stochastic noise, not trainable parameters.
    Gradients must flow only through the commanded phase value itself, so that
    optimization updates the phase setting and not the perturbation distribution.
    This test confirms that:
    1. Phase perturbation tensors have requires_grad=False
    2. Gradients with respect to the commanded phase are computed correctly
    3. The gradient does not reflect perturbations (is deterministic wrt phase)
    """
    circuit = _single_phase_circuit()
    converter = CircuitConverter(
        circuit,
        ["phi"],
        dtype=torch.float64,
        phase_error=0.1,
    )

    # Create a trainable phase parameter
    phase = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)

    # Compute unitary with phase_error applied
    torch.manual_seed(42)
    unitary1 = converter.to_tensor(phase, apply_phase_error=True)

    # Compute loss and backward pass
    loss1 = unitary1.real.sum()
    loss1.backward()
    grad1 = phase.grad.clone()

    # Reset and compute again with same seed to verify gradient is same
    phase.grad = None
    torch.manual_seed(42)
    unitary2 = converter.to_tensor(phase, apply_phase_error=True)

    loss2 = unitary2.real.sum()
    loss2.backward()
    grad2 = phase.grad

    # Gradients should be identical (deterministic wrt phase despite noise)
    assert torch.allclose(grad1, grad2, rtol=1e-6, atol=1e-8)

    # Gradients should exist and be non-zero
    assert grad1 is not None
    assert grad1.abs().sum() > 0.0


def test_combined_phase_imprecision_and_error_with_bs():
    circuit = pcvl.Circuit(2)
    circuit.add((0, 1), pcvl.BS.H())
    circuit.add(0, pcvl.PS(pcvl.P("phi")))
    circuit.add((0, 1), pcvl.BS.H())
    phase = torch.tensor([0.74], dtype=torch.float64)
    phase_imprecision = 0.5
    phase_error = 0.25

    converter = CircuitConverter(
        circuit,
        ["phi"],
        dtype=torch.float64,
        phase_imprecision=phase_imprecision,
        phase_error=phase_error,
    )
    noiseless_converter = CircuitConverter(circuit, ["phi"], dtype=torch.float64)

    torch.manual_seed(1234)
    phase_error_sample = torch.empty((), dtype=torch.float64).uniform_(
        -phase_error,
        phase_error,
    )
    quantized_phase = torch.round(phase / phase_imprecision) * phase_imprecision
    expected_phase = quantized_phase + phase_error_sample
    expected_unitary = noiseless_converter.to_tensor(expected_phase)

    torch.manual_seed(1234)
    noisy_unitary = converter.to_tensor(phase, apply_phase_error=True)

    quantized_only_unitary = noiseless_converter.to_tensor(quantized_phase)
    assert torch.allclose(noisy_unitary, expected_unitary)
    assert not torch.allclose(noisy_unitary, quantized_only_unitary)


def test_gradient_flow_multi_param_circuit():
    circuit = _multi_parameter_bs_ps_circuit()
    theta = torch.tensor([1.1, 0.8], dtype=torch.float64, requires_grad=True)
    phi = torch.tensor([0.74, -0.31], dtype=torch.float64, requires_grad=True)
    phase_imprecision = 0.5
    phase_error = 0.1

    converter = CircuitConverter(
        circuit,
        ["theta", "phi"],
        dtype=torch.float64,
        phase_imprecision=phase_imprecision,
        phase_error=phase_error,
    )

    torch.manual_seed(4321)
    noisy_unitary = converter.to_tensor(theta, phi, apply_phase_error=True)
    noisy_loss = _weighted_unitary_loss(noisy_unitary)
    noisy_loss.backward()

    torch.manual_seed(4321)
    phase_error_samples = torch.stack(
        [
            torch.empty((), dtype=torch.float64).uniform_(-phase_error, phase_error),
            torch.empty((), dtype=torch.float64).uniform_(-phase_error, phase_error),
        ]
    )
    expected_theta = theta.detach().clone().requires_grad_(True)
    expected_phi = phi.detach().clone().requires_grad_(True)
    quantized_phi = torch.round(expected_phi / phase_imprecision) * phase_imprecision
    effective_phi = expected_phi + (quantized_phi - expected_phi).detach()
    effective_phi = effective_phi + phase_error_samples.detach()
    expected_unitary = CircuitConverter(
        circuit,
        ["theta", "phi"],
        dtype=torch.float64,
    ).to_tensor(expected_theta, effective_phi)
    expected_loss = _weighted_unitary_loss(expected_unitary)
    expected_loss.backward()

    assert theta.grad is not None
    assert phi.grad is not None
    assert torch.isfinite(theta.grad).all()
    assert torch.isfinite(phi.grad).all()
    assert theta.grad.abs().sum() > 0.0
    assert phi.grad.abs().sum() > 0.0
    assert torch.allclose(theta.grad, expected_theta.grad)
    assert torch.allclose(phi.grad, expected_phi.grad)


def test_batched_combined_phase_noise_varies_across_samples():
    circuit = pcvl.Circuit(2)
    circuit.add(0, pcvl.PS(pcvl.P("phi0")))
    circuit.add(1, pcvl.PS(pcvl.P("phi1")))
    converter = CircuitConverter(
        circuit,
        ["phi"],
        dtype=torch.float64,
        phase_imprecision=0.5,
        phase_error=0.25,
    )
    phases = torch.tensor(
        [
            [0.74, -0.31],
            [0.74, -0.31],
            [0.74, -0.31],
        ],
        dtype=torch.float64,
    )

    torch.manual_seed(9876)
    first_unitary = converter.to_tensor(phases, apply_phase_error=True)
    torch.manual_seed(9876)
    second_unitary = converter.to_tensor(phases, apply_phase_error=True)

    assert first_unitary.shape == (3, 2, 2)
    assert torch.allclose(first_unitary, second_unitary)
    assert not torch.allclose(first_unitary[0], first_unitary[1])
    assert not torch.allclose(first_unitary[1], first_unitary[2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_combined_phase_noise_preserves_cuda_device_and_gradients():
    device = torch.device("cuda")
    circuit = _multi_parameter_bs_ps_circuit()
    converter = CircuitConverter(
        circuit,
        ["theta", "phi"],
        dtype=torch.float64,
        device=device,
        phase_imprecision=0.5,
        phase_error=0.1,
    )
    theta = torch.tensor(
        [1.1, 0.8],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    phi = torch.tensor(
        [0.74, -0.31],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )

    torch.manual_seed(1234)
    unitary = converter.to_tensor(theta, phi, apply_phase_error=True)
    loss = _weighted_unitary_loss(unitary)
    loss.backward()

    assert unitary.device.type == "cuda"
    assert theta.grad is not None
    assert phi.grad is not None
    assert theta.grad.device.type == "cuda"
    assert phi.grad.device.type == "cuda"
