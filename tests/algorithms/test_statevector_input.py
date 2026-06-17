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
Tests for StateVector input type support in QuantumLayer.

These tests verify the implementation of PML-120-C:
- StateVector as canonical input_state type in constructor
- forward() dispatch by input type (tensor vs StateVector)
- Removal errors for legacy constructor amplitude inputs
"""

import perceval as pcvl
import pytest
import torch

import merlin as ML
from merlin.core.state_vector import StateVector


class TestConstructorInputTypes:
    """Test suite for constructor input_state type handling."""

    def test_input_state_accepts_statevector(self):
        """StateVector should be accepted as input_state (canonical type)."""
        circuit = pcvl.Circuit(4)
        sv = StateVector.from_basic_state([1, 0, 1, 0])

        # StateVector (FOCK space) is accepted regardless of computation_space
        # because computation_space only affects output post-selection
        layer = ML.QuantumLayer(
            input_size=0,
            circuit=circuit,
            input_state=sv,
            n_photons=2,
            measurement_strategy=ML.MeasurementStrategy.probs(),
        )

        assert layer.n_photons == 2

    def test_input_state_statevector_n_photons_mismatch_raises(self):
        """StateVector photon count must match explicit n_photons."""
        circuit = pcvl.Circuit(4)
        sv = StateVector.from_basic_state([1, 0, 1, 0])

        with pytest.raises(
            ValueError,
            match="Inconsistent number of photons between input_state and n_photons",
        ):
            ML.QuantumLayer(
                input_size=0,
                circuit=circuit,
                input_state=sv,
                n_photons=1,
                measurement_strategy=ML.MeasurementStrategy.probs(),
            )

    def test_input_state_accepts_perceval_statevector(self):
        """pcvl.StateVector should be accepted and converted."""
        circuit = pcvl.Circuit(4)
        pcvl_sv = pcvl.StateVector(pcvl.BasicState([1, 0, 1, 0]))

        layer = ML.QuantumLayer(
            input_size=0,
            circuit=circuit,
            input_state=pcvl_sv,
            measurement_strategy=ML.MeasurementStrategy.probs(),
        )

        assert layer.n_photons == 2

    def test_input_state_accepts_basicstate(self):
        """pcvl.BasicState should be accepted and converted."""
        circuit = pcvl.Circuit(3)
        basic_state = pcvl.BasicState("|1,0,1>")

        layer = ML.QuantumLayer(
            input_size=0,
            circuit=circuit,
            input_state=basic_state,
            measurement_strategy=ML.MeasurementStrategy.probs(),
        )

        assert layer.input_state == pcvl.BasicState([1, 0, 1])

    def test_input_state_accepts_list(self):
        """List should be accepted as input_state."""
        circuit = pcvl.Circuit(3)

        layer = ML.QuantumLayer(
            input_size=0,
            circuit=circuit,
            input_state=[1, 1, 0],
            measurement_strategy=ML.MeasurementStrategy.probs(),
        )

        assert layer.input_state == pcvl.BasicState([1, 1, 0])

    def test_input_state_accepts_tuple(self):
        """Tuple should be accepted as input_state."""
        circuit = pcvl.Circuit(3)

        layer = ML.QuantumLayer(
            input_size=0,
            circuit=circuit,
            input_state=(1, 0, 1),
            measurement_strategy=ML.MeasurementStrategy.probs(),
        )

        # Layer should work regardless of internal representation
        assert layer.n_photons == 2
        output = layer()
        assert torch.all(torch.isfinite(output))

    def test_input_state_tensor_raises_clear_error(self):
        """torch.Tensor as constructor input_state should be rejected."""
        circuit = pcvl.Circuit(2)
        tensor_state = torch.tensor([1.0, 0.0], dtype=torch.complex64)

        with pytest.raises(ValueError) as exc_info:
            ML.QuantumLayer(
                circuit=circuit,
                input_state=tensor_state,
                n_photons=1,
                measurement_strategy=ML.MeasurementStrategy.NONE,
            )

        message = str(exc_info.value)
        assert "torch.Tensor" in message
        assert "QuantumLayer input_state" in message
        assert "StateVector" in message
        assert "StateVector.from_tensor()" in message

    def test_set_input_state_tensor_raises_clear_error(self):
        """torch.Tensor should be rejected by set_input_state."""
        layer = ML.QuantumLayer(
            circuit=pcvl.Circuit(2),
            input_state=[1, 0],
            measurement_strategy=ML.MeasurementStrategy.NONE,
        )
        tensor_state = torch.tensor([1.0, 0.0], dtype=torch.complex64)

        with pytest.raises(ValueError) as exc_info:
            layer.set_input_state(tensor_state)

        message = str(exc_info.value)
        assert "torch.Tensor" in message
        assert "QuantumLayer input_state" in message
        assert "StateVector.from_tensor()" in message


class TestRemovedConstructorCompatibility:
    """Test suite for removed constructor amplitude compatibility."""

    def test_amplitude_encoding_true_raises_clear_error(self):
        """amplitude_encoding=True should be hard-rejected."""
        circuit = pcvl.Circuit(2)

        with pytest.raises(ValueError) as exc_info:
            ML.QuantumLayer(
                circuit=circuit,
                n_photons=1,
                amplitude_encoding=True,
                measurement_strategy=ML.MeasurementStrategy.NONE,
            )

        message = str(exc_info.value)
        assert "amplitude_encoding=True" in message
        assert "forward(StateVector)" in message
        assert "forward(complex_tensor)" in message
        assert "StateVector.from_tensor()" in message

    def test_amplitude_encoding_error_mentions_0_4(self):
        """Removal error should name the 0.4 removal target."""
        circuit = pcvl.Circuit(2)

        with pytest.raises(ValueError) as exc_info:
            ML.QuantumLayer(
                circuit=circuit,
                n_photons=1,
                amplitude_encoding=True,
                measurement_strategy=ML.MeasurementStrategy.NONE,
            )

        assert "0.4" in str(exc_info.value)


class TestForwardDispatch:
    """Test suite for forward() input type dispatch."""

    @pytest.fixture
    def angle_encoding_layer(self):
        """Create a layer configured for angle encoding."""
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        return ML.QuantumLayer(
            input_size=2,
            input_state=[1, 0, 1, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.probs(),
        )

    @pytest.fixture
    def amplitude_layer_fock(self):
        """Create a layer suitable for amplitude encoding with FOCK space.

        FOCK space is used to match StateVector's full Fock basis dimensions.
        For 4 modes, 2 photons: basis size = C(5,2) = 10 states.
        """
        circuit = pcvl.Circuit(4)
        return ML.QuantumLayer(
            circuit=circuit,
            n_photons=2,
            measurement_strategy=ML.MeasurementStrategy.amplitudes(
                computation_space=ML.ComputationSpace.FOCK
            ),
        )

    def test_forward_float_tensor_uses_angle_encoding(self, angle_encoding_layer):
        """Float tensor input should use angle encoding path."""
        layer = angle_encoding_layer
        x = torch.rand(3, 2)  # Float tensor

        output = layer(x)

        assert output.shape == (3, layer.output_size)
        assert torch.all(torch.isfinite(output))

    def test_forward_complex_tensor_uses_amplitude_encoding(self, amplitude_layer_fock):
        """Complex tensor input should use amplitude encoding path."""
        layer = amplitude_layer_fock
        n_states = layer.output_size  # Should be 10 for 4 modes, 2 photons in FOCK

        # Create normalized complex amplitudes
        amplitudes = torch.randn(2, n_states, dtype=torch.complex64)
        amplitudes = (
            amplitudes / amplitudes.abs().pow(2).sum(dim=-1, keepdim=True).sqrt()
        )

        output = layer(amplitudes)

        assert output.shape[0] == 2
        assert torch.all(torch.isfinite(output))

    def test_forward_statevector_uses_amplitude_encoding(self, amplitude_layer_fock):
        """StateVector input should use amplitude encoding path."""
        layer = amplitude_layer_fock

        # Create StateVector - for 4 modes, 2 photons, creates 10-dim sparse vector
        sv = StateVector.from_basic_state(
            [1, 0, 1, 0],
            device=layer.device,
            dtype=layer.complex_dtype,
        )

        output = layer(sv)

        assert output.shape[0] == 1  # Single state
        assert torch.all(torch.isfinite(output))

    def test_forward_mixed_inputs_raises_type_error(self, angle_encoding_layer):
        """Mixing tensor and StateVector inputs should raise TypeError."""
        layer = angle_encoding_layer
        tensor_input = torch.rand(2, 2)
        sv_input = StateVector.from_basic_state([1, 0, 1, 0])

        with pytest.raises(TypeError) as exc_info:
            layer(tensor_input, sv_input)

        assert "mix" in str(exc_info.value).lower() or "StateVector" in str(
            exc_info.value
        )

    def test_forward_unsupported_type_raises_type_error(self, angle_encoding_layer):
        """Unsupported input types should raise TypeError."""
        layer = angle_encoding_layer

        with pytest.raises(TypeError):
            layer("not a tensor")

    def test_forward_multiple_statevectors_raises_value_error(
        self, amplitude_layer_fock
    ):
        """Multiple StateVector inputs should raise ValueError."""
        layer = amplitude_layer_fock
        sv1 = StateVector.from_basic_state([1, 0, 1, 0])
        sv2 = StateVector.from_basic_state([0, 1, 0, 1])

        with pytest.raises(ValueError) as exc_info:
            layer(sv1, sv2)

        assert "one" in str(exc_info.value).lower() or "StateVector" in str(
            exc_info.value
        )


class TestNNSequentialCompatibility:
    """Test suite for nn.Sequential compatibility."""

    def test_nn_sequential_with_float_tensor_input(self):
        """nn.Sequential should work with float tensor inputs (angle encoding)."""
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        layer = ML.QuantumLayer(
            input_size=2,
            input_state=[1, 0, 1, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.probs(),
        )

        model = torch.nn.Sequential(
            layer,
            torch.nn.Linear(layer.output_size, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 3),
        )

        x = torch.rand(5, 2)
        output = model(x)

        assert output.shape == (5, 3)
        assert torch.all(torch.isfinite(output))

    def test_nn_sequential_gradient_flow(self):
        """Gradients should flow through nn.Sequential with QuantumLayer."""
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        layer = ML.QuantumLayer(
            input_size=2,
            input_state=[1, 0, 1, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.probs(),
        )

        model = torch.nn.Sequential(
            layer,
            torch.nn.Linear(layer.output_size, 3),
        )

        x = torch.rand(3, 2, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check input gradients
        assert x.grad is not None

        # Check layer parameters have gradients
        for param in layer.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestAngleEncodingBackwardCompatibility:
    """Test suite for existing angle encoding model compatibility."""

    def test_existing_angle_encoding_models_continue_to_run(self):
        """Existing tensor-based models using angle encoding should continue to work."""
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        layer = ML.QuantumLayer(
            input_size=2,
            input_state=[1, 0, 1, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.probs(),
        )

        # Standard usage pattern
        x = torch.rand(10, 2)
        output = layer(x)

        assert output.shape == (10, layer.output_size)
        # Probabilities should sum to 1
        assert torch.allclose(output.sum(dim=-1), torch.ones(10), atol=1e-5)

    def test_batched_angle_encoding_forward(self):
        """Batched forward pass with angle encoding should work as before."""
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1, 2], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        layer = ML.QuantumLayer(
            input_size=3,
            input_state=[1, 0, 1, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.probs(),
        )

        # Test various batch sizes
        for batch_size in [1, 5, 10, 32]:
            x = torch.rand(batch_size, 3)
            output = layer(x)
            assert output.shape == (batch_size, layer.output_size)

    def test_multiple_input_prefixes_still_work(self):
        """Multiple angle encoding prefixes should continue to work."""
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=False, name="pre_mix")
        builder.add_angle_encoding(modes=[0, 1], name="input_a")
        builder.add_angle_encoding(modes=[2, 3], name="input_b")
        builder.add_entangling_layer(trainable=False, name="post_mix")

        layer = ML.QuantumLayer(
            input_size=4,
            input_state=[1, 0, 1, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.probs(),
        )

        # Single combined input
        x_combined = torch.rand(3, 4)
        output = layer(x_combined)
        assert output.shape == (3, layer.output_size)

        # Separate inputs
        x_a = torch.rand(3, 2)
        x_b = torch.rand(3, 2)
        output_separate = layer(x_a, x_b)
        assert output_separate.shape == (3, layer.output_size)


class TestStateVectorForwardPath:
    """Test suite for the new StateVector forward path."""

    def test_statevector_forward_produces_valid_output(self):
        """StateVector forward should produce valid probability/amplitude output."""
        circuit = pcvl.Circuit(4)
        layer = ML.QuantumLayer(
            circuit=circuit,
            n_photons=2,
            measurement_strategy=ML.MeasurementStrategy.amplitudes(
                computation_space=ML.ComputationSpace.FOCK
            ),
        )

        sv = StateVector.from_basic_state(
            [1, 0, 1, 0],
            device=layer.device,
            dtype=layer.complex_dtype,
        )

        output = layer(sv)

        # Output should be normalized amplitudes
        assert torch.allclose(
            output.abs().pow(2).sum(),
            torch.tensor(1.0, dtype=layer.dtype),
            atol=1e-5,
        )

    def test_statevector_forward_with_superposition(self):
        """StateVector with superposition should work correctly."""
        circuit = pcvl.Circuit(4)
        layer = ML.QuantumLayer(
            circuit=circuit,
            n_photons=2,
            measurement_strategy=ML.MeasurementStrategy.amplitudes(
                computation_space=ML.ComputationSpace.FOCK
            ),
        )

        # Create a superposition state
        sv1 = StateVector.from_basic_state([1, 0, 1, 0], dtype=layer.complex_dtype)
        sv2 = StateVector.from_basic_state([0, 1, 0, 1], dtype=layer.complex_dtype)
        superposition = sv1 + sv2
        superposition.normalize()

        output = layer(superposition)

        assert output.shape[0] == 1
        assert torch.all(torch.isfinite(output))

    def test_statevector_device_dtype_handling(self):
        """StateVector should be moved to correct device/dtype automatically."""
        circuit = pcvl.Circuit(4)
        layer = ML.QuantumLayer(
            circuit=circuit,
            n_photons=2,
            measurement_strategy=ML.MeasurementStrategy.amplitudes(
                computation_space=ML.ComputationSpace.FOCK
            ),
            dtype=torch.float64,
        )

        # Create StateVector with different dtype
        sv = StateVector.from_basic_state(
            [1, 0, 1, 0],
            dtype=torch.complex64,
        )

        output = layer(sv)

        assert torch.all(torch.isfinite(output))


class TestComplexTensorForwardPath:
    """Test suite for complex tensor forward path (amplitude encoding)."""

    def test_complex_tensor_forward_uses_amplitude_path(self):
        """Complex tensor should be routed to amplitude encoding."""
        circuit = pcvl.Circuit(4)
        layer = ML.QuantumLayer(
            circuit=circuit,
            n_photons=2,
            measurement_strategy=ML.MeasurementStrategy.amplitudes(
                computation_space=ML.ComputationSpace.FOCK
            ),
        )

        n_states = layer.output_size
        amplitudes = torch.randn(n_states, dtype=torch.complex64)
        amplitudes = amplitudes / amplitudes.abs().pow(2).sum().sqrt()

        output = layer(amplitudes)

        assert output.shape[-1] == n_states
        assert torch.all(torch.isfinite(output))

    def test_complex_tensor_batched_forward(self):
        """Batched complex tensor should work correctly."""
        circuit = pcvl.Circuit(4)
        layer = ML.QuantumLayer(
            circuit=circuit,
            n_photons=2,
            measurement_strategy=ML.MeasurementStrategy.amplitudes(
                computation_space=ML.ComputationSpace.FOCK
            ),
        )

        batch_size = 5
        n_states = layer.output_size
        amplitudes = torch.randn(batch_size, n_states, dtype=torch.complex64)
        amplitudes = (
            amplitudes / amplitudes.abs().pow(2).sum(dim=-1, keepdim=True).sqrt()
        )

        output = layer(amplitudes)

        assert output.shape[0] == batch_size
        assert torch.all(torch.isfinite(output))


class TestErrorHandling:
    """Test suite for error handling in input validation."""

    def test_unsupported_type_raises_typeerror(self):
        """Unsupported input types should fail with clear TypeError."""
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=False, name="pre_mix")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=False, name="post_mix")

        layer = ML.QuantumLayer(
            input_size=2,
            input_state=[1, 0, 1, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.probs(),
        )

        with pytest.raises(TypeError) as exc_info:
            layer([1, 2, 3])  # List instead of tensor

        assert "Unsupported input types" in str(exc_info.value)
        assert "list" in str(exc_info.value)

    def test_mixed_inputs_clear_error_message(self):
        """Mixed inputs should provide clear error message."""
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=False, name="pre_mix")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=False, name="post_mix")

        layer = ML.QuantumLayer(
            input_size=2,
            input_state=[1, 0, 1, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.probs(),
        )

        tensor = torch.rand(2, 2)
        sv = StateVector.from_basic_state([1, 0, 1, 0])

        with pytest.raises(TypeError) as exc_info:
            layer(tensor, sv)

        assert "mix" in str(exc_info.value).lower() or "Cannot" in str(exc_info.value)
