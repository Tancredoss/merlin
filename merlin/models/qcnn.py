from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import perceval as pcvl
import torch

from ..algorithms import QuantumLayer
from ..core import ComputationSpace, StateVector
from ..core.partial_measurement import PartialMeasurement
from ..measurement import MeasurementStrategy
from ..utils.grouping import LexGrouping


class QCNNClassifier(torch.nn.Module):
    """ """

    def __init__(
        self, input_shape: tuple, num_classes: int, stages: list[_Stage] | None = None
    ):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.stages = stages

        # Verify inputs
        max_input_size = 28  # Set to 28 by default (to reevaluate)
        if not isinstance(input_shape, tuple):
            raise TypeError("input_shape must have tuple type")
        if not isinstance(input_shape[0], int) or not isinstance(input_shape[1], int):
            raise TypeError("input_shape elements must have int type")
        if len(input_shape) != 2:
            raise ValueError("input_shape must be a tuple of size 2.")
        if type(input_shape[0]) is not int or type(input_shape[1]) is not int:
            raise ValueError("input_shape must contain integers.")
        if input_shape[0] != input_shape[1]:
            raise ValueError(
                "input_shape must represent a square (i.e. input_shape[0] == input_shape[1])."
            )
        if input_shape[0] <= 0 or input_shape[1] <= 0:
            raise ValueError("input_shape must contain values superior to 0.")
        if input_shape[0] > max_input_size:
            raise ValueError(
                f"input_shape values must be inferior or equal to {max_input_size}."
            )

        if num_classes <= 0:
            raise ValueError("num_classes must be superior to 0.")
        if type(num_classes) is not int:
            raise TypeError("num_classes must have int type.")

        if stages is not None and type(stages) is not list:
            raise TypeError("stages must be None or have the list type.")

        self._resolved_stages = self.resolve_stages()
        self.qcnn = self.build_qcnn_model()

    @property
    def resolved_stages(self) -> list[_Stage]:
        return self._resolved_stages

    class _QCNNStageTypes(Enum):
        QConv = "QConv"
        QPool = "QPool"
        QDense = "QDense"

    @dataclass
    class _Stage:
        def __init__(self, type: QCNNClassifier._QCNNStageTypes):
            self.type = type

    class QConv(_Stage):
        def __init__(self, kernel_size: int, stride: int):
            self.type = QCNNClassifier._QCNNStageTypes.QConv
            self.kernel_size = kernel_size
            self.stride = stride

            # Verification of inputs
            if kernel_size <= 0:
                raise ValueError("kernel_size must be superior to 0.")
            if type(kernel_size) is not int:
                raise TypeError("kernel_size must have int type.")
            if stride <= 0:
                raise ValueError("stride must be superior to 0.")
            if type(stride) is not int:
                raise TypeError("stride must have int type.")

        def __eq__(self, other):
            if not isinstance(other, type(self)):
                return NotImplemented
            return (
                self.type == other.type
                and self.kernel_size == other.kernel_size
                and self.stride == other.stride
            )

    class QPool(_Stage):
        def __init__(self, kernel_size: int):
            self.type = QCNNClassifier._QCNNStageTypes.QPool
            self.kernel_size = kernel_size

            # Verification of input
            if kernel_size <= 1:
                raise ValueError("kernel_size must be superior to 1.")
            if type(kernel_size) is not int:
                raise TypeError("kernel_size must have int type.")

        def __eq__(self, other):
            if not isinstance(other, type(self)):
                return NotImplemented
            return self.type == other.type and self.kernel_size == other.kernel_size

    class QDense(_Stage):
        def __init__(self):
            self.type = QCNNClassifier._QCNNStageTypes.QDense

        def __eq__(self, other):
            if not isinstance(other, type(self)):
                return NotImplemented
            return self.type == other.type

    def resolve_stages(self) -> list[_Stage]:
        """
        Resolving of stages:

        If no stage was specified -> return default stage structure

        If stages were specified -> verify that the structure respects the following constraints:
        1. QDense must only be the last stage specified (i.e. only one QDense that is mandatory as the last stage)
        2. QConv kernel_size is inferior or equal to the register dimension and superior or equal to the stride
        3. QPool kernel size must divide the dimension of the registers
        """
        # Check if stages is None or empty
        if self.stages is None or not self.stages:
            resolved_stages: list[QCNNClassifier._Stage] = []
            # Default stages
            resolved_stages.append(QCNNClassifier.QConv(2, 2))
            resolved_stages.append(QCNNClassifier.QPool(2))
            resolved_stages.append(QCNNClassifier.QDense())
            return resolved_stages

        # If stages were specified
        resolved_stages: list[QCNNClassifier._Stage] = self.stages
        # Check that only last stage is QDense
        for i, stage in enumerate(self.stages):
            if stage.type == QCNNClassifier._QCNNStageTypes.QDense and i != (
                len(self.stages) - 1
            ):
                raise ValueError(
                    "Invalid stage specification: only last stage can be QDense"
                )
            if (
                i == (len(self.stages) - 1)
                and stage.type != QCNNClassifier._QCNNStageTypes.QDense
            ):
                raise ValueError(
                    "Invalid stage specification: last stage has to be QDense"
                )

        # Check QConv and QPool compatibility with current dimensions
        dim = self.input_shape[0]
        for stage in self.stages:
            if isinstance(stage, QCNNClassifier.QConv):
                # Appropriate conv if kernel_size <= dim of register
                appropriate_conv = dim >= stage.kernel_size
                if not appropriate_conv:
                    raise ValueError(
                        f"Invalid stage specification: current spatial dimension ({dim}) must be superior or equal to the convolution kernel size ({stage.kernel_size})."
                    )
                # Appropriate conv if kernel_size >= stride; else some modes are not covered
                appropriate_conv = stage.kernel_size >= stage.stride
                if not appropriate_conv:
                    raise ValueError(
                        f"Invalid stage specification: current convolution kernel size ({stage.kernel_size}) must be superior or equal to convolution stride ({stage.stride})."
                    )
                # Give freedom over the stride to the user. If the convolution window exceeds current dimension, it is not added to the circuit
                # TOVERIFY
            elif isinstance(stage, QCNNClassifier.QPool):
                appropriate_pooling = dim % stage.kernel_size == 0
                if not appropriate_pooling:
                    raise ValueError(
                        f"Invalid stage specification: current spatial dimension ({dim}) must be divisible by the pooling kernel size ({stage.kernel_size})."
                    )
                # Adjust current dimension after the pooling
                dim = dim - (dim / stage.kernel_size)

            elif isinstance(stage, QCNNClassifier.QDense):
                continue

            else:
                raise ValueError(
                    f"Invalid stage type; must be QConv, QPool or QDense but got: {type(stage)}"
                )

        return resolved_stages

    def summary(self):
        """Return a concise, human-readable description of the model architecture."""
        stage_parts = []
        for stage in self._resolved_stages:
            if isinstance(stage, QCNNClassifier.QConv):
                stage_parts.append(
                    f"QConv(kernel_size={stage.kernel_size}, stride={stage.stride})"
                )
            elif isinstance(stage, QCNNClassifier.QPool):
                stage_parts.append(f"QPool(kernel_size={stage.kernel_size})")
            else:
                stage_parts.append("QDense()")

        stage_str = " -> ".join(stage_parts)
        return (
            "QCNNClassifier("
            f"input_shape={self.input_shape}, "
            f"num_classes={self.num_classes}, "
            f"stages=[{stage_str}]"
            ")"
        )

    def export_config(self):
        """
        Export a serializable architecture config (weights are intentionally excluded).
        """
        serializable_stages = []
        for stage in self._resolved_stages:
            stage_cfg = {"type": stage.type.value}
            if isinstance(stage, QCNNClassifier.QConv):
                stage_cfg["kernel_size"] = stage.kernel_size
                stage_cfg["stride"] = stage.stride
            elif isinstance(stage, QCNNClassifier.QPool):
                stage_cfg["kernel_size"] = stage.kernel_size
            serializable_stages.append(stage_cfg)

        return {
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            "_resolved_stages": serializable_stages,
        }

    def build_qcnn_model(self):
        empty_circuit = pcvl.Circuit(sum(self.input_shape))
        current_shape = self.input_shape
        qcnn_layers = []
        layer_names = []
        num_qconv = 0
        num_qpool = 0

        # Keep the last stage (QDense) for after
        for stage in self.resolved_stages[:-1]:
            if isinstance(stage, QCNNClassifier.QConv):
                num_qconv += 1
                conv_circuit = self.build_conv_circuit(current_shape, stage)
                conv_layer = QuantumLayer(
                    circuit=conv_circuit,
                    n_photons=2,
                    trainable_parameters=["px"],
                    measurement_strategy=MeasurementStrategy.amplitudes(
                        computation_space=ComputationSpace.FOCK
                    ),
                )
                qcnn_layers.append(conv_layer)
                layer_names.append(f"QConv_{num_qconv}")

            elif isinstance(stage, QCNNClassifier.QPool):
                num_qpool += 1
                measured_modes, reinsert_modes, new_shape = self.resolve_pooling_modes(
                    current_shape, stage
                )
                empty_circuit = pcvl.Circuit(sum(current_shape))
                pool_layer = QuantumLayer(
                    circuit=empty_circuit,
                    n_photons=2,
                    measurement_strategy=MeasurementStrategy.partial(
                        measured_modes, computation_space=ComputationSpace.FOCK
                    ),
                )
                qcnn_layers.append(pool_layer)
                layer_names.append(f"QPool_{num_qpool}")
                # Pooling post-process
                # TODO
                # Change shape of circuit after pooling
                current_shape = new_shape

            else:
                raise TypeError(f"Unknown stage type encountered: {type(stage)}")

        # Apply QDense (mandatory last stage)
        dense_circuit = self.build_dense_circuit(current_shape)
        dense_layer = QuantumLayer(
            circuit=dense_circuit,
            n_photons=2,
            trainable_parameters=["px"],
            measurement_strategy=MeasurementStrategy.mode_expectations(
                computation_space=ComputationSpace.FOCK
            ),
        )

        # Readout layer
        readout = LexGrouping(sum(current_shape), self.num_classes)

        qcnn_model = torch.nn.Sequential()
        for name, layer in zip(layer_names, qcnn_layers, strict=False):
            qcnn_model.add_module(name, layer)
        qcnn_model.add_module("QDense_1", dense_layer)
        qcnn_model.add_module("Readout", readout)

        return qcnn_model

    def build_conv_circuit(self, shape, stage):
        """
        Build the complete circuit containing convolutions

        The two registers must remain separate (i.e. no convolution between the two).
        """
        circuit = pcvl.Circuit(sum(shape))

        # First register
        i = 0
        while i < shape[0] and (i + stage.kernel_size <= shape[0]):
            circuit = self.build_single_conv(i, stage.kernel_size, circuit)
            i += stage.stride

        # Second register
        i = shape[0]
        while (i < shape[0] * 2) and (i + stage.kernel_size <= shape[0] * 2):
            circuit = self.build_single_conv(i, stage.kernel_size, circuit)
            i += stage.stride

        return circuit

    def build_single_conv(self, i, kernel_size, circuit):
        """
        Build a single convolutional window.

        A single convolution is made up of several beam splitters (BS) that allow photon passage from any mode to any mode within the convolution kernel_size.
        """
        # Slide the BS down on the circuit
        for j in range(i, i + kernel_size - 1):
            circuit.add(j, pcvl.BS(theta=pcvl.P(f"px_bs_down_{i}_{j}")))
        # Slide the BS up on the circuit
        for k in range(j - 1, i - 1, -1):
            circuit.add(k, pcvl.BS(theta=pcvl.P(f"px_bs_up_{i}_{k}")))
        return circuit

    def resolve_pooling_modes(self, shape, stage):
        """
        Get measured_modes, reinsert_modes and new_shape for the QPool layer.

        reinsert_mode is always measured_mode + 1.
        """
        measured_modes = []
        reinsert_modes = []

        # First register
        i = 0
        while i < shape[0] and (i + stage.kernel_size <= shape[0]):
            measured_modes.append(i)
            reinsert_modes.append(i + 1)
            i += stage.kernel_size
        num_mesured_modes_0 = len(measured_modes)
        new_shape_0 = shape[0] - num_mesured_modes_0

        # Second register
        i = shape[0]
        while (i < shape[0] * 2) and (i + stage.kernel_size <= shape[0] * 2):
            measured_modes.append(i)
            reinsert_modes.append(i + 1)
            i += stage.kernel_size
        new_shape_1 = shape[1] - len(measured_modes) + num_mesured_modes_0

        assert new_shape_0 == new_shape_1, (
            f"New shape after QPool must have the same shape[0] and shape[1] but got new_shape[0]: {new_shape_0} and new_shape[1]: {new_shape_1}"
        )

        return measured_modes, reinsert_modes, (new_shape_0, new_shape_1)

    def build_dense_circuit(self, shape):
        """
        BS based circuit based on the circuit presented in Monbroussou et al.
        """
        circuit = pcvl.Circuit(sum(shape))

        # Slide the BS down the circuit
        i = 0
        while i < sum(shape) - 1:
            circuit.add(i, pcvl.BS(theta=pcvl.P(f"px_dense_bs_down_{i}")))
            # Add BS above if there is space
            j = i - 2
            while j > -1:
                circuit.add(j, pcvl.BS(theta=pcvl.P(f"px_dense_bs_down_{i}_{j}")))
                j -= 2
            i += 1

        # Slide the BS up the circuit
        i = sum(shape) - 3
        while i > -1:
            circuit.add(i, pcvl.BS(theta=pcvl.P(f"px_dense_bs_up_{i}")))
            # Add BS above if there is space
            j = i - 2
            while j > -1:
                circuit.add(j, pcvl.BS(theta=pcvl.P(f"px_dense_bs_up_{i}_{j}")))
                j -= 2
            i -= 1

        return circuit

    def forward(self, x):
        """
        Expects input x, Tensor, of shape (batch_size, 1, input_size[0], input_size[1]).
        The second dimension represents channels: current implementation only supports one channel.

        Returns logits with shape (batch_size, num_classes).
        """
        if not len(x.shape) == 4:
            raise ValueError(
                "Model input is expected to have the shape: [batch_size, 1, input_size[0], input_size[1]]"
            )
        if not x.shape[1] == 1:
            raise ValueError(
                "Model input must only have 1 channel at its second dimension."
            )
        if not x.shape[2] == x.shape[3] == self.input_shape[0]:
            raise ValueError(
                "Third and fourth input dimension must fit with the specified input_shape."
            )
        batch_size = x.shape[0]

        # Encode x using amplitude encoding
        x = self.amplitude_encode(x)

        for layer_index, (name, layer) in enumerate(self.qcnn.named_children()):
            x = layer(x)
            if name[:5] == "QPool":
                x = self.postprocess_pooling(x, layer_index + 1)
                break

        logits = x
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (batch_size, self.num_classes)
        return logits

    def recursive_forward(self, x, layer_index):
        """
        Recursive version of the forward method. Used to compute forward across the mixed states after pooling.

        Expects input x of type merlin.StateVector, already amplitude encoded and of shape (batch_size, basis_size).

        Returns logits with shape (batch_size, num_classes).
        """
        for new_layer_index, (name, layer) in enumerate(
            list(self.qcnn.named_children())[layer_index:], start=layer_index
        ):
            x = layer(x)
            if name.startswith("QPool"):
                x = self.postprocess_pooling(x, new_layer_index + 1)
                break

        return x

    def amplitude_encode(self, x: torch.Tensor):
        """
        Expects input x, Tensor, of shape (batch_size, 1, input_size[0], input_size[1]).

        Returns StateVector object: amplitude encoded data with shape (batch_size, basis_size).

        basis_size depends on the number of modes (sum(self.input_shape)) and photons (2).
        """
        batch_size = x.shape[0]

        # Prepare amplitude encoded tensor `state_tensor`
        empty_tensor = torch.tensor([])
        state_vector = StateVector(empty_tensor, sum(self.input_shape), 2)
        basis_size = state_vector.basis_size
        state_tensor = torch.zeros((batch_size, basis_size), dtype=torch.complex64)

        for i in range(self.input_shape[0]):
            for j in range(self.input_shape[1]):
                # Build basic state
                basic_state = [0] * self.input_shape[0] + [0] * self.input_shape[1]
                basic_state[i] = 1
                basic_state[self.input_shape[0] + j] = 1
                basic_state_vector = StateVector.from_basic_state(basic_state)
                repeated_basic_state_tensor = (
                    basic_state_vector.tensor.to_dense()
                    .unsqueeze(0)
                    .repeat(batch_size, 1)
                )
                # Scale by pixel value
                pixel_scaling = x[:, :, i, j]
                pixel_state_tensor = pixel_scaling * repeated_basic_state_tensor
                # Add this pixel scaled state vector to the amplitude encoded tensor
                state_tensor += pixel_state_tensor

        # Convert from tensor to StateVector
        final_state_vector = StateVector.from_tensor(
            state_tensor, n_modes=sum(self.input_shape), n_photons=2
        )
        # Normalization
        final_state_vector_norm = final_state_vector.clone().normalize()

        return final_state_vector_norm

    def postprocess_pooling(self, x: PartialMeasurement, layer_index: int):
        """
        Expects input x of type merlin.PartialMeasurement, from which mixed states are created and sent through the remaining of the QCNN to then obtain the output logits.

        Returns tensor of shape (batch_size, num_classes)
        """
        batch_size = x.branches[0].probability.shape[0]
        branches = x.branches
        measured_modes = x.measured_modes
        x_combine = torch.zeros(
            batch_size,
            self.num_classes,
            dtype=torch.float64,
            device=x.branches[0].amplitudes.device,
        )
        for branch in branches:
            outcome = branch.outcome
            probabilities = branch.probability
            state_vector = branch.amplitudes

            possible = self.verify_outcome(list(outcome), probabilities)
            if possible:
                # Reinsert measured photons
                insertions = 0
                for index, outcome_elem in enumerate(outcome):
                    # Maximum of two insertions since there are two photons
                    if insertions == 2:
                        break
                    if outcome_elem > 0:
                        measured_mode = measured_modes[index]
                        reinsert_mode = measured_mode - index
                        # Reinsertion
                        state_vector = self.reinsert_photon(
                            state_vector.clone(), reinsert_mode
                        )
                        insertions += 1

                assert state_vector.n_photons == 2
                # Continue QCNN pipeline on that specific state_vector among the mixed states
                # recursive_forward is called iteratively. We might optimize to run in parallel.
                x_result = self.recursive_forward(state_vector, layer_index)

                assert x_result.shape == (batch_size, self.num_classes)
                assert probabilities.shape == (batch_size,)
                # Combine results from all mixed state computations
                x_combine = x_combine + probabilities.unsqueeze(1) * x_result

        return x_combine

    def verify_outcome(self, outcome, probabilities):
        """
        Verification method: verifies that possible outcomes are allowed in the QCNN setting

        Returns boolean that indicate whether or not the outcome is possible
        """
        possible_outcome = True
        assert len(outcome) % 2 == 0
        half_index = int(len(outcome) / 2)
        first_register_outcome = outcome[:half_index]
        second_register_outcome = outcome[half_index:]

        photon_measured = False
        for outcome_elem in first_register_outcome:
            # Forbidden to measure more than 1 photon in QCNN setting
            if outcome_elem > 1:
                possible_outcome = False
                assert torch.allclose(
                    probabilities, torch.zeros_like(probabilities), atol=1e-6
                ), (
                    f"Expected 0 probabilities but got: {probabilities}, for outcome: {outcome}"
                )
                break
            # Forbidden to measure more than 1 photon per register in QCNN setting
            if outcome_elem == 1:
                if photon_measured:
                    possible_outcome = False
                    assert torch.allclose(
                        probabilities, torch.zeros_like(probabilities), atol=1e-6
                    ), (
                        f"Expected 0 probabilities but got: {probabilities}, for outcome: {outcome}"
                    )
                else:
                    photon_measured = True
                continue
            # Forbidden to have measurement result different from 0 or 1 in QCNN setting
            if outcome_elem != 0:
                possible_outcome = False
                assert torch.allclose(
                    probabilities, torch.zeros_like(probabilities), atol=1e-6
                ), (
                    f"Expected 0 probabilities but got: {probabilities}, for outcome: {outcome}"
                )
                break

        photon_measured = False
        for outcome_elem in second_register_outcome:
            # Forbidden to measure more than 1 photon in QCNN setting
            if outcome_elem > 1:
                possible_outcome = False
                assert torch.allclose(
                    probabilities, torch.zeros_like(probabilities), atol=1e-6
                ), (
                    f"Expected 0 probabilities but got: {probabilities}, for outcome: {outcome}"
                )
                break
            # Forbidden to measure more than 1 photon per register in QCNN setting
            if outcome_elem == 1:
                if photon_measured:
                    possible_outcome = False
                    assert torch.allclose(
                        probabilities, torch.zeros_like(probabilities), atol=1e-6
                    ), (
                        f"Expected 0 probabilities but got: {probabilities}, for outcome: {outcome}"
                    )
                else:
                    photon_measured = True
                continue
            # Forbidden to have measurement result different from 0 or 1 in QCNN setting
            if outcome_elem != 0:
                possible_outcome = False
                assert torch.allclose(
                    probabilities, torch.zeros_like(probabilities), atol=1e-6
                ), (
                    f"Expected 0 probabilities but got: {probabilities}, for outcome: {outcome}"
                )
                break

        return possible_outcome

    def reinsert_photon(self, state_vector: StateVector, reinsert_mode: int):
        """
        Inserts one photon at mode `reinsert_mode` and returns new state vector.

        Torch computation graph is kept.
        """
        state_tensor = state_vector.tensor
        batch_size = state_tensor.shape[0]
        states = list(state_vector.basis.iter_states())

        new_dummy_state_vector = StateVector(
            torch.tensor([0]),
            n_modes=state_vector.n_modes,
            n_photons=state_vector.n_photons + 1,
        )
        new_state_basis_size = new_dummy_state_vector.basis_size
        new_state_tensor = torch.zeros(
            batch_size,
            new_state_basis_size,
            dtype=state_tensor.dtype,
            device=state_tensor.device,
        )

        # Setup mapping from old state indices (no photon at mode reinsert_mode) to new state index (photon reinserted)
        for i, state in enumerate(states):
            # i is the fock_to_index(state) returned index on state_vector
            if state[reinsert_mode] == 0:
                new_state = state[:reinsert_mode] + (1,) + state[reinsert_mode + 1 :]
                new_state_index = new_dummy_state_vector.basis.fock_to_index(new_state)
                new_state_tensor[:, new_state_index] = state_tensor[:, i]

        new_state_vector = StateVector(
            new_state_tensor,
            n_modes=state_vector.n_modes,
            n_photons=state_vector.n_photons + 1,
        )
        return new_state_vector
