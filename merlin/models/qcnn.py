from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import perceval as pcvl
import torch

from ..algorithms import QuantumLayer
from ..measurement import MeasurementStrategy


class QCNNClassifier(torch.nn.Module):
    """ """

    def __init__(
        self, input_shape: tuple, num_classes: int, stages: list[_Stage] | None = None
    ) -> None:
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
        self.build_photonic_circuit()

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

    def build_photonic_circuit(self):
        return

    def build_qcnn_model(self):
        empty_circuit = pcvl.Circuit(sum(self.input_shape))
        QuantumLayer(
            circuit=empty_circuit,
            n_photons=2,
            measurement_strategy=MeasurementStrategy.amplitudes(),
        )
        current_shape = self.input_shape
        qcnn_layers = []

        # Keep the last stage (QDense) for after
        for stage in self.resolved_stages[:-1]:
            if isinstance(stage, QCNNClassifier.QConv):
                conv_circuit = self.build_conv_circuit(current_shape, stage)
                conv_layer = QuantumLayer(
                    circuit=conv_circuit,
                    n_photons=2,
                    measurement_strategy=MeasurementStrategy.amplitudes(),
                )
                qcnn_layers.append(conv_layer)

            elif isinstance(stage, QCNNClassifier.QPool):
                measured_modes, reinsert_modes, new_shape = self.resolve_pooling_modes(
                    current_shape, stage
                )
                empty_circuit = pcvl.Circuit(sum(current_shape))
                pool_layer = QuantumLayer(
                    circuit=empty_circuit,
                    n_photons=2,
                    measurement_strategy=MeasurementStrategy.partial(measured_modes),
                )
                qcnn_layers.append(pool_layer)
                # Change shape of circuit after pooling
                current_shape = new_shape

            else:
                raise TypeError(f"Unknown stage type encountered: {type(stage)}")

        # Apply QDense (mandatory last stage)
        dense_circuit = self.build_dense_circuit(current_shape)
        QuantumLayer(
            circuit=dense_circuit,
            n_photons=2,
            measurement_strategy=MeasurementStrategy.mode_expectations(),  # TO VERIFY
        )

        # Finalize complete qcnn model
        # return model

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
            i = i + stage.stride

        # Second register
        i = shape[0]
        while (i < shape[0] * 2) and (i + stage.kernel_size <= shape[0] * 2):
            circuit = self.build_single_conv(i, stage.kernel_size, circuit)
            i = i + stage.stride

        return circuit

    def build_single_conv(self, i, kernel_size, circuit):
        """
        Build a single convolutional window.

        A single convolution is made up of several beam splitters (BS) that allow photon passage from any mode to any mode within the convolution kernel_size.
        """
        # Slide the BS down on the circuit
        for j in range(i, i + kernel_size - 1):
            circuit.add(j, pcvl.BS(theta=pcvl.P(f"px_bs_down_{j}")))
        # Slide the BS up on the circuit
        for k in range(j - 1, i - 1, -1):
            circuit.add(k, pcvl.BS(theta=pcvl.P(f"px_bs_up_{k}")))
        return circuit

    def resolve_pooling_modes(self, shape, stage):
        """
        Get measured_modes, reinsert_modes and new_shape for the QPool layer.

        By default, reinsert_mode is always measured_mode + 1.
        """
        measured_modes = []
        reinsert_modes = []

        # First register
        i = 0
        while i < shape[0] and (i + stage.kernel_size <= shape[0]):
            measured_modes.append(i)
            reinsert_modes.append(i + 1)
        num_mesured_modes_0 = len(measured_modes)
        new_shape_0 = shape[0] - num_mesured_modes_0

        # Second register
        i = shape[0]
        while (i < shape[0] * 2) and (i + stage.kernel_size <= shape[0] * 2):
            measured_modes.append(i)
            reinsert_modes.append(i + 1)
        new_shape_1 = shape[1] - len(measured_modes) + num_mesured_modes_0

        assert new_shape_0 == new_shape_1, (
            f"New shape after QPool must have the same shape[0] and shape[1] but got new_shape[0]: {new_shape_0} and new_shape[1]: {new_shape_1}"
        )

        return measured_modes, reinsert_modes, (new_shape_0, new_shape_1)

    def build_dense_circuit(self, shape):
        return
        # return circuit

    def forward(self, x):
        """
        Expects input x of shape [batch_size, 1, input_size[0], input_size[1]].
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

        # Encode x using amplitude encoding

        # Compute circuit output

        # Get logits
        logits = torch.ones(x.shape[0], self.num_classes)
        return logits
