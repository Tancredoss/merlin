from dataclasses import dataclass
from enum import Enum

import torch


class _QCNNStageTypes(Enum):
    QConv = "QConv"
    QPool = "QPool"
    QDense = "QDense"


@dataclass
class _Stage:
    def __init__(self, type: _QCNNStageTypes):
        self.type = type


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

    class QConv(_Stage):
        def __init__(self, kernel_size: int, stride: int):
            self.type = _QCNNStageTypes.QConv
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
            self.type = _QCNNStageTypes.QPool
            self.kernel_size = kernel_size

            # Verification of input
            if kernel_size <= 0:
                raise ValueError("kernel_size must be superior to 0.")
            if type(kernel_size) is not int:
                raise TypeError("kernel_size must have int type.")

        def __eq__(self, other):
            if not isinstance(other, type(self)):
                return NotImplemented
            return self.type == other.type and self.kernel_size == other.kernel_size

    class QDense(_Stage):
        def __init__(self):
            self.type = _QCNNStageTypes.QDense

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
        2. QConv and QPool kernel sizes must divide the dimension of the registers
        """
        resolved_stages: list[_Stage] = []
        # Check if stages is None or empty
        if self.stages is None or not self.stages:
            # Default stages
            resolved_stages.append(QCNNClassifier.QConv(2, 2))
            resolved_stages.append(QCNNClassifier.QPool(2))
            resolved_stages.append(QCNNClassifier.QDense())
            return resolved_stages

        # If stages were specified
        resolved_stages: list[_Stage] = self.stages
        # Check that only last stage is QDense
        for i, stage in enumerate(self.stages):
            if stage.type == _QCNNStageTypes.QDense and i != (len(self.stages) - 1):
                raise ValueError(
                    "Invalid stage specification: only last stage can be QDense"
                )
            if i == (len(self.stages) - 1) and stage.type != _QCNNStageTypes.QDense:
                raise ValueError(
                    "Invalid stage specification: last stage has to be QDense"
                )

        # Check QConv and QPool compatibility with current dimensions
        dim = self.input_shape[0]
        for stage in self.stages:
            if isinstance(stage, QCNNClassifier.QConv):
                appropriate_conv = dim % stage.kernel_size == 0
                if not appropriate_conv:
                    raise ValueError(
                        f"Invalid stage specification: current spatial dimension ({dim}) must be divisible by the convolution kernel size ({stage.kernel_size})."
                    )
            elif isinstance(stage, QCNNClassifier.QPool):
                appropriate_pooling = dim % stage.kernel_size == 0
                if not appropriate_pooling:
                    raise ValueError(
                        f"Invalid stage specification: current spatial dimension ({dim}) must be divisible by the pooling kernel size ({stage.kernel_size})."
                    )
                # Adjust current dimension after the pooling
                dim = dim - (dim / stage.kernel_size)

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
