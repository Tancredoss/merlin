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

"""Tests for the QCNNClassifier model class"""

import pytest
import torch

from merlin.core import StateVector
from merlin.models import QCNNClassifier


def test_qcnn_basic_api():
    accepted_input_shape = (4, 4)
    non_square_input_shape = (4, 8)
    too_big_input_shape = (1080, 1080)
    three_tuple_input_shape = (4, 4, 4)
    array_input_shape = [4, 4]
    float_input_shape_v1 = (4.0, 4.0)
    float_input_shape_v2 = (float(4), float(4))
    negative_input_shape = (-4, -4)

    accepted_num_classes = 10
    zero_num_classes = 0
    negative_num_classes = -1
    float_num_classes_v1 = 4.0
    float_num_classes_v2 = float(4)

    # API tests for input_shape and num_classes

    qcnn_classifier = QCNNClassifier(accepted_input_shape, accepted_num_classes)

    with pytest.raises(ValueError, match="input_shape must represent a square"):
        QCNNClassifier(non_square_input_shape, accepted_num_classes)

    with pytest.raises(ValueError, match="input_shape must be a tuple of size 2"):
        QCNNClassifier(three_tuple_input_shape, accepted_num_classes)

    with pytest.raises(TypeError, match="input_shape must have tuple type"):
        QCNNClassifier(array_input_shape, accepted_num_classes)

    with pytest.raises(TypeError, match="input_shape elements must have int type"):
        QCNNClassifier(float_input_shape_v1, accepted_num_classes)

    with pytest.raises(TypeError, match="input_shape elements must have int type"):
        QCNNClassifier(float_input_shape_v2, accepted_num_classes)

    with pytest.raises(
        ValueError, match="input_shape must contain values superior to 0"
    ):
        QCNNClassifier(negative_input_shape, accepted_num_classes)

    with pytest.raises(
        ValueError, match="input_shape values must be inferior or equal to 28"
    ):
        QCNNClassifier(too_big_input_shape, accepted_num_classes)

    with pytest.raises(ValueError, match="num_classes must be superior to 0"):
        QCNNClassifier(accepted_input_shape, zero_num_classes)

    with pytest.raises(ValueError, match="num_classes must be superior to 0"):
        QCNNClassifier(accepted_input_shape, negative_num_classes)

    with pytest.raises(TypeError, match="num_classes must have int type"):
        QCNNClassifier(accepted_input_shape, float_num_classes_v1)

    with pytest.raises(TypeError, match="num_classes must have int type"):
        QCNNClassifier(accepted_input_shape, float_num_classes_v2)

    assert qcnn_classifier.input_shape == accepted_input_shape
    assert qcnn_classifier.num_classes == accepted_num_classes


def test_qcnn_stage_api():
    accepted_kernel_size = 2
    zero_kernel_size = 0
    one_kernel_size = 1
    negative_kernel_size = -2
    float_kernel_size = float(2)

    accepted_stride = 2
    zero_stride = 0
    negative_Stride = -2
    float_stride = float(2)

    # QConv and QPool initializations
    qconv = QCNNClassifier.QConv(accepted_kernel_size, accepted_stride)

    with pytest.raises(ValueError, match="kernel_size must be superior to 0"):
        QCNNClassifier.QConv(zero_kernel_size, accepted_stride)

    with pytest.raises(ValueError, match="kernel_size must be superior to 0"):
        QCNNClassifier.QConv(negative_kernel_size, accepted_stride)

    with pytest.raises(ValueError, match="stride must be superior to 0"):
        QCNNClassifier.QConv(accepted_kernel_size, zero_stride)

    with pytest.raises(ValueError, match="stride must be superior to 0"):
        QCNNClassifier.QConv(accepted_kernel_size, negative_Stride)

    with pytest.raises(TypeError, match="kernel_size must have int type"):
        QCNNClassifier.QConv(float_kernel_size, accepted_stride)

    with pytest.raises(TypeError, match="stride must have int type"):
        QCNNClassifier.QConv(accepted_kernel_size, float_stride)

    assert qconv.type.value == "QConv"

    qpool = QCNNClassifier.QPool(accepted_kernel_size)
    assert qpool.type.value == "QPool"
    assert qpool.kernel_size == accepted_kernel_size

    with pytest.raises(ValueError, match="kernel_size must be superior to 1"):
        QCNNClassifier.QPool(zero_kernel_size)

    with pytest.raises(ValueError, match="kernel_size must be superior to 1"):
        QCNNClassifier.QPool(one_kernel_size)

    with pytest.raises(ValueError, match="kernel_size must be superior to 1"):
        QCNNClassifier.QPool(negative_kernel_size)

    with pytest.raises(TypeError, match="kernel_size must have int type"):
        QCNNClassifier.QPool(float_kernel_size)

    accepted_input_shape = (4, 4)
    accepted_num_classes = 10
    qcnn_classifier = QCNNClassifier(accepted_input_shape, accepted_num_classes)

    # Default stages
    stages = []
    stages.append(QCNNClassifier.QConv(2, 2))
    stages.append(QCNNClassifier.QPool(2))
    stages.append(QCNNClassifier.QDense())

    assert qcnn_classifier.resolved_stages == stages
    assert qcnn_classifier.stages is None

    # QCNNClassifier.resolve_stages() cases
    valid_custom_stages = [
        QCNNClassifier.QConv(2, 1),
        QCNNClassifier.QPool(2),
        QCNNClassifier.QDense(),
    ]
    custom_qcnn = QCNNClassifier(
        accepted_input_shape, accepted_num_classes, valid_custom_stages
    )
    assert custom_qcnn.resolved_stages == valid_custom_stages

    with pytest.raises(ValueError, match="only last stage can be QDense"):
        QCNNClassifier(
            accepted_input_shape,
            accepted_num_classes,
            [QCNNClassifier.QDense(), QCNNClassifier.QDense()],
        )

    with pytest.raises(ValueError, match="last stage has to be QDense"):
        QCNNClassifier(
            accepted_input_shape,
            accepted_num_classes,
            [QCNNClassifier.QConv(2, 1), QCNNClassifier.QPool(2)],
        )

    # Accepted
    QCNNClassifier(
        accepted_input_shape,
        accepted_num_classes,
        [QCNNClassifier.QConv(3, 1), QCNNClassifier.QDense()],
    )

    with pytest.raises(
        ValueError, match="must be divisible by the pooling kernel size"
    ):
        QCNNClassifier(
            accepted_input_shape,
            accepted_num_classes,
            [QCNNClassifier.QPool(3), QCNNClassifier.QDense()],
        )

    with pytest.raises(TypeError, match="stages must be None or have the list type"):
        QCNNClassifier(accepted_input_shape, accepted_num_classes, ())

    # kernel_size > input_shape[0]
    with pytest.raises(
        ValueError, match="must be superior or equal to the convolution kernel size"
    ):
        QCNNClassifier(
            accepted_input_shape,
            accepted_num_classes,
            [QCNNClassifier.QConv(5, 1), QCNNClassifier.QDense()],
        )

    # Accepted
    QCNNClassifier(
        accepted_input_shape,
        accepted_num_classes,
        [QCNNClassifier.QConv(4, 1), QCNNClassifier.QDense()],
    )

    # stride > kernel_size
    with pytest.raises(
        ValueError, match="must be superior or equal to convolution stride"
    ):
        QCNNClassifier(
            accepted_input_shape,
            accepted_num_classes,
            [QCNNClassifier.QConv(2, 3), QCNNClassifier.QDense()],
        )

    # QCNNClassifier._resolved_stages is read only
    with pytest.raises(AttributeError):
        qcnn_classifier.resolved_stages = []

    # Try to access stages class through import without QCNNClassifier
    # Have to ignore ruff here or else it replaces unused imports with `pass` statements
    with pytest.raises(ImportError):
        from merlin import QConv  # noqa: F401

    with pytest.raises(ImportError):
        from merlin.models import QPool  # noqa: F401

    with pytest.raises(ImportError):
        from merlin.models.qcnn import QDense  # noqa: F401

    # Try to access private classes through import without QCNNClassifier
    with pytest.raises(ImportError):
        from merlin.models.qcnn import _QCNNStageTypes  # noqa: F401

    with pytest.raises(ImportError):
        from merlin.models.qcnn import _Stage  # noqa: F401

    # Cannot instanciate QCNNClassifier with _Stage objects directly
    qconv_type = QCNNClassifier._QCNNStageTypes.QConv
    qconv_stage = QCNNClassifier._Stage(qconv_type)

    qdense_type = QCNNClassifier._QCNNStageTypes.QDense
    qdense_stage = QCNNClassifier._Stage(qdense_type)

    stages = [qconv_stage, qdense_stage]

    with pytest.raises(ValueError, match="Invalid stage type"):
        QCNNClassifier((4, 4), 2, stages)


def test_qcnn_summary():
    qcnn_classifier = QCNNClassifier((4, 4), 10)

    expected_summary = (
        "QCNNClassifier("
        "input_shape=(4, 4), "
        "num_classes=10, "
        "stages=[QConv(kernel_size=2, stride=2) -> QPool(kernel_size=2) -> QDense()]"
        ")"
    )
    assert qcnn_classifier.summary() == expected_summary


def test_qcnn_export_config():
    stages = [
        QCNNClassifier.QConv(4, 1),
        QCNNClassifier.QDense(),
    ]
    qcnn_classifier = QCNNClassifier((4, 4), 3, stages=stages)

    expected_config = {
        "input_shape": (4, 4),
        "num_classes": 3,
        "_resolved_stages": [
            {"type": "QConv", "kernel_size": 4, "stride": 1},
            {"type": "QDense"},
        ],
    }
    assert qcnn_classifier.export_config() == expected_config

    # Round trip test
    config = qcnn_classifier.export_config()

    # Utils to deserialize the config into actual Stages
    STAGE_REGISTRY = {
        "QConv": QCNNClassifier.QConv,
        "QPool": QCNNClassifier.QPool,
        "QDense": QCNNClassifier.QDense,
    }

    def build_stage(stage_config: dict):
        stage_type = stage_config["type"]
        cls = STAGE_REGISTRY[stage_type]

        # remove "type" key and pass the rest as kwargs
        kwargs = {k: v for k, v in stage_config.items() if k != "type"}
        return cls(**kwargs)

    config_stages = [build_stage(s) for s in config["_resolved_stages"]]
    # Build new QCNNClassifier from the config
    new_qcnn_classifier = QCNNClassifier(
        config["input_shape"], config["num_classes"], config_stages
    )

    assert new_qcnn_classifier.input_shape == qcnn_classifier.input_shape
    assert new_qcnn_classifier.num_classes == qcnn_classifier.num_classes
    assert new_qcnn_classifier.resolved_stages == qcnn_classifier.resolved_stages


def test_qcnn_amplitude_encoding():
    input_shape = (4, 4)
    num_classes = 2
    qcnn = QCNNClassifier(input_shape, num_classes)

    # Build input to amplitude encode
    x_tensor_0 = torch.tensor([
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]).unsqueeze(0)
    x_tensor_1 = torch.tensor([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
    ]).unsqueeze(0)
    x_tensor_2 = torch.tensor([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ]).unsqueeze(0)
    x_tensor_3 = torch.rand((4, 4)).unsqueeze(0)

    x_tensor = torch.cat([x_tensor_0, x_tensor_1, x_tensor_2, x_tensor_3], dim=0)
    x_tensor = x_tensor.unsqueeze(1)

    assert x_tensor.shape == (4, 1, 4, 4)

    amplitude_encoded_state_vector = qcnn.amplitude_encode(x_tensor)

    assert isinstance(amplitude_encoded_state_vector, StateVector)
    basis_size = StateVector(
        torch.tensor([]), n_modes=sum(input_shape), n_photons=2
    ).basis_size
    assert amplitude_encoded_state_vector.shape == (4, basis_size)
    assert amplitude_encoded_state_vector.is_normalized

    # Verify that the obtained state vector is the one expected
    first_state_tensor = amplitude_encoded_state_vector.tensor[0]
    first_expected_state = [
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
    ]  # By definition of the amplitude encoding used (2 registers)
    first_expected_tensor = StateVector.from_basic_state(first_expected_state).tensor
    assert torch.allclose(
        first_state_tensor.to_dense(),
        first_expected_tensor.to_dense(),
        atol=1e-6,
        rtol=1e-6,
    )

    # Verify that the obtained state vector is the one expected
    second_state_tensor = amplitude_encoded_state_vector.tensor[1]
    second_expected_state = [
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
    ]  # By definition of the amplitude encoding used (2 registers)
    second_expected_tensor = StateVector.from_basic_state(second_expected_state).tensor
    assert torch.allclose(
        second_state_tensor.to_dense(),
        second_expected_tensor.to_dense(),
        atol=1e-6,
        rtol=1e-6,
    )

    # Verify that the obtained state vector is a uniform combination of all allowed basic states
    third_state_tensor = amplitude_encoded_state_vector.tensor[2]
    all_basic_states_indices = []
    for i in range(4):
        for j in range(4):
            state = [0] * 8
            state[i] = 1
            state[4 + j] = 1
            basic_state_index = amplitude_encoded_state_vector.index(state)
            all_basic_states_indices.append(basic_state_index)

    allowed_amplitudes = []
    forbidden_amplitudes = []
    for index, amplitude in enumerate(third_state_tensor):
        if index in all_basic_states_indices:
            allowed_amplitudes.append(amplitude)
        else:
            forbidden_amplitudes.append(amplitude)

    allowed = torch.tensor(allowed_amplitudes).to_dense()
    forbidden = torch.tensor(forbidden_amplitudes).to_dense()

    assert torch.allclose(
        allowed[0].unsqueeze(0).repeat(len(allowed)), allowed, atol=1e-6, rtol=1e-6
    )
    assert torch.allclose(torch.zeros_like(forbidden), forbidden, atol=1e-6, rtol=1e-6)

    # Verify that forbidden basic states are not returned
    fourth_state_tensor = amplitude_encoded_state_vector.tensor[3]
    for index, amplitude in enumerate(fourth_state_tensor):
        # Use the same basic state indices
        if index in all_basic_states_indices:
            allowed_amplitudes.append(amplitude)
        else:
            forbidden_amplitudes.append(amplitude)

    allowed = torch.tensor(allowed_amplitudes).to_dense()
    forbidden = torch.tensor(forbidden_amplitudes).to_dense()

    assert torch.allclose(torch.zeros_like(forbidden), forbidden, atol=1e-6, rtol=1e-6)


def test_full_default_qcnn():
    input_shape = (4, 4)
    num_classes = 2
    qcnn = QCNNClassifier(input_shape, num_classes)

    initial_param_values = {}
    for name, param in qcnn.named_parameters():
        print(f"Param {name}")
        before = param.detach().clone()
        initial_param_values[name] = before

    # Generate data and labels
    x_tensor_0 = torch.tensor([
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]).unsqueeze(0)
    x_tensor_1 = torch.tensor([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
    ]).unsqueeze(0)
    x_tensor_2 = torch.tensor([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ]).unsqueeze(0)
    x_tensor_3 = torch.rand((4, 4)).unsqueeze(0)

    x_tensor = torch.cat([x_tensor_0, x_tensor_1, x_tensor_2, x_tensor_3], dim=0)
    x_tensor = x_tensor.unsqueeze(1)

    y_0 = torch.tensor([1, 0], dtype=torch.float).unsqueeze(0)
    y_1 = torch.tensor([1, 0], dtype=torch.float).unsqueeze(0)
    y_2 = torch.tensor([0, 1], dtype=torch.float).unsqueeze(0)
    y_3 = torch.tensor([0, 1], dtype=torch.float).unsqueeze(0)

    y_tensor = torch.cat([y_0, y_1, y_2, y_3], dim=0)

    assert x_tensor.shape == (4, 1, 4, 4)
    assert y_tensor.shape == (4, 2)

    optimizer = torch.optim.Adam(qcnn.parameters(), lr=1e-1)

    qcnn.train()
    optimizer.zero_grad()
    logits = qcnn(x_tensor)

    loss_function = torch.nn.CrossEntropyLoss()
    loss = loss_function(logits, y_tensor)
    loss.backward()

    # Check that gradients exist and are defined
    for name, param in qcnn.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} gradient has NaN/Inf"
        assert torch.any(param.grad.abs() > 1e-8), f"{name} gradient is zero"

    optimizer.step()

    # Check that parameter values have changed after optimizer step
    for name, param in qcnn.named_parameters():
        print(f"Param {name}")
        before = initial_param_values[name]
        after = param.detach().clone()
        assert not torch.allclose(before, after, atol=1e-4, rtol=1e-4)
