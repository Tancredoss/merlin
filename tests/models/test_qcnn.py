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

    with pytest.raises(ValueError, match="kernel_size must be superior to 0"):
        QCNNClassifier.QPool(zero_kernel_size)

    with pytest.raises(ValueError, match="kernel_size must be superior to 0"):
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
            [QCNNClassifier.QDense(), QCNNClassifier.QConv(2, 1)],
        )

    with pytest.raises(ValueError, match="last stage has to be QDense"):
        QCNNClassifier(
            accepted_input_shape,
            accepted_num_classes,
            [QCNNClassifier.QConv(2, 1), QCNNClassifier.QPool(2)],
        )

    with pytest.raises(
        ValueError, match="must be divisible by the convolution kernel size"
    ):
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

    # QCNNClassifier._resolved_stages is read only
    with pytest.raises(AttributeError):
        qcnn_classifier.resolved_stages = []


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
