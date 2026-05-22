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
# DEALINGS IN THE SOFTWARE.

"""Tests for the PhotonicGenerator model."""

from __future__ import annotations

from typing import cast

import pytest
import torch
from torch import nn

import merlin as ML
from merlin.core.partial_measurement import PartialMeasurement
from merlin.models import GeneratorMeasurements


def _make_layer(
    *,
    input_size: int = 2,
    measurement_strategy: ML.MeasurementStrategy | None = None,
) -> ML.QuantumLayer:
    """Build a small trainable layer for generator tests."""
    builder = ML.CircuitBuilder(n_modes=max(3, input_size + 1))
    builder.add_entangling_layer(trainable=True, name="U1")
    builder.add_angle_encoding(modes=list(range(input_size)), name="input")
    builder.add_entangling_layer(trainable=True, name="U2")

    input_state = [0] * builder.n_modes
    input_state[0] = 1
    strategy = measurement_strategy or ML.MeasurementStrategy.probs(
        computation_space=ML.ComputationSpace.FOCK
    )
    return ML.QuantumLayer(
        input_size=input_size,
        builder=builder,
        input_state=input_state,
        measurement_strategy=strategy,
    )


class SumAdapter(nn.Module):
    """Adapter used to prove custom nn.Module adapters are accepted."""

    def forward(self, measurements: GeneratorMeasurements) -> torch.Tensor:
        tensors = []
        for output in measurements.outputs:
            if not isinstance(output, torch.Tensor):
                raise TypeError("SumAdapter only supports tensor outputs.")
            tensors.append(output.reshape(output.shape[0], -1))
        return torch.cat(tensors, dim=1).sum(dim=1, keepdim=True)


class ConstantLatent(ML.LatentDistribution):
    """Latent sampler used to prove custom latent distributions are accepted."""

    def __init__(self, dim: int, value: float) -> None:
        super().__init__(dim)
        self.value = value

    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        resolved_dtype = dtype if dtype is not None else torch.get_default_dtype()
        return torch.full(
            (batch_size, self.dim),
            self.value,
            device=device,
            dtype=resolved_dtype,
        )


class FirstFeatureAdapter(ML.OutputAdapter):
    """OutputAdapter subclass used to prove the extension contract works."""

    def forward(self, measurements: GeneratorMeasurements) -> torch.Tensor:
        output = measurements.outputs[0]
        if not isinstance(output, torch.Tensor):
            raise TypeError("FirstFeatureAdapter only supports tensor outputs.")
        return output.reshape(output.shape[0], -1)[:, :1]


class PartialProbabilityAdapter(ML.OutputAdapter):
    """Adapter used to prove custom adapters can consume partial measurements."""

    def forward(self, measurements: GeneratorMeasurements) -> torch.Tensor:
        output = measurements.outputs[0]
        if not isinstance(output, PartialMeasurement):
            raise TypeError("PartialProbabilityAdapter expects PartialMeasurement.")
        return output.tensor[:, :1]


def test_latent_distribution_is_abstract():
    with pytest.raises(TypeError, match="abstract class"):
        ML.LatentDistribution(dim=2)


def test_output_adapter_is_abstract():
    with pytest.raises(TypeError, match="abstract class"):
        ML.OutputAdapter()


def test_generator_rejects_empty_layers():
    with pytest.raises(ValueError, match="at least one QuantumLayer"):
        ML.PhotonicGenerator(layers=[], output_adapter=ML.VectorAdapter(size=4))


def test_generator_rejects_non_quantum_layer_objects():
    bad_layer = cast(ML.QuantumLayer, nn.Linear(2, 2))

    with pytest.raises(TypeError, match="QuantumLayer"):
        ML.PhotonicGenerator(
            layers=[bad_layer], output_adapter=ML.VectorAdapter(size=4)
        )


def test_generator_accepts_single_layer():
    generator = ML.PhotonicGenerator(
        layers=[_make_layer(input_size=2)],
        output_adapter=ML.VectorAdapter(size=4),
    )

    assert len(generator) == 1
    assert generator.latent_dim == 2


def test_generator_rejects_inconsistent_latent_dims():
    layers = [_make_layer(input_size=2), _make_layer(input_size=3)]

    with pytest.raises(ValueError, match="same input_size"):
        ML.PhotonicGenerator(layers=layers, output_adapter=ML.VectorAdapter(size=4))


def test_generator_rejects_amplitude_measurement_strategy():
    layer = _make_layer(
        measurement_strategy=ML.MeasurementStrategy.amplitudes(
            computation_space=ML.ComputationSpace.FOCK
        )
    )

    with pytest.raises(ValueError, match="does not support amplitude"):
        ML.PhotonicGenerator(layers=[layer], output_adapter=ML.VectorAdapter(size=4))


def test_generator_accepts_non_amplitude_measurement_strategies():
    mode_layer = _make_layer(
        measurement_strategy=ML.MeasurementStrategy.mode_expectations(
            computation_space=ML.ComputationSpace.FOCK
        )
    )
    partial_layer = _make_layer(
        measurement_strategy=ML.MeasurementStrategy.partial(
            modes=[0],
            computation_space=ML.ComputationSpace.FOCK,
        )
    )

    assert (
        ML.PhotonicGenerator(
            layers=[mode_layer], output_adapter=SumAdapter()
        ).latent_dim
        == 2
    )
    assert (
        ML.PhotonicGenerator(
            layers=[partial_layer], output_adapter=SumAdapter()
        ).latent_dim
        == 2
    )


def test_latent_dim_is_inferred_from_layers():
    generator = ML.PhotonicGenerator(
        layers=[_make_layer(input_size=2), _make_layer(input_size=2)],
        output_adapter=ML.VectorAdapter(size=4),
    )

    assert generator.latent_dim == 2


def test_sample_latent_shape():
    generator = ML.PhotonicGenerator(
        layers=[_make_layer(input_size=3)],
        output_adapter=ML.VectorAdapter(size=4),
    )

    z = generator.sample_latent(batch_size=5)

    assert z.shape == (5, 3)


def test_sample_latent_respects_explicit_dtype():
    generator = ML.PhotonicGenerator(
        layers=[_make_layer(input_size=3)],
        output_adapter=ML.VectorAdapter(size=4),
    )

    z = generator.sample_latent(batch_size=5, dtype=torch.float64)

    assert z.dtype == torch.float64


def test_custom_latent_distribution_is_supported():
    generator = ML.PhotonicGenerator(
        layers=[_make_layer(input_size=2)],
        output_adapter=ML.VectorAdapter(size=4),
        latent=ConstantLatent(dim=2, value=0.25),
    )

    z = generator.sample_latent(batch_size=3, dtype=torch.float64)

    assert z.dtype == torch.float64
    assert torch.allclose(z, torch.full((3, 2), 0.25, dtype=torch.float64))


def test_custom_latent_distribution_dimension_must_match_layers():
    with pytest.raises(ValueError, match="Latent dimension"):
        ML.PhotonicGenerator(
            layers=[_make_layer(input_size=2)],
            output_adapter=ML.VectorAdapter(size=4),
            latent=ConstantLatent(dim=3, value=0.25),
        )


def test_normal_latent_validates_configuration_and_batch_size():
    with pytest.raises(ValueError, match="dim"):
        ML.NormalLatent(dim=0)
    with pytest.raises(ValueError, match="std"):
        ML.NormalLatent(dim=2, std=0.0)

    latent = ML.NormalLatent(dim=2)
    with pytest.raises(TypeError, match="batch_size"):
        latent.sample(cast(int, 1.0))
    with pytest.raises(ValueError, match="batch_size"):
        latent.sample(batch_size=0)


def test_measure_returns_one_output_per_layer():
    layers = [_make_layer(input_size=2), _make_layer(input_size=2)]
    generator = ML.PhotonicGenerator(
        layers=layers,
        output_adapter=ML.VectorAdapter(size=8),
    )
    z = torch.randn(3, generator.latent_dim)

    measurements = generator.measure(z)

    assert len(measurements.outputs) == 2
    assert len(measurements.output_keys) == 2
    assert measurements.output_keys[0] == tuple(layers[0].output_keys)
    assert measurements.outputs[0].shape[0] == 3


def test_forward_uses_output_adapter():
    generator = ML.PhotonicGenerator(
        layers=[_make_layer(input_size=2), _make_layer(input_size=2)],
        output_adapter=SumAdapter(),
    )
    z = torch.randn(4, generator.latent_dim)

    output = generator(z)

    assert output.shape == (4, 1)


def test_output_adapter_subclass_is_supported():
    generator = ML.PhotonicGenerator(
        layers=[_make_layer(input_size=2)],
        output_adapter=FirstFeatureAdapter(),
    )
    z = torch.randn(4, generator.latent_dim)

    output = generator(z)

    assert output.shape == (4, 1)


def test_partial_measurement_output_can_use_custom_adapter():
    partial_layer = _make_layer(
        measurement_strategy=ML.MeasurementStrategy.partial(
            modes=[0],
            computation_space=ML.ComputationSpace.FOCK,
        )
    )
    generator = ML.PhotonicGenerator(
        layers=[partial_layer],
        output_adapter=PartialProbabilityAdapter(),
    )
    z = torch.randn(4, generator.latent_dim)

    output = generator(z)

    assert output.shape == (4, 1)


def test_generate_samples_latent_and_forwards():
    generator = ML.PhotonicGenerator(
        layers=[_make_layer(input_size=2)],
        output_adapter=ML.VectorAdapter(size=5),
    )

    output = generator.generate(batch_size=6)

    assert output.shape == (6, 5)


def test_getitem_returns_quantum_layer():
    layer = _make_layer(input_size=2)
    generator = ML.PhotonicGenerator(
        layers=[layer],
        output_adapter=ML.VectorAdapter(size=4),
    )

    assert generator[0] is layer


def test_len_returns_number_of_layers():
    generator = ML.PhotonicGenerator(
        layers=[_make_layer(input_size=2), _make_layer(input_size=2)],
        output_adapter=ML.VectorAdapter(size=4),
    )

    assert len(generator) == 2


def test_vector_adapter_output_shape():
    adapter = ML.VectorAdapter(size=5)
    measurements = GeneratorMeasurements(
        outputs=(torch.ones(2, 2), torch.ones(2, 4)),
        output_keys=((), ()),
    )

    output = adapter(measurements)

    assert output.shape == (2, 5)


def test_vector_adapter_center_crops_and_zero_pads():
    measurements = GeneratorMeasurements(
        outputs=(torch.arange(6, dtype=torch.float32).reshape(1, 6),),
        output_keys=((),),
    )

    cropped = ML.VectorAdapter(size=4)(measurements)
    padded = ML.VectorAdapter(size=8)(measurements)

    assert torch.equal(cropped, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    assert torch.equal(padded, torch.tensor([[0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0]]))


def test_image_adapter_output_shape_grayscale():
    adapter = ML.ImageAdapter(shape=(2, 3))
    measurements = GeneratorMeasurements(
        outputs=(torch.ones(4, 2), torch.ones(4, 4)),
        output_keys=((), ()),
    )

    output = adapter(measurements)

    assert output.shape == (4, 1, 2, 3)


def test_image_adapter_output_shape_multichannel():
    adapter = ML.ImageAdapter(shape=(2, 2, 3))
    measurements = GeneratorMeasurements(
        outputs=(torch.ones(4, 12),),
        output_keys=((),),
    )

    output = adapter(measurements)

    assert output.shape == (4, 2, 2, 3)


def test_parameters_include_all_layer_parameters():
    layers = [_make_layer(input_size=2), _make_layer(input_size=2)]
    generator = ML.PhotonicGenerator(
        layers=layers,
        output_adapter=ML.VectorAdapter(size=4),
    )

    generator_param_ids = {id(param) for param in generator.parameters()}
    for layer in layers:
        for param in layer.parameters():
            assert id(param) in generator_param_ids


def test_state_dict_includes_all_layer_state():
    layers = [_make_layer(input_size=2), _make_layer(input_size=2)]
    generator = ML.PhotonicGenerator(
        layers=layers,
        output_adapter=ML.VectorAdapter(size=4),
    )

    generator_state = generator.state_dict()

    for layer_index, layer in enumerate(layers):
        for key in layer.state_dict():
            assert f"layers.{layer_index}.{key}" in generator_state


def test_gradients_flow_through_photonic_generator():
    generator = ML.PhotonicGenerator(
        layers=[_make_layer(input_size=2)],
        output_adapter=ML.VectorAdapter(size=3),
    )
    z = torch.randn(4, generator.latent_dim)

    output = generator(z)
    loss = output[:, 0].sum()
    loss.backward()

    grads = [param.grad for param in generator.parameters() if param.requires_grad]
    assert any(grad is not None for grad in grads)


def test_forward_rejects_wrong_latent_shape():
    generator = ML.PhotonicGenerator(
        layers=[_make_layer(input_size=2)],
        output_adapter=ML.VectorAdapter(size=3),
    )

    with pytest.raises(ValueError, match="shape"):
        generator(torch.randn(2, 3))


def test_forward_rejects_non_batched_latent_input():
    generator = ML.PhotonicGenerator(
        layers=[_make_layer(input_size=2)],
        output_adapter=ML.VectorAdapter(size=3),
    )

    with pytest.raises(ValueError, match="rank"):
        generator(torch.randn(2))
