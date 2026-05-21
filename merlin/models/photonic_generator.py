"""Photonic generator model built from QuantumLayer heads.

The generator is a thin PyTorch model abstraction: it runs one latent batch
through one or more :class:`~merlin.algorithms.layer.QuantumLayer` heads and
delegates task-specific output shaping to an adapter.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch
from torch import nn
from torch.nn import functional as F

from merlin.algorithms.layer import QuantumLayer
from merlin.core.partial_measurement import PartialMeasurement
from merlin.measurement.strategies import (
    MeasurementKind,
    _resolve_measurement_kind,
)

GeneratorOutput: TypeAlias = torch.Tensor | PartialMeasurement


@dataclass(frozen=True)
class GeneratorMeasurements:
    """Raw outputs produced by a PhotonicGenerator.

    Parameters
    ----------
    outputs : tuple[torch.Tensor | merlin.core.partial_measurement.PartialMeasurement, ...]
        Per-layer batched outputs. Tensor outputs have their first dimension as
        the latent batch dimension. Partial measurements keep their native
        Merlin representation.
    output_keys : tuple[tuple[Any, ...], ...]
        Per-layer output-basis metadata copied from each underlying
        :class:`~merlin.algorithms.layer.QuantumLayer`.
    """

    outputs: tuple[GeneratorOutput, ...]
    output_keys: tuple[tuple[Any, ...], ...]


class LatentDistribution:
    """Base interface for latent distributions used by PhotonicGenerator.

    Parameters
    ----------
    dim : int
        Dimension of the latent vector sampled for each batch item.

    Raises
    ------
    TypeError
        If ``dim`` does not have int type.
    ValueError
        If ``dim`` is not positive.
    """

    dim: int

    def __init__(self, dim: int) -> None:
        if type(dim) is not int:
            raise TypeError("dim must have int type.")
        if dim <= 0:
            raise ValueError("dim must be positive.")
        self.dim = dim

    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Sample a latent batch.

        Parameters
        ----------
        batch_size : int
            Number of latent vectors to sample.
        device : torch.device | None
            Device on which the tensor is created. If omitted, the distribution
            implementation chooses its default.
        dtype : torch.dtype | None
            Floating dtype of the returned tensor. If omitted, the distribution
            implementation chooses its default.

        Returns
        -------
        torch.Tensor
            Tensor with shape ``(batch_size, dim)``.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement sampling.
        """
        raise NotImplementedError


class NormalLatent(LatentDistribution):
    """Independent normal latent distribution.

    Parameters
    ----------
    dim : int
        Dimension of the latent vector sampled for each batch item.
    mean : float
        Mean of the normal distribution. Default is ``0.0``.
    std : float
        Standard deviation of the normal distribution. Default is ``1.0``.

    Raises
    ------
    ValueError
        If ``std`` is not positive.
    """

    def __init__(self, dim: int, mean: float = 0.0, std: float = 1.0) -> None:
        super().__init__(dim)
        if std <= 0:
            raise ValueError("std must be positive.")
        self.mean = float(mean)
        self.std = float(std)

    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Sample normally distributed latent vectors.

        Parameters
        ----------
        batch_size : int
            Number of latent vectors to sample.
        device : torch.device | None
            Device on which the tensor is created. If omitted, CPU is used.
        dtype : torch.dtype | None
            Floating dtype of the returned tensor. If omitted, the current
            PyTorch default dtype is used.

        Returns
        -------
        torch.Tensor
            Tensor with shape ``(batch_size, dim)``.

        Raises
        ------
        TypeError
            If ``batch_size`` does not have int type.
        ValueError
            If ``batch_size`` is not positive.
        """
        if type(batch_size) is not int:
            raise TypeError("batch_size must have int type.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        resolved_dtype = dtype if dtype is not None else torch.get_default_dtype()
        return (
            torch
            .randn(
                batch_size,
                self.dim,
                device=device,
                dtype=resolved_dtype,
            )
            .mul(self.std)
            .add(self.mean)
        )


class OutputAdapter(nn.Module):
    """Base module for adapting generator measurements to classical samples."""

    def forward(self, measurements: GeneratorMeasurements) -> torch.Tensor:
        """Adapt raw generator measurements.

        Parameters
        ----------
        measurements : GeneratorMeasurements
            Raw outputs and output-key metadata returned by
            :meth:`PhotonicGenerator.measure`.

        Returns
        -------
        torch.Tensor
            Task-native generated samples.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement adaptation.
        """
        raise NotImplementedError


class VectorAdapter(OutputAdapter):
    """Concatenate tensor measurements into fixed-width vectors.

    The adapter flattens every tensor output after its batch dimension,
    concatenates the flattened outputs, and center-crops or zero-pads to
    ``size``.

    Parameters
    ----------
    size : int
        Number of features in each generated vector.

    Raises
    ------
    TypeError
        If ``size`` does not have int type.
    ValueError
        If ``size`` is not positive.
    """

    size: int

    def __init__(self, size: int) -> None:
        super().__init__()
        if type(size) is not int:
            raise TypeError("size must have int type.")
        if size <= 0:
            raise ValueError("size must be positive.")
        self.size = size

    def forward(self, measurements: GeneratorMeasurements) -> torch.Tensor:
        """Return fixed-width vectors from tensor measurements.

        Parameters
        ----------
        measurements : GeneratorMeasurements
            Raw generator measurements. All outputs must be batched tensors with
            a common first dimension.

        Returns
        -------
        torch.Tensor
            Tensor with shape ``(batch_size, size)``.

        Raises
        ------
        TypeError
            If any measurement output is not a tensor.
        ValueError
            If no outputs are provided, if tensor outputs are not batched, or if
            batch dimensions differ.
        """
        flattened = _flatten_tensor_outputs(measurements.outputs)
        combined = torch.cat(flattened, dim=1)
        return _center_crop_or_pad(combined, self.size)


class ImageAdapter(OutputAdapter):
    """Adapt tensor measurements to GAN-native image tensors.

    Parameters
    ----------
    shape : tuple[int, int] | tuple[int, int, int]
        Image shape. ``(height, width)`` produces single-channel output with
        shape ``(batch_size, 1, height, width)``. ``(channels, height, width)``
        preserves the specified channel count.

    Raises
    ------
    TypeError
        If ``shape`` is not a tuple of integers.
    ValueError
        If ``shape`` does not have length 2 or 3, or if any dimension is not
        positive.
    """

    shape: tuple[int, int, int]

    def __init__(self, shape: tuple[int, int] | tuple[int, int, int]) -> None:
        super().__init__()
        self.shape = _normalize_image_shape(shape)
        channels, height, width = self.shape
        self._vector_adapter = VectorAdapter(channels * height * width)

    def forward(self, measurements: GeneratorMeasurements) -> torch.Tensor:
        """Return image tensors from raw generator measurements.

        Parameters
        ----------
        measurements : GeneratorMeasurements
            Raw generator measurements accepted by :class:`VectorAdapter`.

        Returns
        -------
        torch.Tensor
            Tensor with shape ``(batch_size, channels, height, width)``.
        """
        vector = self._vector_adapter(measurements)
        channels, height, width = self.shape
        return vector.reshape(vector.shape[0], channels, height, width)


class PhotonicGenerator(nn.Module):
    """Generative model composed from one or more QuantumLayer heads.

    Each generator head receives the same latent batch. The raw per-head
    measurements are exposed through :meth:`measure` and converted to
    task-native samples through ``output_adapter`` in :meth:`forward`.

    Parameters
    ----------
    layers : Sequence[merlin.algorithms.layer.QuantumLayer]
        Non-empty sequence of quantum generator heads. All layers must expose
        the same ``input_size``. Amplitude-output measurement strategies are not
        supported because they do not directly represent classical generated
        samples.
    output_adapter : torch.nn.Module
        Module that maps :class:`GeneratorMeasurements` to a tensor. Built-in
        adapters inherit from :class:`OutputAdapter`, but custom adapters only
        need to be PyTorch modules with a compatible ``forward`` method.
    latent : LatentDistribution | None
        Latent distribution used by :meth:`sample_latent` and :meth:`generate`.
        If omitted, :class:`NormalLatent` with the inferred latent dimension is
        used. Default is ``None``.

    Raises
    ------
    TypeError
        If ``layers`` is not a sequence of
        :class:`~merlin.algorithms.layer.QuantumLayer`, if
        ``output_adapter`` is not a :class:`torch.nn.Module`, or if ``latent`` is
        not a :class:`LatentDistribution`.
    ValueError
        If no layers are provided, if layer input sizes differ, if a layer uses
        amplitude outputs, or if the latent distribution dimension does not
        match the inferred latent dimension.
    """

    layers: nn.ModuleList
    output_adapter: nn.Module
    latent: LatentDistribution

    def __init__(
        self,
        layers: Sequence[QuantumLayer],
        output_adapter: nn.Module,
        latent: LatentDistribution | None = None,
    ) -> None:
        super().__init__()
        validated_layers = _validate_layers(layers)
        self.layers = nn.ModuleList(validated_layers)
        if not isinstance(output_adapter, nn.Module):
            raise TypeError("output_adapter must be a torch.nn.Module.")
        self.output_adapter = output_adapter

        inferred_dim = validated_layers[0].input_size
        if latent is None:
            latent = NormalLatent(inferred_dim)
        elif not isinstance(latent, LatentDistribution):
            raise TypeError("latent must be a LatentDistribution or None.")
        if latent.dim != inferred_dim:
            raise ValueError(
                f"Latent dimension ({latent.dim}) must match layer input_size "
                f"({inferred_dim})."
            )
        self.latent = latent

    @property
    def latent_dim(self) -> int:
        """Dimension of the latent vectors accepted by the generator."""
        return self.latent.dim

    def __getitem__(self, index: int) -> QuantumLayer:
        """Return a generator head by index.

        Parameters
        ----------
        index : int
            Index of the underlying
            :class:`~merlin.algorithms.layer.QuantumLayer`.

        Returns
        -------
        merlin.algorithms.layer.QuantumLayer
            The selected quantum generator head.
        """
        return self.layers[index]

    def __len__(self) -> int:
        """Return the number of quantum generator heads.

        Returns
        -------
        int
            Number of underlying
            :class:`~merlin.algorithms.layer.QuantumLayer` modules.
        """
        return len(self.layers)

    def sample_latent(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Sample latent vectors from the configured latent distribution.

        Parameters
        ----------
        batch_size : int
            Number of latent vectors to sample.
        device : torch.device | None
            Device on which the tensor is created. If omitted, the generator
            uses the first parameter or buffer device, falling back to CPU.
            Default is ``None``.
        dtype : torch.dtype | None
            Floating dtype of the returned tensor. If omitted, the generator
            uses the first floating parameter or buffer dtype, falling back to
            the current PyTorch default dtype. Default is ``None``.

        Returns
        -------
        torch.Tensor
            Tensor with shape ``(batch_size, latent_dim)``.
        """
        resolved_device, resolved_dtype = self._resolve_sample_device_dtype(
            device, dtype
        )
        return self.latent.sample(
            batch_size, device=resolved_device, dtype=resolved_dtype
        )

    def generate(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Sample latent vectors and generate classical samples.

        Parameters
        ----------
        batch_size : int
            Number of samples to generate.
        device : torch.device | None
            Device on which latent vectors are sampled. If omitted, the
            generator chooses a device from its parameters and buffers. Default
            is ``None``.
        dtype : torch.dtype | None
            Floating dtype of sampled latent vectors. If omitted, the generator
            chooses a dtype from its parameters and buffers. Default is
            ``None``.

        Returns
        -------
        torch.Tensor
            Generated samples returned by ``output_adapter``.
        """
        z = self.sample_latent(batch_size, device=device, dtype=dtype)
        return self(z)

    def measure(self, z: torch.Tensor) -> GeneratorMeasurements:
        """Evaluate every quantum generator head on a latent batch.

        Parameters
        ----------
        z : torch.Tensor
            Latent input tensor with shape ``(batch_size, latent_dim)``.

        Returns
        -------
        GeneratorMeasurements
            Per-layer outputs and output-key metadata.

        Raises
        ------
        TypeError
            If ``z`` is not a :class:`torch.Tensor`.
        ValueError
            If ``z`` does not have shape ``(batch_size, latent_dim)``.
        """
        self._validate_latent_input(z)
        outputs = tuple(layer(z) for layer in self.layers)
        output_keys = tuple(tuple(layer.output_keys) for layer in self.layers)
        return GeneratorMeasurements(outputs=outputs, output_keys=output_keys)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate samples from latent vectors.

        Parameters
        ----------
        z : torch.Tensor
            Latent input tensor with shape ``(batch_size, latent_dim)``.

        Returns
        -------
        torch.Tensor
            Task-native generated samples produced by ``output_adapter``.
        """
        return self.output_adapter(self.measure(z))

    def _validate_latent_input(self, z: torch.Tensor) -> None:
        """Validate the latent input shape.

        Parameters
        ----------
        z : torch.Tensor
            Candidate latent tensor.

        Raises
        ------
        TypeError
            If ``z`` is not a :class:`torch.Tensor`.
        ValueError
            If ``z`` is not a rank-2 tensor with the configured latent
            dimension.
        """
        if not isinstance(z, torch.Tensor):
            raise TypeError("z must be a torch.Tensor.")
        if z.dim() != 2:
            raise ValueError(
                "PhotonicGenerator expects z with shape "
                f"(batch_size, {self.latent_dim}), but got rank {z.dim()}."
            )
        if z.shape[1] != self.latent_dim:
            raise ValueError(
                "PhotonicGenerator expects z with shape "
                f"(batch_size, {self.latent_dim}), but got {tuple(z.shape)}."
            )

    def _resolve_sample_device_dtype(
        self,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> tuple[torch.device | None, torch.dtype]:
        """Resolve latent sampling device and dtype from the model state."""
        resolved_device = device
        resolved_dtype = dtype

        for tensor in list(self.parameters()) + list(self.buffers()):
            if resolved_device is None:
                resolved_device = tensor.device
            if resolved_dtype is None and tensor.dtype.is_floating_point:
                resolved_dtype = tensor.dtype
            if resolved_device is not None and resolved_dtype is not None:
                break

        if resolved_dtype is None:
            resolved_dtype = torch.get_default_dtype()
        return resolved_device, resolved_dtype


def _validate_layers(layers: Sequence[QuantumLayer]) -> list[QuantumLayer]:
    """Return validated generator heads.

    Parameters
    ----------
    layers : Sequence[merlin.algorithms.layer.QuantumLayer]
        Candidate quantum generator heads.

    Returns
    -------
    list[merlin.algorithms.layer.QuantumLayer]
        Validated list of layers.

    Raises
    ------
    TypeError
        If ``layers`` is not a sequence of
        :class:`~merlin.algorithms.layer.QuantumLayer` objects.
    ValueError
        If the sequence is empty, if input sizes are inconsistent, or if a layer
        uses amplitude outputs.
    """
    if isinstance(layers, (str, bytes)) or not isinstance(layers, Sequence):
        raise TypeError("layers must be a non-empty sequence of QuantumLayer objects.")
    validated_layers = list(layers)
    if not validated_layers:
        raise ValueError("layers must contain at least one QuantumLayer.")
    for index, layer in enumerate(validated_layers):
        if not isinstance(layer, QuantumLayer):
            raise TypeError(
                f"layers[{index}] must be a QuantumLayer, got {type(layer)}."
            )
        if (
            _resolve_measurement_kind(layer.measurement_strategy)
            is MeasurementKind.AMPLITUDES
        ):
            raise ValueError(
                "PhotonicGenerator does not support amplitude-output layers; "
                f"layers[{index}] uses MeasurementStrategy.amplitudes()."
            )

    latent_dim = validated_layers[0].input_size
    for index, layer in enumerate(validated_layers[1:], start=1):
        if layer.input_size != latent_dim:
            raise ValueError(
                "All PhotonicGenerator layers must have the same input_size; "
                f"layers[0] has {latent_dim}, layers[{index}] has {layer.input_size}."
            )
    return validated_layers


def _flatten_tensor_outputs(outputs: tuple[GeneratorOutput, ...]) -> list[torch.Tensor]:
    """Flatten tensor outputs after their batch dimension."""
    if not outputs:
        raise ValueError("measurements.outputs must not be empty.")

    flattened: list[torch.Tensor] = []
    batch_size: int | None = None
    for index, output in enumerate(outputs):
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                "VectorAdapter and ImageAdapter require tensor measurement "
                f"outputs, but outputs[{index}] has type {type(output)}."
            )
        if output.dim() < 2:
            raise ValueError(
                "Measurement tensor outputs must include batch and feature "
                f"dimensions, but outputs[{index}] has shape {tuple(output.shape)}."
            )
        if batch_size is None:
            batch_size = output.shape[0]
        elif output.shape[0] != batch_size:
            raise ValueError(
                "All measurement tensor outputs must share the same batch size; "
                f"expected {batch_size}, got {output.shape[0]} at outputs[{index}]."
            )
        flattened.append(output.reshape(output.shape[0], -1))

    return flattened


def _center_crop_or_pad(x: torch.Tensor, size: int) -> torch.Tensor:
    """Center-crop or zero-pad a batched matrix to a target feature width."""
    current_size = x.shape[1]
    if current_size == size:
        return x
    if current_size > size:
        left = (current_size - size) // 2
        return x[:, left : left + size]

    left = (size - current_size) // 2
    right = size - current_size - left
    return F.pad(x, (left, right))


def _normalize_image_shape(
    shape: tuple[int, int] | tuple[int, int, int],
) -> tuple[int, int, int]:
    """Normalize image shape to ``(channels, height, width)``."""
    if type(shape) is not tuple:
        raise TypeError("shape must be a tuple.")
    if len(shape) == 2:
        height, width = shape
        normalized = (1, height, width)
    elif len(shape) == 3:
        normalized = shape
    else:
        raise ValueError("shape must have length 2 or 3.")

    if any(type(dim) is not int for dim in normalized):
        raise TypeError("shape dimensions must have int type.")
    if any(dim <= 0 for dim in normalized):
        raise ValueError("shape dimensions must be positive.")
    return normalized
