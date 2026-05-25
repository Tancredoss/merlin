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

"""Grouping policies."""

from collections.abc import Sequence
from numbers import Integral

import torch
import torch.nn as nn
import torch.nn.functional as F


class LexGrouping(nn.Module):
    """
    Maps tensor to a lexical grouping of its components.

    This mapper groups consecutive elements of the input tensor into equal-sized buckets and sums them to
    produce the output. If the input size is not evenly divisible by the output size, padding is applied.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
    ):
        """
        Initialize the converter from input tensor to a lexical grouping of its elements.

        Parameters
        ----------
        input_size : int
            Size of the input tensor.
        output_size : int
            Desired size of the output tensor.
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        """
        Map the input tensor to the desired output_size utilizing lexical grouping.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, input_size)`` or
            ``(input_size,)``.

        Returns
        -------
        torch.Tensor
            Grouped tensor of shape ``(batch_size, output_size)`` or
            ``(output_size,)``.

        Raises
        ------
        ValueError
            If the last dimension of ``x`` does not match ``input_size``.
        """
        if x.shape[-1] != self.input_size:
            raise ValueError(
                f"Input tensor's last dimension ({x.shape[-1]}) does not correspond to the provided input_size ({self.input_size})"
            )

        pad_size = (
            self.output_size - (self.input_size % self.output_size)
        ) % self.output_size
        if pad_size > 0:
            padded = F.pad(x, (0, pad_size))
        else:
            padded = x

        if x.dim() == 2:
            return padded.reshape(x.shape[0], self.output_size, -1).sum(dim=-1)
        else:
            return padded.reshape(self.output_size, -1).sum(dim=-1)


class ModGrouping(nn.Module):
    """
    Maps tensor to a modulo grouping of its components.

    This mapper groups elements of the input tensor based on their index modulo the output size. Elements
    with the same modulo value are summed together to produce the output.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
    ):
        """
        Initialize the converter from input tensor to a modulo grouping of its elements.

        Parameters
        ----------
        input_size : int
            Size of the input tensor.
        output_size : int
            Desired size of the output tensor.
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        """
        Map the input tensor to the desired output_size utilizing modulo grouping.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, input_size)`` or
            ``(input_size,)``.

        Returns
        -------
        torch.Tensor
            Grouped tensor of shape ``(batch_size, output_size)`` or
            ``(output_size,)``.

        Raises
        ------
        ValueError
            If the last dimension of ``x`` does not match ``input_size``.
        """
        if x.shape[-1] != self.input_size:
            raise ValueError(
                f"Input tensor's last dimension ({x.shape[-1]}) does not correspond to the provided input_size ({self.input_size})"
            )

        if self.output_size > self.input_size:
            if x.dim() == 2:
                pad_size = self.output_size - self.input_size
                padded = F.pad(x, (0, pad_size))
                return padded
            else:
                pad_size = self.output_size - self.input_size
                padded = F.pad(x, (0, pad_size))
                return padded

        indices = torch.arange(self.input_size, device=x.device)
        group_indices = indices % self.output_size

        if x.dim() == 2:
            batch_size = x.shape[0]
            result = torch.zeros(
                batch_size,
                self.output_size,
                device=x.device,
                dtype=x.dtype,
            )
            result.index_add_(1, group_indices, x)
            return result
        else:
            result = torch.zeros(
                self.output_size,
                device=x.device,
                dtype=x.dtype,
            )
            result.index_add_(0, group_indices, x)
            return result


class OccupancyGrouping(nn.Module):
    """Group probability columns by effective output occupancy keys.

    ``OccupancyGrouping`` is a key-aware grouping policy. Unlike
    :class:`LexGrouping` and :class:`ModGrouping`, its output size is inferred
    from the output keys it is bound to. Columns whose keys resolve to the same
    occupancy key are summed into the same output bin. Output bins are ordered
    lexicographically by grouped key so the result is deterministic and does not
    depend on simulator column order.

    Parameters
    ----------
    output_keys : Sequence[Sequence[int]] | None
        Effective output keys whose order matches the input probability tensor.
        If omitted, call :meth:`bind_output_keys` before using the grouping.
        Default is ``None``.
    max_count_per_mode : int | None
        Maximum allowed count in each mode. Keys with a larger count are
        dropped. If omitted, all keys are retained. Default is ``None``.
    renormalize : bool
        Whether to renormalize by the kept probability mass when keys are
        dropped. Default is ``True``.

    Raises
    ------
    TypeError
        If ``max_count_per_mode`` is not an integer or ``None``, if
        ``renormalize`` is not a bool, or if output keys contain non-integer
        entries.
    ValueError
        If ``max_count_per_mode`` is negative, if output keys are empty, if
        output keys have inconsistent lengths, if key entries are negative, or
        if filtering drops every key.
    """

    input_size: int | None
    output_size: int
    output_keys: tuple[tuple[int, ...], ...]

    def __init__(
        self,
        output_keys: Sequence[Sequence[int]] | None = None,
        *,
        max_count_per_mode: int | None = None,
        renormalize: bool = True,
    ) -> None:
        """Initialize an occupancy grouping.

        Parameters
        ----------
        output_keys : Sequence[Sequence[int]] | None
            Effective output keys whose order matches the input probability
            tensor. If omitted, call :meth:`bind_output_keys` before use.
            Default is ``None``.
        max_count_per_mode : int | None
            Maximum allowed count in each mode. Keys with a larger count are
            dropped. If omitted, all keys are retained. Default is ``None``.
        renormalize : bool
            Whether to renormalize by the kept probability mass when keys are
            dropped. Default is ``True``.

        Raises
        ------
        TypeError
            If ``max_count_per_mode`` is not an integer or ``None``, if
            ``renormalize`` is not a bool, or if output keys contain
            non-integer entries.
        ValueError
            If ``max_count_per_mode`` is negative, if output keys are empty, if
            output keys have inconsistent lengths, if key entries are negative,
            or if filtering drops every key.
        """
        super().__init__()
        if max_count_per_mode is not None:
            if not isinstance(max_count_per_mode, Integral):
                raise TypeError("max_count_per_mode must be an int or None.")
            if max_count_per_mode < 0:
                raise ValueError("max_count_per_mode must be non-negative.")
            max_count_per_mode = int(max_count_per_mode)
        if type(renormalize) is not bool:
            raise TypeError("renormalize must be a bool.")

        self.max_count_per_mode = max_count_per_mode
        self.renormalize = renormalize
        self.input_size = None
        self.output_size = 0
        self.output_keys = ()
        self._is_bound = False
        self._drops_keys = False
        self.register_buffer(
            "_kept_indices", torch.empty(0, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_group_indices", torch.empty(0, dtype=torch.long), persistent=False
        )

        if output_keys is not None:
            self._bind_in_place(output_keys)

    def bind_output_keys(
        self, output_keys: Sequence[Sequence[int]]
    ) -> "OccupancyGrouping":
        """Return a bound grouping for the provided output keys.

        Parameters
        ----------
        output_keys : Sequence[Sequence[int]]
            Effective output keys whose order matches the input probability
            tensor.

        Returns
        -------
        OccupancyGrouping
            New grouping instance bound to ``output_keys``.

        Raises
        ------
        TypeError
            If output keys contain non-integer entries.
        ValueError
            If output keys are empty, if their lengths are inconsistent, if key
            entries are negative, or if filtering drops every key.
        """
        return type(self)(
            output_keys,
            max_count_per_mode=self.max_count_per_mode,
            renormalize=self.renormalize,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Group a probability tensor by bound occupancy keys.

        Parameters
        ----------
        x : torch.Tensor
            Probability tensor whose last dimension matches the bound
            ``input_size``.

        Returns
        -------
        torch.Tensor
            Tensor with the same leading dimensions as ``x`` and final
            dimension ``output_size``.

        Raises
        ------
        RuntimeError
            If the grouping has not been bound to output keys.
        ValueError
            If the last dimension of ``x`` does not match the bound input size.
        """
        if not self._is_bound or self.input_size is None:
            raise RuntimeError(
                "OccupancyGrouping must be bound to output keys before use."
            )
        if x.shape[-1] != self.input_size:
            raise ValueError(
                f"Input tensor's last dimension ({x.shape[-1]}) does not correspond "
                f"to the bound input_size ({self.input_size})"
            )

        original_shape = x.shape
        if x.dim() == 1:
            matrix = x.unsqueeze(0)
        else:
            matrix = x.reshape(-1, self.input_size)

        kept_indices = self._kept_indices.to(device=x.device)
        group_indices = self._group_indices.to(device=x.device)
        kept_values = matrix.index_select(1, kept_indices)
        grouped = torch.zeros(
            matrix.shape[0],
            self.output_size,
            device=x.device,
            dtype=x.dtype,
        )
        grouped.index_add_(1, group_indices, kept_values)

        if self.renormalize and self._drops_keys:
            kept_mass = kept_values.sum(dim=1, keepdim=True)
            if not torch.is_floating_point(grouped):
                raise TypeError(
                    "OccupancyGrouping renormalization requires a floating-point tensor."
                )
            safe_mass = kept_mass.clamp_min(torch.finfo(grouped.dtype).eps)
            grouped = torch.where(kept_mass > 0, grouped / safe_mass, grouped)

        if x.dim() == 1:
            return grouped.squeeze(0)
        return grouped.reshape(*original_shape[:-1], self.output_size)

    def _bind_in_place(self, output_keys: Sequence[Sequence[int]]) -> None:
        normalized_keys = _normalize_occupancy_keys(output_keys)
        kept_columns: list[int] = []
        kept_keys: list[tuple[int, ...]] = []

        for index, key in enumerate(normalized_keys):
            if self.max_count_per_mode is not None and any(
                count > self.max_count_per_mode for count in key
            ):
                continue
            kept_columns.append(index)
            kept_keys.append(key)

        if not kept_keys:
            raise ValueError("OccupancyGrouping must retain at least one output key.")

        grouped_keys = tuple(sorted(set(kept_keys)))
        key_to_group = {key: index for index, key in enumerate(grouped_keys)}
        group_indices = [key_to_group[key] for key in kept_keys]

        self.input_size = len(normalized_keys)
        self.output_size = len(grouped_keys)
        self.output_keys = grouped_keys
        self._is_bound = True
        self._drops_keys = len(kept_keys) != len(normalized_keys)
        self._kept_indices = torch.tensor(kept_columns, dtype=torch.long)
        self._group_indices = torch.tensor(group_indices, dtype=torch.long)


def _normalize_occupancy_keys(
    output_keys: Sequence[Sequence[int]],
) -> list[tuple[int, ...]]:
    """Validate and normalize occupancy keys to tuples of integers."""
    if len(output_keys) == 0:
        raise ValueError("output_keys must not be empty.")

    normalized: list[tuple[int, ...]] = []
    key_length: int | None = None
    for index, key in enumerate(output_keys):
        if isinstance(key, (str, bytes)) or not isinstance(key, Sequence):
            raise TypeError(f"output_keys[{index}] must be a sequence of integers.")
        normalized_key: list[int] = []
        for value in key:
            if not isinstance(value, Integral):
                raise TypeError("output keys must contain only integer values.")
            int_value = int(value)
            if int_value < 0:
                raise ValueError("output keys must contain non-negative integers.")
            normalized_key.append(int_value)
        if key_length is None:
            key_length = len(normalized_key)
        elif len(normalized_key) != key_length:
            raise ValueError("All output keys must have the same length.")
        normalized.append(tuple(normalized_key))

    return normalized


def _bind_grouping_to_output_keys(
    grouping: nn.Module,
    output_keys: Sequence[Sequence[int]],
) -> nn.Module:
    """Return ``grouping`` bound to ``output_keys`` when it supports binding.

    Parameters
    ----------
    grouping : torch.nn.Module
        Grouping module to bind.
    output_keys : Sequence[Sequence[int]]
        Effective output keys whose order matches the measured probability
        tensor.

    Returns
    -------
    torch.nn.Module
        Bound grouping for key-aware groupings, otherwise the original grouping.
    """
    bind_output_keys = getattr(grouping, "bind_output_keys", None)
    if bind_output_keys is None:
        return grouping
    return bind_output_keys(output_keys)
