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

"""Sparse tensor construction helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

_SPARSE_POLICY_INITIALIZED = False


def ensure_sparse_invariant_policy() -> None:
    """Set sparse invariant policy explicitly once to avoid implicit-warning mode."""
    global _SPARSE_POLICY_INITIALIZED
    if _SPARSE_POLICY_INITIALIZED:
        return
    sparse_ns = getattr(torch, "sparse", None)
    checker = getattr(sparse_ns, "check_sparse_tensor_invariants", None)
    if checker is not None and hasattr(checker, "disable"):
        checker.disable()
    _SPARSE_POLICY_INITIALIZED = True


def _ensure_sparse_invariant_policy() -> None:
    ensure_sparse_invariant_policy()


def sparse_coo_tensor(
    indices: torch.Tensor,
    values: torch.Tensor,
    size: Sequence[int] | torch.Size,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    requires_grad: bool = False,
    is_coalesced: bool | None = None,
) -> torch.Tensor:
    """Build a COO sparse tensor with explicit invariant-check policy.

    PyTorch >=2.11 may emit a UserWarning if sparse invariant checks are left
    implicit. This helper forces an explicit policy while preserving behavior on
    older versions that do not support the keyword.
    """
    ensure_sparse_invariant_policy()
    normalized_size = tuple(int(dim) for dim in size)

    target_device = (
        torch.device(device)
        if device is not None
        else values.device
        if values.device.type != "cpu" or indices.device.type == "cpu"
        else indices.device
    )

    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device": target_device,
        "requires_grad": requires_grad,
        "is_coalesced": is_coalesced,
    }
    try:
        return torch.sparse_coo_tensor(
            indices,
            values,
            normalized_size,
            check_invariants=False,
            **kwargs,
        )
    except TypeError:
        # Backward compatibility for torch versions without check_invariants.
        return torch.sparse_coo_tensor(indices, values, normalized_size, **kwargs)
