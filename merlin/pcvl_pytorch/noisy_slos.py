"""Noisy SLOS probability graphs for source-indistinguishability models.

This module implements a probability-only simulation backend for source noise
in Merlin's SLOS pipeline. The main entry point,
``NoisySLOSComputeGraph``, caches one noisy subgraph per input Fock state in
``_slos_graph_per_input`` and reuses those cached subgraphs across repeated
evaluations.

The implementation follows the Orthogonal Bad Bits model: each input state is
expanded into partitions of fully indistinguishable and distinguishable photon
subsets, and the corresponding probability distributions are convolved back
together to obtain the final noisy output distribution.
"""

import warnings
from collections.abc import Sequence
from functools import reduce
from itertools import combinations

import torch
from torch import Tensor

from merlin.algorithms.layer_utils import NoiseGroups
from merlin.core.computation_space import ComputationSpace
from merlin.utils.combinadics import Combinadics
from merlin.utils.dtypes import resolve_float_complex


class NoisySLOSComputeGraph:
    """Probability-only SLOS graph with source noise.

    The graph caches one ``_InputStateNoisySLOSComputeGraph`` per input
    Fock state. Each cached graph expands that input according to the
    Orthogonal Bad Bits model and returns an output probability distribution in
    Fock space.

    Parameters
    ----------
    noise_groups : NoiseGroups | None
        Noise configuration extracted from the layer or experiment. Source noise
        must be present.
    m : int
        Number of optical modes.
    n_photons : int
        Total photon number represented by the graph.
    computation_space : ComputationSpace
        Requested computation space. Source-noise simulations currently operate
        in Fock space only. Default is ``ComputationSpace.FOCK``.
    keep_keys : bool
        If True, return output basis keys together with probabilities. Default
        is True.
    device : str | torch.device | None
        Target device for cached tensors and subgraphs. Default is None.
    dtype : torch.dtype
        Real dtype used by the probability graph. Default is ``torch.float``.

    Raises
    ------
    RuntimeError
        If ``noise_groups`` is missing or does not contain source noise.
    """

    def __init__(
        self,
        noise_groups: NoiseGroups | None,
        m: int,
        n_photons: int,
        computation_space: ComputationSpace = ComputationSpace.FOCK,
        keep_keys: bool = True,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float,
    ) -> None:
        if noise_groups is None:
            raise RuntimeError(
                "The NoisySLOSComputeGraph should only be used if there is source noise in the circuit."
            )
        if noise_groups.source is None:
            raise RuntimeError(
                "The NoisySLOSComputeGraph should only be used if there is source noise in the circuit."
            )

        self.indistinguishability = noise_groups.source.get("indistinguishability", 1.0)

        self.g2_distinguishable = noise_groups.source.get("g2_distinguishable", None)
        self._slos_graph_per_input: dict[
            tuple[int, ...], _InputStateNoisySLOSComputeGraph
        ] = {}

        self.m = m
        self.n_photons = n_photons
        self.computation_space = computation_space
        # TODO Change with post-selection if it applies
        if not self.computation_space == ComputationSpace.FOCK:
            warnings.warn(
                "Noisy simulations with source noise currently use ComputationSpace.FOCK. Other computation spaces are not yet supported for noise models.",
                UserWarning,
                stacklevel=2,
            )
            self.computation_space = ComputationSpace.FOCK

        self.keep_keys = keep_keys
        self.device = device
        self.device = device
        self.dtype = dtype
        self.cdtype = resolve_float_complex(dtype)[1]

        self.mapped_keys = [
            tuple(state)
            for state in Combinadics(
                self.computation_space.casefold(), n=self.n_photons, m=self.m
            ).enumerate_states()
        ]

    def compute_probs(
        self,
        unitary: torch.Tensor,
        input_state: list[int] | tuple[int, ...],
    ) -> tuple[list[tuple[int, ...]], torch.Tensor] | torch.Tensor:
        """Compute noisy output probabilities for one input Fock state.

        Parameters
        ----------
        unitary : torch.Tensor
            Circuit unitary with shape ``[m, m]`` or batched shape
            ``[batch_size, m, m]``. Its dtype must match the complex dtype
            associated with ``self.dtype``.
        input_state : list[int] | tuple[int, ...]
            Input Fock occupation numbers.

        Returns
        -------
        tuple[list[tuple[int, ...]], torch.Tensor] | torch.Tensor
            If ``keep_keys`` is True, returns the Fock output keys and a tensor
            of probabilities with shape ``[batch_size, n_output_states]``.
            Otherwise returns the probability tensor directly.

        Raises
        ------
        ValueError
            If the unitary shape is invalid, the dtype is incompatible, or the
            input state contains negative occupations or no photons.
        """

        if len(unitary.shape) == 2:
            unitary = unitary.unsqueeze(0)  # Add batch dimension [1 x m x m]
        else:
            pass

        batch_size, m, m2 = unitary.shape
        if m != m2 or m != self.m:
            raise ValueError(
                f"Unitary matrix must be square with dimension {self.m}x{self.m}"
            )

        if unitary.dtype != self.cdtype:
            # Raise an error instead of just warning and converting
            raise ValueError(
                f"Unitary dtype {unitary.dtype} doesn't match the expected complex dtype {self.cdtype} "
                f"for the graph built with dtype {self.dtype}. Please provide a unitary with the correct dtype "
                f"or rebuild the graph with a compatible dtype."
            )

        input_state = tuple(input_state)
        if any(n < 0 for n in input_state) or sum(input_state) == 0:
            raise ValueError("Photon numbers cannot be negative or all zeros")

        if input_state not in self._slos_graph_per_input:
            slos_graph = _InputStateNoisySLOSComputeGraph(
                input_state,
                self.indistinguishability,
                self.computation_space,
                self.device,
                self.dtype,
            )
            self.computation_space = slos_graph.computation_space
            self._slos_graph_per_input[input_state] = slos_graph
        else:
            slos_graph = self._slos_graph_per_input[input_state]

        output = torch.empty(
            (batch_size, len(self.mapped_keys)),
            dtype=self.dtype,
            device=unitary.device,
        )
        for i in range(batch_size):
            keys, probs = slos_graph.compute_probs(unitary[i])
            output[i] = probs.squeeze(0)

        if self.keep_keys:
            return keys, output
        return output

    def to(self, device: str | torch.device) -> "NoisySLOSComputeGraph":
        """Move cached tensors and subgraphs to a specific device.

        Parameters
        ----------
        device : str | torch.device
            Target device.

        Returns
        -------
        NoisySLOSComputeGraph
            The graph instance moved to ``device``.

        Raises
        ------
        TypeError
            If ``device`` is neither a string nor a ``torch.device``.
        """
        if isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            raise TypeError(
                f"Expected a string or torch.device, but got {type(device).__name__}"
            )

        for slos_graph in self._slos_graph_per_input.values():
            slos_graph.device = self.device
            slos_graph._obb_input_states = slos_graph._obb_input_states.to(self.device)
            slos_graph._weights = [
                weight.to(self.device) for weight in slos_graph._weights
            ]
            slos_graph._partitions = [
                [partition[0].to(self.device), partition[1].to(self.device)]
                for partition in slos_graph._partitions
            ]
            slos_graph._fock_states_per_n = {
                n: states.to(self.device)
                for n, states in slos_graph._fock_states_per_n.items()
            }

            for graph in slos_graph._slos_graphs:
                graph.to(self.device)

        return self


class _InputStateNoisySLOSComputeGraph:
    """Noisy SLOS graph specialized to one fixed input Fock state.

    This helper precomputes the Orthogonal Bad Bits partitions and the SLOS
    subgraphs needed to evaluate all distinguishability sectors derived from
    one input state.
    """

    def __init__(
        self,
        input_state: list[int] | tuple[int, ...],
        indistinguishability: float,
        computation_space: ComputationSpace = ComputationSpace.UNBUNCHED,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float,
    ) -> None:
        """Initialize the cached noisy graph for one input state.

        Parameters
        ----------
        input_state : list[int] | tuple[int, ...]
            Fixed input Fock state for which all noisy partitions are built.
        indistinguishability : float
            Source indistinguishability parameter in the interval ``[0, 1]``.
        computation_space : ComputationSpace
            Requested computation space for the SLOS subgraphs. Default is
            ``ComputationSpace.UNBUNCHED``.
        device : str | torch.device | None
            Target device for cached tensors. Default is None.
        dtype : torch.dtype
            Real dtype for internal probability tensors. Default is
            ``torch.float``.

        Raises
        ------
        ValueError
            If ``indistinguishability`` lies outside ``[0, 1]``.
        """
        from .slos_torchscript import (
            build_slos_distribution_computegraph as build_slos_graph,
        )

        self.input_state = input_state
        self.indistinguishability = torch.as_tensor(
            indistinguishability, dtype=torch.float64
        )
        self.m = len(input_state)
        self.n_photons = sum(input_state)
        self.computation_space = computation_space
        if (computation_space is not ComputationSpace.FOCK) and (max(input_state) > 1):
            self.computation_space = ComputationSpace.FOCK

        if indistinguishability < 0 or indistinguishability > 1:
            raise ValueError("Indistinguishability must be in range (0, 1).")

        self.device = device
        self.dtype = dtype

        self._slos_graphs = [
            build_slos_graph(
                self.m,
                n_i,
                computation_space=computation_space,
                device=device,
                dtype=dtype,
            )
            for n_i in range(1, self.n_photons + 1)
        ]

        # Weights of good & bad bits respectively
        self.g = torch.sqrt(self.indistinguishability)
        self.b = 1 - self.g

        # Weights associated with each cell in each partition
        self._weights = [
            self.g ** (self.n_photons - i) * self.b**i
            for i in range(self.n_photons + 1)
        ]

        # List of partitions of cells of states.
        self._partitions = [
            self._generate_obb_partition(input_state, num_bad_photons, device=device)
            for num_bad_photons in range(0, self.n_photons + 1)
        ]
        # Extract all input states from self._partitions
        self._obb_input_states = self._generate_obb_states(
            input_state, self.n_photons, device=device
        )

        # All fock states associated with each photon number n
        self._fock_states_per_n = {
            i: torch.tensor(Combinadics("fock", n=i, m=self.m).enumerate_states())
            for i in range(1, self.n_photons + 1)
        }

    def compute_probs(
        self, unitary: torch.Tensor
    ) -> tuple[list[tuple[int, ...]], torch.Tensor]:
        """Compute noisy probabilities for the cached input state.

        Parameters
        ----------
        unitary : torch.Tensor
            Circuit unitary with shape ``[m, m]`` or batched shape
            ``[batch_size, m, m]``.

        Returns
        -------
        tuple[list[tuple[int, ...]], torch.Tensor]
            Output Fock keys and the corresponding probabilities with shape
            ``[batch_size, n_output_states]``.
        """
        if unitary.size(0) == unitary.size(1) and unitary.ndim == 2:
            unitary = unitary.unsqueeze(0)

        probs_per_obb_state = {}
        for state in self._obb_input_states:
            key = tuple(state.tolist())
            n = sum(key)

            _, probs = self._slos_graphs[n - 1].compute_probs(unitary, state)

            if probs.ndim == 1:
                probs = probs.unsqueeze(0)

            probs_per_obb_state[key] = probs

        self._probs_per_obb_state = probs_per_obb_state

        b = len(unitary)
        output_keys_tensor = self._fock_states_per_n[self.n_photons]
        output_keys = [tuple(row) for row in output_keys_tensor.tolist()]

        output_probs = torch.zeros(b, len(output_keys))

        for i, partition in enumerate(self._partitions):
            bit_weight = self._weights[i]

            for cell, count in zip(partition[0], partition[1], strict=True):
                cell_distributions = [
                    probs_per_obb_state[tuple(state.tolist())] for state in cell
                ]
                fock_states = [
                    self._fock_states_per_n[int(sum(state))] for state in cell
                ]
                _, convolution = convolve_distributions(
                    fock_states,
                    *cell_distributions,
                )
                output_probs += bit_weight * convolution * count.item()

        output_probs = output_probs / output_probs.sum(dim=1).unsqueeze(1)
        return output_keys, output_probs

    @staticmethod
    def _generate_obb_partition(
        input_state: list[int] | tuple[int, ...] | torch.Tensor,
        order: int,
        device: str | torch.device | None = None,
    ) -> list[torch.Tensor]:
        """Generate one Orthogonal Bad Bits partition.

        Parameters
        ----------
        input_state : list[int] | tuple[int, ...] | torch.Tensor
            Input Fock state to partition.
        order : int
            Number of distinguishable, or "bad", photons to extract from the
            input state.
        device : str | torch.device | None
            Device on which the returned tensors are allocated. Default is
            None.

        Returns
        -------
        list[torch.Tensor]
            Two-element list containing the partition cells and their
            multiplicities. The first tensor has shape
            ``[n_cells, cell_size, m]`` and the second tensor stores the count
            for each cell.

        Raises
        ------
        ValueError
            If ``order`` exceeds the total number of photons.
        """
        total_photons = (
            int(torch.sum(input_state).item())
            if isinstance(input_state, Tensor)
            else sum(input_state)
        )
        if order > total_photons:
            raise ValueError("OBB order cannot exceed the number of photons")

        # Convert to tensor if not already
        if not isinstance(input_state, Tensor):
            input_state_tensor = torch.tensor(list(input_state), dtype=torch.int32)
        else:
            input_state_tensor = input_state.int()

        if order == 0:
            counts = torch.tensor([1], dtype=torch.int64, device=device)
            return [input_state_tensor.unsqueeze(0).unsqueeze(0).to(device), counts]

        # Create a 1D tensor with position of each photon
        positions = torch.arange(len(input_state_tensor), dtype=torch.long).to(device)
        photon_positions = torch.repeat_interleave(positions, input_state_tensor).to(
            device
        )

        # All combinations of photons to remove
        remove_indices = list(combinations(photon_positions.tolist(), order))
        remove_indices = torch.tensor(remove_indices, dtype=torch.long)

        n_comb = remove_indices.shape[0]
        input_state_len = input_state_tensor.size(0)

        # Base matrix: original vector repeated for each combination
        base = input_state_tensor.unsqueeze(0).repeat(n_comb, 1)
        for i, remove_index in enumerate(remove_indices):
            for j in remove_index:
                base[i, j] = base[i, j] - 1  # remove chosen ones

        # Should work, create the one hot vectors to convolve that were removed in the good state. So there is order one hot states per combination
        missing = torch.zeros((n_comb, order, input_state_len), dtype=torch.int32).to(
            device
        )
        rows = torch.arange(n_comb).unsqueeze(1)
        cols = torch.arange(order).unsqueeze(0)
        missing[rows, cols, remove_indices] = 1

        result = torch.cat([base.unsqueeze(1), missing], dim=1)

        # Remove empty states vectors
        if order == torch.sum(input_state_tensor).item():
            mask = result.any(dim=2)
            result = result[mask]
            result = result.unsqueeze(0)

        result, counts = torch.unique(result, return_counts=True, dim=0)
        return [result.to(device), counts.to(device)]

    def _generate_obb_states(
        self,
        input_state: list[int] | tuple[int, ...] | torch.Tensor,
        order: int,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Generate all OBB-derived input states up to a given order.

        Parameters
        ----------
        input_state : list[int] | tuple[int, ...] | torch.Tensor
            Reference input Fock state.
        order : int
            Maximum number of bad photons to include.
        device : str | torch.device | None
            Device on which the returned tensor is allocated. Default is None.

        Returns
        -------
        torch.Tensor
            Tensor of unique OBB states sorted by decreasing photon number.

        Raises
        ------
        ValueError
            If ``order`` exceeds the total number of photons.
        """
        if not isinstance(input_state, Tensor):
            input_state = torch.tensor(list(input_state), dtype=torch.int32)
        else:
            input_state = input_state.int()

        if order > torch.sum(input_state).item():
            raise ValueError("OBB order cannot exceed the number of photons")

        total_obb_states = input_state.unsqueeze(0)

        for num_bad_photons in range(1, order + 1):
            obb_states = self._generate_obb_partition(
                input_state, num_bad_photons, device=device
            )[0]
            obb_states = obb_states.reshape(-1, obb_states.shape[2])
            total_obb_states = torch.vstack((total_obb_states, obb_states))

        # Remove duplicate rows
        total_obb_states = torch.unique(total_obb_states, dim=0).to(device)

        # Sort by decreasing number of photons
        photon_sums = torch.sum(total_obb_states, dim=1)
        sort_indices = torch.argsort(-photon_sums)
        total_obb_states = total_obb_states[sort_indices]

        return total_obb_states


def convolve_distributions(
    keys: Sequence[Tensor | Sequence[tuple[int, ...]]], *probs: Tensor
) -> tuple[Tensor | list[tuple[int, ...]], Tensor]:
    """Convolve one or more probability distributions over Fock states.

    This helper performs the same mode-merging tensor product used by Perceval
    when combining independent distributions over mode occupations.

    Parameters
    ----------
    keys : list[torch.Tensor | list[tuple[int, ...]]]
        Sequence of state lists matching the input distributions.
    *probs : torch.Tensor
        Input probability distributions. Each tensor is either one-dimensional
        or batched on its leading axis.

    Returns
    -------
    tuple[torch.Tensor | list[tuple[int, ...]], torch.Tensor]
        Combined keys and the corresponding convolved probabilities.

    Raises
    ------
    ValueError
        If the number of key sets does not match the number of probability
        tensors.
    """
    if len(probs[0].shape) == 1:
        probs = reduce(lambda acc, x: acc + (x.unsqueeze(0),), probs, ())
        batched_input = False
    else:
        batched_input = True

    num_probs = len(probs)
    num_batches = probs[0].size(0)

    if len(keys) != len(probs):
        raise ValueError(
            f"Invalid probability distribution for different length keys "
            f"({len(keys)}) & probs ({len(probs)})"
        )

    if num_probs == 1:
        return keys[0], probs[0]

    def _cartesian_sum(k1, k2):
        k1 = torch.as_tensor(k1)
        k2 = torch.as_tensor(k2)
        return (k1.unsqueeze(1) + k2.unsqueeze(0)).reshape(-1, k1.shape[1])

    new_keys = reduce(_cartesian_sum, keys)

    # Cartesian product of every pair of probs
    def _cartesian_product(p1, p2):
        output = p1.unsqueeze(-1) * p2.unsqueeze(-2)
        return output.flatten(start_dim=-2)

    # Unsqueeze each input tensor
    probs = reduce(lambda acc, x: acc + (x.unsqueeze(0),), probs, ())

    new_probs = reduce(_cartesian_product, probs).view(num_batches, -1)

    # Remove duplicated keys & sum corresponding probs
    new_keys, inverse_idx = torch.unique(new_keys, dim=0, return_inverse=True)
    inverse_idx = inverse_idx.unsqueeze(0).expand(num_batches, -1)
    new_probs = torch.zeros(
        num_batches, len(new_keys), dtype=new_probs.dtype
    ).scatter_add_(dim=1, index=inverse_idx, src=new_probs)

    # Correct the order of the keys & probs
    new_keys = new_keys.flip(0)
    new_probs = new_probs.flip(1)

    if not batched_input:
        new_probs = new_probs.squeeze(0)

    return new_keys, new_probs
