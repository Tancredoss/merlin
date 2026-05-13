from functools import reduce
from itertools import combinations
from _collections_abc import Callable
import warnings
import torch
from merlin.algorithms.layer_utils import NoiseGroups
from .slos_torchscript import build_slos_distribution_computegraph as build_slos_graph
from .slos_torchscript import SLOSComputeGraph
from merlin.core.computation_space import ComputationSpace
from merlin.utils.combinadics import Combinadics
from torch import Tensor
from merlin.utils.deprecations import raise_no_bunching_deprecated
import os


class NoisySLOSComputeGraph:
    """
    Equivalent to merlin.pcvl_pytorch.SLOSGraph but with partial
    distinguishability using the Orthogonal Bad Bits model.

    Args:
        input_state (list): Input state into circuit.
        indistinguishability (float).

    >>> beamsplitter = 1 / math.sqrt(2) * torch.tensor([[1., 1j], [1j, 1.]])
    >>> noisy_slos_graph = NoisySLOSComputeGraph(indistinguishability=0.0)
    >>>
    >>> keys, probs = noisy_slos_graph.compute_probs(beamsplitter, [1, 1])
    >>> print(keys, probs)
    [(2, 0), (1, 1), (0, 2)] tensor([[0.3750, 0.2500, 0.3750]])
    """

    def __init__(
        self,
        noise_groups: NoiseGroups | None,
        m,
        n_photons,
        computation_space: ComputationSpace = ComputationSpace.UNBUNCHED,
        keep_keys: bool = True,
        device=None,  # Optional device parameter
        dtype: torch.dtype = torch.float,  # Optional dtype parameter
    ):
        if noise_groups is None:
            raise RuntimeError(
                f"The NoisyComputationProcess should only be used if there is a indistinguishability factor that is not 1.0."
            )
        if noise_groups.source is None:
            raise RuntimeError(
                f"The NoisyComputationProcess should only be used if there is a indistinguishability factor that is not 1.0."
            )

        self.indistinguishability = noise_groups.source.get(
            "indistinguishability", None
        )
        if self.indistinguishability is None:
            raise RuntimeError(
                f"The NoisyComputationProcess should only be used if there is a indistinguishability factor that is not 1.0."
            )

        self.g2_distinguishable = noise_groups.source.get("g2_distinguishable", None)
        self._slos_graph_per_input = {}

        self.m = m
        self.n_photons = n_photons
        if computation_space is ComputationSpace.DUAL_RAIL:
            if m % 2 != 0:
                raise ValueError("dual_rail compute space requires even m")
            if n_photons != m // 2:
                raise ValueError("dual_rail compute space requires n_photons = m // 2")

        self.computation_space = computation_space
        # TODO Change with post-selection if it applies
        if not self.computation_space == ComputationSpace.FOCK:
            warnings.warn(
                "Noisy SLOS simulations currently use ComputationSpace.FOCK. "
                "Other computation spaces are not yet supported for noise models.",
                UserWarning,
                stacklevel=2,
            )
            self.computation_space = ComputationSpace.FOCK

        self.keep_keys = keep_keys
        self.device = device
        self.device = device
        self.dtype = dtype

    def compute_probs(self, unitary, input_state: list):
        input_state = tuple(input_state)

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

        keys, probs = slos_graph.compute_probs(unitary)

        if self.keep_keys:
            return keys, probs
        return probs


class _InputStateNoisySLOSComputeGraph:
    def __init__(
        self,
        input_state: list,
        indistinguishability: float,
        computation_space: ComputationSpace,
        device,
        dtype,
    ):
        self.input_state = input_state
        self.indistinguishability = torch.as_tensor(indistinguishability, dtype=float)
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
            self._new_generate_obb_partition(input_state, num_bad_photons)
            for num_bad_photons in range(0, self.n_photons + 1)
        ]
        # Extract all input states from self._partitions
        self._obb_input_states = self._generate_obb_states(input_state, self.n_photons)

        # All fock states associated with each photon number n
        self._fock_states_per_n = {
            i: torch.tensor(Combinadics("fock", n=i, m=self.m).enumerate_states())
            for i in range(1, self.n_photons + 1)
        }

    def compute_probs(self, unitary):
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
        output_keys = self._fock_states_per_n[self.n_photons]
        output_keys = [tuple(row) for row in output_keys.tolist()]

        output_probs = torch.zeros(b, len(output_keys))

        for i, partition in enumerate(self._partitions):
            bit_weight = self._weights[i]

            for cell in partition:
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
                output_probs += bit_weight * convolution

        output_probs = output_probs / output_probs.sum(dim=1).unsqueeze(1)
        return output_keys, output_probs

    @staticmethod
    def _generate_obb_partition(input_state, order):
        """Generates list of cells for a particular partition and OBB
        "order" or number of "bad" photons.
        """
        if order > sum(input_state):
            raise ValueError("OBB order cannot exceed the number of photons")

        # Convert to tensor if not already
        if not isinstance(input_state, Tensor):
            input_state = torch.tensor(list(input_state), dtype=torch.int32)
        else:
            input_state = input_state.int()

        if order == 0:
            return input_state.unsqueeze(0).unsqueeze(0)

        # Find positions of ones
        one_positions = torch.where(input_state == 1)[0]

        # All combinations of ones to remove
        remove_indices = list(combinations(one_positions.tolist(), order))
        remove_indices = torch.tensor(remove_indices, dtype=torch.long)

        n_comb = remove_indices.shape[0]
        input_state_len = input_state.size(0)

        # Base matrix: original vector repeated for each combination
        base = input_state.unsqueeze(0).repeat(n_comb, 1)
        row_indices = torch.arange(n_comb).unsqueeze(1)
        base[row_indices, remove_indices] = 0  # remove chosen ones

        missing = torch.zeros((n_comb, order, input_state_len), dtype=torch.int32)
        rows = torch.arange(n_comb).unsqueeze(1)
        cols = torch.arange(order).unsqueeze(0)
        missing[rows, cols, remove_indices] = 1

        result = torch.cat([base.unsqueeze(1), missing], dim=1)

        # Remove empty states vectors
        if order == torch.sum(input_state).item():
            mask = result.any(dim=2)
            result = result[mask]
            result = result.unsqueeze(0)

        return result

    @staticmethod
    def _new_generate_obb_partition(input_state, order):
        """Generates list of cells for a particular partition and OBB
        "order" or number of "bad" photons.
        """
        if order > sum(input_state):
            raise ValueError("OBB order cannot exceed the number of photons")

        # Convert to tensor if not already
        if not isinstance(input_state, Tensor):
            input_state = torch.tensor(list(input_state), dtype=torch.int32)
        else:
            input_state = input_state.int()

        if order == 0:
            return input_state.unsqueeze(0).unsqueeze(0)

        # Create a 1D tensor with position of each photon
        positions = torch.arange(len(input_state), dtype=torch.long)
        photon_positions = torch.repeat_interleave(positions, input_state)

        # All combinations of photons to remove
        remove_indices = list(combinations(photon_positions.tolist(), order))
        remove_indices = torch.tensor(remove_indices, dtype=torch.long)

        n_comb = remove_indices.shape[0]
        input_state_len = input_state.size(0)

        # Base matrix: original vector repeated for each combination
        base = input_state.unsqueeze(0).repeat(n_comb, 1)
        for i, remove_index in enumerate(remove_indices):
            for j in remove_index:
                base[i, j] = base[i, j] - 1  # remove chosen ones

        # Should work, create the one hot vectors to convolve that were removed in the good state. So there is order one hot states per combination
        missing = torch.zeros((n_comb, order, input_state_len), dtype=torch.int32)
        rows = torch.arange(n_comb).unsqueeze(1)
        cols = torch.arange(order).unsqueeze(0)
        missing[rows, cols, remove_indices] = 1

        result = torch.cat([base.unsqueeze(1), missing], dim=1)

        # Remove empty states vectors
        if order == torch.sum(input_state).item():
            mask = result.any(dim=2)
            result = result[mask]
            result = result.unsqueeze(0)
        return result

    def _generate_obb_states(self, input_state, order):
        """Generates all possible input states for a given OBB order."""
        if not isinstance(input_state, Tensor):
            input_state = torch.tensor(list(input_state), dtype=torch.int32)
        else:
            input_state = input_state.int()

        if order > torch.sum(input_state).item():
            raise ValueError("OBB order cannot exceed the number of photons")

        total_obb_states = input_state.unsqueeze(0)

        for num_bad_photons in range(1, order + 1):
            obb_states = self._new_generate_obb_partition(input_state, num_bad_photons)
            obb_states = obb_states.reshape(-1, obb_states.shape[2])
            total_obb_states = torch.vstack((total_obb_states, obb_states))

        # Remove duplicate rows
        total_obb_states = torch.unique(total_obb_states, dim=0)

        # Sort by decreasing number of photons
        photon_sums = torch.sum(total_obb_states, dim=1)
        sort_indices = torch.argsort(-photon_sums)
        total_obb_states = total_obb_states[sort_indices]

        return total_obb_states


def convolve_distributions(keys: list[Tensor], *probs: Tensor):
    """
    Performs convolution on two probability distributions. Based on
    `perceval.utils.statevector.BSDistribution.list_tensor_product` with
    `merge_modes = True`.

    Args:
        keys: Stack of states
        probs: Input probability distributions.
    Returns:
        Tuple of new keys and new corresponding probabilities. If keys
        are given as Tensor, then a Tensor is returned instead.

    >>> keys1, probs1 = [(1, 0), (0, 1)], torch.tensor([0.5, 0.5])
    >>> keys2, probs2 = [(1, 0)], torch.tensor([1.0])

    >>> print(convolve_distributions([keys1, keys2], probs1, probs2))
    [(2, 0), (1, 1)], tensor([0.5000, 0.5000])
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


def build_noisy_slos_distribution_computegraph(
    m,
    n_photons,
    computation_space: ComputationSpace | None = None,
    no_bunching: bool | None = None,
    keep_keys: bool = True,
    device=None,
    dtype: torch.dtype = torch.float,
    noise_groups: None = None,
) -> NoisySLOSComputeGraph:
    """Construct a reusable SLOS computation graph.

    Parameters
    ----------
    m : int
        Number of modes in the circuit.
    n_photons : int
        Total number of photons injected in the circuit.
    computation_space : ComputationSpace | None
        Logical computation subspace used to build the basis and transitions.
        When omitted, defaults to ``ComputationSpace.UNBUNCHED``.
    no_bunching : bool | None
        Deprecated legacy flag. Use ``computation_space`` instead.
    keep_keys : bool
        Whether to keep the list of mapped Fock states. Default is ``True``.
    device : torch.device | str | None
        Device on which tensors should be allocated.
    dtype : torch.dtype
        Real dtype controlling numerical precision. Default is ``torch.float``.
    index_photons : list[tuple[int, ...]] | None
        Bounds for each photon placement.
        noise_groups : NoiseGroups|None
        The noise groups defined in the creation of the QuantumLayer. Default is None (no noise).

    Returns
    -------
    NoisySLOSComputeGraph
        Pre-built computation graph ready for repeated evaluations.

    """

    if no_bunching is not None:
        raise_no_bunching_deprecated(stacklevel=2)

    if computation_space is None:
        computation_space = ComputationSpace.UNBUNCHED

    compute_graph = NoisySLOSComputeGraph(
        noise_groups,
        m,
        n_photons,
        computation_space,
        keep_keys,
        device,
        dtype,
    )

    # Add save method to the returned object
    def save(path):
        """
        Save the SLOS computation graph to a file.

        Parameters
        ----------
        path : str | os.PathLike[str]
            Destination path.
        """
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Save metadata
        metadata = {
            "noise_groups": noise_groups,
            "m": compute_graph.m,
            "n_photons": compute_graph.n_photons,
            "computation_space": compute_graph.computation_space.value,
            "keep_keys": compute_graph.keep_keys,
            "dtype_str": str(compute_graph.dtype),
        }

    # Attach the save method to the compute_graph
    compute_graph.save = save  # type: ignore[attr-defined]

    return compute_graph


def load_noisy_slos_distribution_computegraph(path):
    """
    Load a previously saved SLOS distribution computation graph.

    Parameters
    ----------
    path : str | os.PathLike[str]
        Path to the saved computation graph.

    Returns
    -------
    SLOSComputeGraph
        Loaded computation graph ready for computations.

    Examples
    --------
        >>> # Save a computation graph
        >>> graph = build_noisy_slos_distribution_computegraph([1, 1])
        >>> graph.save("hom_graph.pt")
        >>>
        >>> # Later, load the saved graph
        >>> loaded_graph = load_noisy_slos_distribution_computegraph("hom_graph.pt")
        >>>
        >>> # Use the loaded graph
        >>> unitary = torch.tensor([[0.7071, 0.7071], [0.7071, -0.7071]], dtype=torch.cfloat)
        >>> keys, probs = loaded_graph.compute(unitary)
    """
    # Load saved data
    saved_data = torch.load(path)
    metadata = saved_data["metadata"]

    # Create a minimal graph instance
    m = metadata["m"]
    n_photons = metadata["n_photons"]
    computation_space = ComputationSpace.coerce(metadata.get("computation_space"))
    keep_keys = metadata["keep_keys"]
    noise_groups = metadata["noise_groups"]

    # Parse dtype
    dtype_str = metadata.get("dtype_str", "torch.float32")
    if "float16" in dtype_str:
        dtype = torch.float16
    elif "float64" in dtype_str:
        dtype = torch.float64
    else:
        dtype = torch.float32

    # Create basic graph (without output_map_func for now)
    graph = NoisySLOSComputeGraph(
        noise_groups, m, n_photons, None, computation_space, keep_keys, dtype=dtype
    )

    # Restore mapping information if it was used
    if metadata.get("has_output_map_func", False):
        graph.mapped_indices = saved_data["mapped_indices"]
        graph.total_mapped_keys = saved_data["total_mapped_keys"]
        graph.target_indices = saved_data["target_indices"]

    # Recreate the TorchScript modules
    graph._create_torchscript_modules()

    # Add save method to the loaded graph
    graph.save = lambda p: torch.save(saved_data, p)

    return graph
