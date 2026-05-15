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

import itertools
import warnings
from collections.abc import Callable
from typing import cast

import numpy as np
import perceval as pcvl
import torch
from torch import Tensor

from ..builder.circuit_builder import ANGLE_ENCODING_MODE_ERROR, CircuitBuilder
from ..core.computation_space import ComputationSpace
from ..core.state import StatePattern, generate_state
from ..measurement.autodiff import AutoDiffProcess
from ..measurement.detectors import resolve_detectors
from ..measurement.photon_loss import resolve_photon_loss
from ..measurement.strategies import MeasurementStrategy
from ..pcvl_pytorch.locirc_to_tensor import CircuitConverter
from ..utils.deprecations import sanitize_parameters
from ..utils.dtypes import to_torch_dtype
from .layer import QuantumLayer
from .module import MerlinModule


class FeatureMap:
    """Quantum feature map.

    FeatureMap describes how classical data is embedded in a photonic circuit
    for quantum kernel methods.

    ``FidelityKernel`` treats this object as a descriptor. It passes the stored
    experiment, parameter prefixes, input size, dtype, and device to
    ``_CCInvQuantumLayer``, the internal adapter over the :class:`QuantumLayer`
    backend. Legacy unitary-building state remains on ``FeatureMap`` only to
    support deprecated direct calls to :meth:`compute_unitary`.

    Parameters
    ----------
    circuit : pcvl.Circuit | None
        Pre-compiled Perceval circuit used to encode features.
    input_size : int | None
        Dimension of incoming classical data. Required.
    builder : CircuitBuilder | None
        Optional builder used to compile a circuit declaratively.
    experiment : pcvl.Experiment | None
        Optional experiment providing both the circuit and detector
        configuration. Exactly one of ``circuit``, ``builder``, or
        ``experiment`` must be supplied.
    input_parameters : str | list[str] | None
        Parameter prefix(es) that host the classical data.
    trainable_parameters : list[str] | None
        Optional trainable parameter prefixes.
    dtype : str | torch.dtype
        Torch dtype used when constructing the unitary.
    device : torch.device | None
        Torch device on which unitaries are evaluated.
    encoder : Callable[[torch.Tensor], torch.Tensor] | None
        Optional custom encoder used when the raw input shape does not match the
        circuit parameter layout.
    """

    def __init__(
        self,
        circuit: pcvl.Circuit | None = None,
        input_size: int | None = None,
        *,
        builder: CircuitBuilder | None = None,
        experiment: pcvl.Experiment | None = None,
        input_parameters: str | list[str] | None,
        trainable_parameters: list[str] | None = None,
        dtype: str | torch.dtype = torch.float32,
        device: torch.device | None = None,
        encoder: Callable[[Tensor], Tensor] | None = None,  # was: callable | None
    ):
        builder_trainable: list[str] = []
        builder_input: list[str] = []

        self._angle_encoding_specs: dict[str, dict[str, object]] = {}
        self.experiment: pcvl.Experiment | None = None

        # The feature map can be defined from exactly one artefact among circuit, builder, or experiment.
        if sum(x is not None for x in (circuit, builder, experiment)) != 1:
            raise ValueError(
                "Provide exactly one of 'circuit', 'builder', or 'experiment'."
            )

        resolved_circuit: pcvl.Circuit | None = None

        if builder is not None:
            builder_trainable = builder.trainable_parameter_prefixes
            builder_input = builder.input_parameter_prefixes
            self._angle_encoding_specs = builder.angle_encoding_specs
            resolved_circuit = builder.to_pcvl_circuit(pcvl)
            self.experiment = pcvl.Experiment(resolved_circuit)
        elif circuit is not None:
            resolved_circuit = circuit
            self.experiment = pcvl.Experiment(resolved_circuit)
        elif experiment is not None:
            if (
                not experiment.is_unitary
                or experiment.post_select_fn is not None
                or experiment.heralds
            ):
                raise ValueError(
                    "The provided experiment must be unitary, and must not have post-selection or heralding."
                )
            if experiment.min_photons_filter:
                raise ValueError(
                    "The provided experiment must not have a minimum photons filter."
                )
            self.experiment = experiment
            resolved_circuit = experiment.unitary_circuit()
        else:  # pragma: no cover - defensive guard
            raise RuntimeError("Resolved circuit could not be determined.")

        self.circuit = resolved_circuit
        if input_size is None:
            raise TypeError("FeatureMap requires 'input_size' to be specified.")
        self.input_size = input_size
        if trainable_parameters is None:
            trainable_parameters = builder_trainable
        self.trainable_parameters = list(trainable_parameters or [])
        self.dtype = to_torch_dtype(dtype)
        self.device = device or torch.device("cpu")
        self.is_trainable = bool(self.trainable_parameters)
        self._encoder = encoder  # NEW

        if input_parameters is None:
            if builder_input:
                input_parameters = builder_input[0]
            else:
                raise ValueError(
                    "input_parameters must be provided when no input layer is defined in the builder."
                )

        if isinstance(input_parameters, list):
            if len(input_parameters) > 1:
                raise ValueError("Only a single input parameter is allowed.")

            self.input_parameters = input_parameters[0]
        else:
            self.input_parameters = input_parameters

        self._circuit_graph = CircuitConverter(
            self.circuit,
            [self.input_parameters] + self.trainable_parameters,
            dtype=self.dtype,
            device=self.device,
        )
        # Legacy compiler state kept only for deprecated compute_unitary.
        # FidelityKernel does not read this converter or training dictionary.
        self._training_dict: dict[str, torch.nn.Parameter] = {}
        for param_name in self.trainable_parameters:
            param_length = len(self._circuit_graph.spec_mappings[param_name])

            p = torch.rand(param_length, requires_grad=True)
            self._training_dict[param_name] = torch.nn.Parameter(p)

    def _px_len(self) -> int:
        """Return how many angle-encoding slots the deprecated converter expects.

        .. warning:: *Deprecated since version 0.4:*
            This helper belongs to the legacy ``FeatureMap.compute_unitary``
            path. ``FidelityKernel`` uses ``_CCInvQuantumLayer`` over the
            :class:`QuantumLayer` backend and does not rely on this method.

        Returns
        -------
        int
            Number of angle-encoding slots expected by the legacy converter.
        """
        return len(self._circuit_graph.spec_mappings.get(self.input_parameters, []))

    def _subset_sum_expand(self, x: Tensor, k: int) -> Tensor:
        """Expand an input vector into deterministic subset sums.

        .. warning:: *Deprecated since version 0.4:*
            This helper belongs to the legacy ``FeatureMap.compute_unitary``
            path. ``FidelityKernel`` uses ``_CCInvQuantumLayer`` over the
            :class:`QuantumLayer` backend and does not rely on this method.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor expected to be one-dimensional.
        k : int
            Desired number of encoded features to return.

        Returns
        -------
        torch.Tensor
            Encoded tensor of length ``k`` on the configured device and dtype.
        """
        x = x.to(dtype=self.dtype, device=self.device).reshape(-1)
        d = x.shape[0]
        vals: list[Tensor] = []
        # generate sums for subset sizes 1..d
        for r in range(1, d + 1):
            for idxs in itertools.combinations(range(d), r):
                vals.append(x[list(idxs)].sum())
                if len(vals) == k:
                    return torch.stack(vals, dim=0)
        # if fewer than k (shouldn't happen for k <= 2^d-1), pad with zeros
        if len(vals) == 0:
            return torch.zeros(k, dtype=self.dtype, device=self.device)
        pad = k - len(vals)
        return torch.cat(
            [
                torch.stack(vals, dim=0),
                torch.zeros(pad, dtype=self.dtype, device=self.device),
            ],
            dim=0,
        )

    def _encode_x(self, x: Tensor) -> Tensor:
        """Map raw features to the deprecated converter's parameter shape.

        .. warning:: *Deprecated since version 0.4:*
            This helper belongs to the legacy ``FeatureMap.compute_unitary``
            path. ``FidelityKernel`` uses ``_CCInvQuantumLayer`` over the
            :class:`QuantumLayer` backend and does not rely on this method.

        Preference order:

        1. Builder-provided combination metadata (from :class:`CircuitBuilder`).
        2. A user-supplied encoder callable.
        3. The deterministic subset-sum expansion used by legacy feature maps.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor to be embedded.

        Returns
        -------
        torch.Tensor
            Encoded tensor matching the circuit's expected parameter length.
        """
        x = x.to(dtype=self.dtype, device=self.device).reshape(-1)
        px_len = self._px_len()

        spec = self._angle_encoding_specs.get(self.input_parameters)

        if spec:
            encoded = self._encode_with_specs(x, spec)
            if encoded.numel() != px_len:
                raise ValueError(
                    f"Angle encoding produced {encoded.numel()} parameters but circuit expects {px_len}"
                )
            return encoded

        if x.numel() == px_len:
            return x
        if x.numel() < px_len:
            # Try provided encoder if available
            if callable(self._encoder):
                try:
                    encoded = self._encoder(x)
                    # Allow numpy/torch outputs and ensure correct shape/device/dtype
                    if isinstance(encoded, np.ndarray):
                        encoded = torch.from_numpy(encoded)
                    encoded = torch.as_tensor(
                        encoded, dtype=self.dtype, device=self.device
                    ).reshape(-1)
                    if encoded.numel() != px_len:
                        # Fall back if encoder does not match spec
                        return self._subset_sum_expand(x, px_len)
                    return encoded
                except Exception:
                    # Encoder failed; use deterministic subset-sum expansion
                    return self._subset_sum_expand(x, px_len)
            # No encoder provided; series-style expansion
            return self._subset_sum_expand(x, px_len)
        # x longer than needed; truncate
        return x[:px_len]

    def _encode_with_specs(self, x: Tensor, spec: dict[str, object]) -> Tensor:
        """Encode input vector using builder-provided angle encoding metadata.

        .. warning:: *Deprecated since version 0.4:*
            This helper belongs to the legacy ``FeatureMap.compute_unitary``
            path. ``FidelityKernel`` uses ``_CCInvQuantumLayer`` over the
            :class:`QuantumLayer` backend and does not rely on this method.

        Parameters
        ----------
        x : torch.Tensor
            Flattened input feature tensor.
        spec : dict[str, object]
            Metadata describing combinations and scales produced by the builder.

        Returns
        -------
        torch.Tensor
            Encoded tensor obeying the combination rules.
        """
        combos = spec.get("combinations", [])
        scales = spec.get("scales", {})

        if not isinstance(combos, list) or not all(
            isinstance(c, tuple) for c in combos
        ):
            raise ValueError(
                "Invalid angle encoding metadata: 'combinations' must be a list of tuples"
            )

        if not isinstance(scales, dict):
            raise ValueError("Invalid angle encoding metadata: 'scales' must be a dict")

        x_flat = x.to(dtype=self.dtype, device=self.device).reshape(-1)
        encoded_vals: list[Tensor] = []
        feature_dim = x_flat.shape[0]

        for combo in combos:  # type: ignore[assignment]
            indices = list(combo)
            if any(idx >= feature_dim for idx in indices):
                raise ValueError(
                    f"Input feature dimension {feature_dim} insufficient for angle encoding combination {combo}"
                )

            selected = x_flat[indices]
            scale_tensor = torch.tensor(
                [float(scales.get(idx, 1.0)) for idx in indices],
                dtype=self.dtype,
                device=self.device,
            )
            encoded_vals.append((selected * scale_tensor).sum())

        if not encoded_vals:
            return torch.zeros(0, dtype=self.dtype, device=self.device)

        return torch.stack(encoded_vals, dim=0)

    @sanitize_parameters
    def compute_unitary(
        self, x: torch.Tensor | np.ndarray | float, *training_parameters: torch.Tensor
    ) -> torch.Tensor:
        """Generate the circuit unitary after encoding `x` and applying trainables.

        .. warning:: *Deprecated since version 0.4:*
            ``compute_unitary`` is deprecated and will be removed in a future release.
            It uses legacy compiler state stored on ``FeatureMap``. Use
            :class:`FidelityKernel` for kernel computations; ``FidelityKernel``
            uses ``_CCInvQuantumLayer`` over the :class:`QuantumLayer` backend
            and treats ``FeatureMap`` as a descriptor without relying on this
            method.

        Parameters
        ----------
        x : torch.Tensor | numpy.ndarray | float
            Single datapoint to embed; accepts scalars, NumPy arrays, or
            tensors.
        training_parameters : torch.Tensor
            Optional overriding trainable tensors.

        Returns
        -------
        torch.Tensor
            Complex unitary matrix representing the prepared circuit.
        """
        # Normalize input to tensor on correct device/dtype
        if isinstance(x, torch.Tensor):
            x = x.to(dtype=self.dtype, device=self.device)
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(device=self.device, dtype=self.dtype)
        elif isinstance(x, (float, int)):
            # scalar datapoint: only valid if input_size == 1
            x = torch.tensor([x], dtype=self.dtype, device=self.device)
        else:
            raise TypeError(f"Unsupported input type: {type(x)!r}")

        # Encode x to match the circuit's input parameter spec
        x_encoded = self._encode_x(x)

        if not self.is_trainable:
            return self._circuit_graph.to_tensor(x_encoded)

        # Use provided training parameters or fall back to internal ones
        if training_parameters:
            params_to_use: tuple[Tensor, ...] = training_parameters
        else:
            # Cast to a Tensor tuple for mypy; Parameter is a Tensor subtype
            params_to_use = cast(
                tuple[Tensor, ...], tuple(self._training_dict.values())
            )
        return self._circuit_graph.to_tensor(x_encoded, *params_to_use)

    def is_datapoint(self, x: torch.Tensor | np.ndarray | float | int) -> bool:
        """Determine if ``x`` describes one sample or a batch.

        Parameters
        ----------
        x : torch.Tensor | numpy.ndarray | float | int
            Candidate input data.

        Returns
        -------
        bool
            ``True`` when ``x`` corresponds to a single datapoint.
        """
        if isinstance(x, (float, int)):
            if self.input_size == 1:
                return True
            raise ValueError(
                f"Given value shape () does not match data shape {self.input_size}."
            )

        # x is array-like (Tensor or ndarray)
        if isinstance(x, Tensor):
            ndim = x.ndim
            shape = tuple(x.shape)
            num_elements = x.numel()
        else:
            ndim = x.ndim
            shape = tuple(x.shape)
            num_elements = x.size

        error_msg = (
            f"Given value shape {shape} does not match data shape {self.input_size}."
        )
        if num_elements % self.input_size or ndim > 2:
            raise ValueError(error_msg)

        if self.input_size == 1:
            if num_elements == 1 and ndim == 1:
                return True
            if num_elements == 1 and ndim == 2:
                return False
            if ndim > 1:
                return False
        else:
            if ndim == 1 and shape[0] == self.input_size:
                return True
            if ndim == 2:
                return False
        raise ValueError(error_msg)

    @classmethod
    @sanitize_parameters
    def simple(
        cls,
        input_size: int,
        *,
        dtype: str | torch.dtype = torch.float32,
        device: torch.device | None = None,
        angle_encoding_scale: float = 1.0,
        n_modes: int = None,
    ) -> "FeatureMap":
        """Simple factory method to create a FeatureMap with minimal configuration.

        Parameters
        ----------
        input_size : int
            Classical feature dimension. Maximum is 19.
        dtype : str | torch.dtype
            Target dtype for internal tensors.
        device : torch.device | None
            Optional torch device handle.
        angle_encoding_scale : float
            Global scaling applied to angle encoding features. Default is ``1.0``.
        n_modes : int | None
            Number of photonic modes used by the helper circuit. If omitted,
            ``n_modes = input_size + 1``. Maximum is 20.

        Returns
        -------
        FeatureMap
            Configured feature-map instance.
        """
        if n_modes is None:
            n_modes = input_size + 1
        if n_modes < 2:
            raise ValueError(f"The number of modes must be at least 2, got {n_modes}")
        if input_size > 19 or n_modes > 20:
            raise ValueError(
                "Input size too large for the simple layer construction. For large inputs (with larger size than 20), please use the CircuitBuilder. Here is a quick tutorial on how to use it: https://merlinquantum.ai/quickstart/first_quantum_layer.html#circuitbuilder-walkthrough"
            )
        if input_size < 1:
            raise ValueError(f"input_size must be at least 1, got {input_size}")

        if input_size > n_modes:
            raise ValueError(ANGLE_ENCODING_MODE_ERROR)

        builder = CircuitBuilder(n_modes=n_modes)

        # Trainable entangling layer before encoding
        builder.add_entangling_layer(
            trainable=True,
            name="LI_simple",
        )

        # Angle encoding
        builder.add_angle_encoding(
            modes=list(range(int(input_size))),
            name="input",
            subset_combinations=False,
            scale=angle_encoding_scale,
        )

        # Trainable entangling layer after encoding
        builder.add_entangling_layer(trainable=True, name="RI_simple")

        return cls(
            builder=builder,
            input_size=input_size,
            input_parameters=None,
            trainable_parameters=[
                "LI_simple",
                "RI_simple",
            ],
            dtype=dtype,
            device=device,
        )


class KernelCircuitBuilder:
    """Builder for creating photonic quantum kernel circuits.

    This class provides a fluent interface for building quantum kernel circuits
    with various configurations, inspired by the core.layer architecture.
    """

    def __init__(self) -> None:
        self._input_size: int | None = None
        self._n_modes: int | None = None
        self._n_photons: int | None = None
        self._dtype: str | torch.dtype = torch.float32
        self._device: torch.device | None = None
        self._angle_encoding_scale: float = 1.0
        self._trainable: bool = True
        self._trainable_prefix: str = "phi"

    def input_size(self, size: int) -> "KernelCircuitBuilder":
        """Set the input dimensionality."""
        self._input_size = size
        return self

    def n_modes(self, modes: int) -> "KernelCircuitBuilder":
        """Set the number of modes in the circuit."""
        self._n_modes = modes
        return self

    def n_photons(self, photons: int) -> "KernelCircuitBuilder":
        """Set the number of photons."""
        self._n_photons = photons
        return self

    def trainable(
        self,
        enabled: bool = True,
        *,
        prefix: str = "phi",
    ) -> "KernelCircuitBuilder":
        """Enable or disable trainable rotations generated by the helper."""
        self._trainable = enabled
        if enabled:
            self._trainable_prefix = prefix
        return self

    def dtype(self, dtype: str | torch.dtype) -> "KernelCircuitBuilder":
        """Set the data type for computations."""
        self._dtype = dtype
        return self

    def device(self, device: torch.device) -> "KernelCircuitBuilder":
        """Set the computation device."""
        self._device = device
        return self

    def angle_encoding(
        self,
        *,
        scale: float = 1.0,
    ) -> "KernelCircuitBuilder":
        """Configure the angle encoding scale."""
        self._angle_encoding_scale = scale
        return self

    def build_feature_map(self) -> FeatureMap:
        """Build and return a :class:`FeatureMap` instance.

        Returns
        -------
        FeatureMap
            Configured feature map.

        Raises
        ------
        ValueError
            If required parameters are missing.
        """
        if self._input_size is None:
            raise ValueError("Input size must be specified")

        n_modes = self._n_modes or max(self._input_size + 1, 4)

        trainable_params: list[str] | None
        if self._trainable:
            trainable_params = [self._trainable_prefix]
        else:
            trainable_params = None

        builder = CircuitBuilder(n_modes=n_modes)
        builder.add_superpositions(depth=1)

        if self._input_size > n_modes:
            raise ValueError(ANGLE_ENCODING_MODE_ERROR)

        input_modes = list(range(self._input_size))

        builder.add_angle_encoding(
            modes=input_modes,
            name="input",
            scale=self._angle_encoding_scale,
        )

        if self._trainable:
            builder.add_rotations(trainable=True, name=self._trainable_prefix)

        builder.add_superpositions(depth=1)

        return FeatureMap(
            builder=builder,
            input_size=self._input_size,
            input_parameters=None,
            trainable_parameters=trainable_params,
            dtype=self._dtype,
            device=self._device,
        )

    @sanitize_parameters
    def build_fidelity_kernel(
        self,
        input_state: list[int] | None = None,
        *,
        shots: int = 0,
        sampling_method: str = "multinomial",
        computation_space: ComputationSpace | str | None = None,
        force_psd: bool = True,
    ) -> "FidelityKernel":
        """Build and return a :class:`~merlin.algorithms.kernels.FidelityKernel` instance.

        Parameters
        ----------
        input_state : list[int] | None
            Input Fock state. If ``None``, it is generated automatically.
        shots : int
            Number of sampling shots. Default is ``0``.
        sampling_method : str
            Sampling method for pseudo-sampling. Default is
            ``"multinomial"``.
        computation_space : ComputationSpace | str | None
            Logical computation subspace; one of ``{"fock", "unbunched",
            "dual_rail"}``.
        force_psd : bool
            Whether to project to the nearest positive semi-definite matrix.
            Default is ``True``.

        Returns
        -------
        FidelityKernel
            Configured fidelity kernel.
        """
        feature_map = self.build_feature_map()

        # Generate default input state if not provided
        if input_state is None:
            n_modes = self._n_modes or max(self._input_size or 2, 4)
            n_photons = self._n_photons or (self._input_size or 2)
            input_state = list(generate_state(n_modes, n_photons, StatePattern.SPACED))

        return FidelityKernel(
            feature_map=feature_map,
            input_state=input_state,
            shots=shots,
            sampling_method=sampling_method,
            computation_space=computation_space,
            force_psd=force_psd,
            device=self._device,
            dtype=self._dtype,
        )


class _CCInvQuantumLayer(QuantumLayer):
    """Internal ``QuantumLayer`` subclass used as the computation backend for ``FidelityKernel``.

    The name ``CCInv`` reflects two aspects of the computation: "CC" for
    ``CircuitConverter``, the component used to build the unitary tensor from
    the Perceval circuit; and "Inv" for the conjugate transpose (inverse)
    applied to the second operand when forming the kernel unitary
    ``U(x1) @ U†(x2)``.

    This layer owns encoding, unitary computation, SLOS simulation, photon
    loss, and detector transforms for the fidelity quantum kernel.
    ``FidelityKernel`` constructs one instance and delegates all circuit-level
    computation to it.

    Parameters
    ----------
    experiment : pcvl.Experiment
        Unitary Perceval experiment produced by the FeatureMap descriptor.
    input_state : list[int]
        Input Fock state occupation list.
    input_size : int
        Dimension of the classical input vector.
    input_parameters : list[str]
        Perceval parameter prefix(es) used for angle encoding.
    trainable_parameters : list[str]
        Perceval parameter prefixes exposed as trainable ``nn.Parameter``s.
    computation_space : ComputationSpace
        Computation space used for basis enumeration.
    dtype : torch.dtype
        Real dtype for internal tensors.
    device : torch.device
        Device on which computation graphs are placed.

    Notes
    -----
    ``_compute_unitary`` calls ``computation_process.converter.to_tensor``
    directly. This is safe for single-threaded use but will race under
    ``DataLoader(num_workers>0)`` because the converter holds shared mutable
    state. Use ``num_workers=0`` when the kernel module is in a DataLoader worker.
    """

    def __init__(
        self,
        experiment: pcvl.Experiment,
        input_state: list[int],
        input_size: int,
        input_parameters: list[str],
        trainable_parameters: list[str],
        computation_space: ComputationSpace,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__(
            experiment=experiment,
            input_state=input_state,
            input_size=input_size,
            input_parameters=input_parameters,
            trainable_parameters=trainable_parameters,
            measurement_strategy=MeasurementStrategy.probs(
                computation_space=computation_space,
            ),
            dtype=dtype,
            device=device,
        )

        raw_keys = self._raw_output_keys
        try:
            self._input_state_index: int = raw_keys.index(tuple(input_state))
        except ValueError as exc:
            raise ValueError(
                "Input state is not present in the simulation basis produced by the circuit."
            ) from exc

        # Build the detection weight vector: track which output bin the input
        # state maps to after photon loss and detector transforms.
        weight_device = self.device or torch.device("cpu")
        one_hot = torch.zeros(len(raw_keys), dtype=self.dtype, device=weight_device)
        one_hot[self._input_state_index] = 1.0
        detection_vector = self._apply_detection_pipeline(one_hot)
        detection_vector = detection_vector.to(dtype=self.dtype, device=weight_device)

        nonzero = torch.nonzero(detection_vector > 1e-8, as_tuple=True)[0]
        self._input_detection_index: int | None = None
        if nonzero.numel() == 1 and torch.isclose(
            detection_vector[nonzero[0]],
            torch.tensor(
                1.0, dtype=detection_vector.dtype, device=detection_vector.device
            ),
            atol=1e-6,
        ):
            self._input_detection_index = int(nonzero[0].item())
        self.register_buffer("_input_detection_weights", detection_vector)
        self._kernel_autodiff_process = AutoDiffProcess()

    def _encode_single(self, x: Tensor) -> Tensor:
        """Encode one datapoint to the circuit's input parameter shape.

        Parameters
        ----------
        x : torch.Tensor
            Flat feature tensor of length ``input_size``.

        Returns
        -------
        torch.Tensor
            Encoded tensor matching the number of circuit input parameters.
        """
        x = x.to(dtype=self.dtype, device=self.device).reshape(-1)
        prefix = self.input_parameters[0] if self.input_parameters else None

        # Angle encoding specs (from CircuitBuilder) take priority.
        if prefix and self.angle_encoding_specs:
            spec = self.angle_encoding_specs.get(prefix)
            if spec:
                return self._prepare_input_encoding(x, prefix)

        # No CircuitBuilder metadata available: the circuit was constructed
        # directly with pcvl.Circuit, so angle_encoding_specs is empty.
        # Fall back to direct pass-through or subset-sum expansion.
        spec_mappings = self.computation_process.converter.spec_mappings
        px_len = len(spec_mappings.get(prefix, [])) if prefix else x.numel()

        if x.numel() == px_len or px_len == 0:
            return x
        if x.numel() < px_len:
            return self._subset_sum_expand(x, px_len)
        return x[:px_len]

    @staticmethod
    def _subset_sum_expand(x: Tensor, k: int) -> Tensor:
        """Expand a vector into deterministic subset sums up to length ``k``.

        Parameters
        ----------
        x : torch.Tensor
            One-dimensional input tensor.
        k : int
            Target output length.

        Returns
        -------
        torch.Tensor
            Tensor of length ``k`` containing subset sums, padded with zeros
            when fewer than ``k`` sums are available.
        """
        d = x.shape[0]
        vals: list[Tensor] = []
        for r in range(1, d + 1):
            for idxs in itertools.combinations(range(d), r):
                vals.append(x[list(idxs)].sum())
                if len(vals) == k:
                    return torch.stack(vals, dim=0)
        if len(vals) == 0:
            return torch.zeros(k, dtype=x.dtype, device=x.device)
        pad = k - len(vals)
        return torch.cat(
            [
                torch.stack(vals, dim=0),
                torch.zeros(pad, dtype=x.dtype, device=x.device),
            ],
            dim=0,
        )

    def _apply_detection_pipeline(self, distribution: Tensor) -> Tensor:
        """Apply photon-loss and detector transforms in sequence.

        This is the single canonical place where the two-step detection
        pipeline is applied. Both the detection-weight initialisation in
        ``__init__`` and the per-batch path in ``_compute_transition_probs``
        call this method so that adding a future step only requires one change.

        Parameters
        ----------
        distribution : torch.Tensor
            Probability distribution tensor; either 1-D (single output
            vector) or 2-D ``(batch, bins)``.

        Returns
        -------
        torch.Tensor
            Distribution after photon-loss and detector transforms, with
            any trailing batch dimension squeezed away if the input was 1-D.
        """
        result = self._apply_photon_loss_transform(distribution)
        if result.ndim > 1 and distribution.ndim == 1:
            result = result.squeeze(0)
        result = self._apply_detector_transform(result)
        if result.ndim > 1 and distribution.ndim == 1:
            result = result.squeeze(0)
        return result

    def _compute_unitary(self, x_enc: Tensor) -> Tensor:
        """Evaluate the circuit unitary for an already-encoded input.

        Parameters
        ----------
        x_enc : torch.Tensor
            Encoded input tensor produced by :meth:`_encode_single`.

        Returns
        -------
        torch.Tensor
            Complex unitary matrix of shape ``(m, m)``.
        """
        return self.computation_process.converter.to_tensor(*self.thetas, x_enc)

    def _compute_kernel_unitary(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Compute the combined kernel unitary ``U(x1) @ U†(x2)``.

        Parameters
        ----------
        x1 : torch.Tensor
            First raw feature tensor.
        x2 : torch.Tensor
            Second raw feature tensor.

        Returns
        -------
        torch.Tensor
            Combined kernel unitary of shape ``(m, m)``.
        """
        U1 = self._compute_unitary(self._encode_single(x1))
        U2 = self._compute_unitary(self._encode_single(x2))
        return U1 @ U2.conj().mT

    def _compute_unitary_batch(self, x_batch: Tensor) -> Tensor:
        """Compute a batch of circuit unitaries.

        Parameters
        ----------
        x_batch : torch.Tensor
            Batch of feature tensors with shape ``(N, input_size)``.

        Returns
        -------
        torch.Tensor
            Stacked unitary tensor of shape ``(N, m, m)``.
        """
        # Serial loop is intentional: CircuitConverter.to_tensor holds shared
        # mutable state and is not safe to call concurrently.
        return torch.stack([
            self._compute_unitary(self._encode_single(x)) for x in x_batch
        ])

    def _compute_transition_probs(
        self,
        all_circuits: Tensor,
        input_state: list[int],
        shots: int,
        sampling_method: str,
    ) -> Tensor:
        """Run SLOS and transforms on a batch of combined kernel unitaries.

        Parameters
        ----------
        all_circuits : torch.Tensor
            Batch of combined kernel unitaries with shape ``(P, m, m)``.
        input_state : list[int]
            Input Fock occupation list.
        shots : int
            Number of pseudo-sampling shots; 0 for exact probabilities.
        sampling_method : str
            Sampling method; one of ``"multinomial"``, ``"binomial"``,
            ``"gaussian"``.

        Returns
        -------
        torch.Tensor
            Transition probability for each circuit, shape ``(P,)``.
        """
        _, probabilities = self.computation_process.simulation_graph.compute_probs(
            all_circuits, input_state
        )
        if probabilities.ndim == 1:
            probabilities = probabilities.unsqueeze(0)
        probabilities = probabilities.to(dtype=self.dtype)
        detection_probs = self._apply_detection_pipeline(probabilities)

        if shots > 0:
            detection_probs = self._kernel_autodiff_process.sampling_noise.pcvl_sampler(
                detection_probs, shots, sampling_method
            )

        if self._input_detection_index is not None:
            return detection_probs[:, self._input_detection_index]
        input_detection_weights = cast(Tensor, self._input_detection_weights)
        weights = input_detection_weights.to(
            dtype=detection_probs.dtype, device=detection_probs.device
        )
        return detection_probs @ weights


class FidelityKernel(MerlinModule):
    r"""
    Fidelity Quantum Kernel

    Next-release deprecations for the legacy ``FeatureMap`` unitary path:

    - ``FeatureMap.compute_unitary``
    - ``FeatureMap._px_len``
    - ``FeatureMap._encode_x``
    - ``FeatureMap._encode_with_specs``
    - ``FeatureMap._subset_sum_expand``

    ``FidelityKernel`` does not use these methods. It delegates kernel
    computation to ``_CCInvQuantumLayer``, which uses the :class:`QuantumLayer`
    backend.

    For a given input Fock state, :math:`|s \rangle` and feature map,
    :math:`U`, the fidelity quantum kernel estimates the following inner
    product using SLOS:

    .. math::
        |\langle s | U^{\dagger}(x_2) U(x_1) | s \rangle|^{2}

    Transition probabilities are computed in parallel for each pair of
    datapoints in the input datasets.

    Parameters
    ----------
    feature_map : FeatureMap
        Feature map object that encodes a given datapoint within its circuit.
    input_state : list[int]
        Input state into the circuit.
    shots : int | None
        Number of circuit shots. If ``None``, the exact transition
        probabilities are returned. Default: ``None``.
    sampling_method : str
        Probability distributions are post-processed with a pseudo-sampling
        method: ``"multinomial"``, ``"binomial"``, or ``"gaussian"``.
        Default is ``"multinomial"``.
    computation_space : ComputationSpace | str | None
        Logical computation subspace; one of
        ``{"fock", "unbunched", "dual_rail"}``. Default: ``FOCK``.
    force_psd : bool
        Projects the training kernel matrix to the closest positive
        semi-definite matrix. Default is ``True``.
    device : torch.device | None
        Device on which to perform SLOS.
    dtype : str | torch.dtype | None
        Datatype with which to perform SLOS.

    Examples
    --------
    For a given training and test datasets, one can construct the training and
    test kernel matrices with the following structure:

    .. code-block:: python

        circuit = Circuit(2) // PS(P("X0")) // BS() // PS(P("X1")) // BS()
        feature_map = FeatureMap(circuit, ["X"])

        quantum_kernel = FidelityKernel(
            feature_map,
            input_state=[0, 4],
        )
        K_train = quantum_kernel(X_train)
        K_test = quantum_kernel(X_test, X_train)

    Use with scikit-learn for kernel-based machine learning:

    .. code-block:: python

        from sklearn.svm import SVC

        svc = SVC(kernel="precomputed")
        svc.fit(K_train, y_train)
        y_pred = svc.predict(K_test)

    .. warning::
        ``FidelityKernel`` is **not thread-safe**. The internal
        ``CircuitConverter`` holds shared mutable state. Using this module
        inside a ``DataLoader`` with ``num_workers > 0`` will produce a data
        race. Always set ``num_workers=0`` when the kernel is used as part of
        a DataLoader worker.
    """

    @sanitize_parameters
    def __init__(
        self,
        feature_map: FeatureMap,
        input_state: list[int],
        *,
        shots: int | None = None,
        sampling_method: str = "multinomial",
        computation_space: ComputationSpace | str | None = None,
        force_psd: bool = True,
        device: torch.device | None = None,
        dtype: str | torch.dtype | None = None,
    ):
        super().__init__()
        if computation_space is None:
            computation_space = ComputationSpace.FOCK
        else:
            computation_space = ComputationSpace.coerce(computation_space)
        self.computation_space = computation_space
        self.feature_map = feature_map
        self.input_state = input_state
        self.shots = shots or 0
        self.sampling_method = sampling_method
        self.no_bunching = self.computation_space is not ComputationSpace.FOCK
        self.force_psd = force_psd
        base_device = device if device is not None else feature_map.device
        self.device = (
            torch.device(base_device)
            if base_device is not None
            else torch.device("cpu")
        )
        # Normalize to a torch.dtype
        if dtype is None:
            self.dtype = feature_map.dtype
        else:
            self.dtype = to_torch_dtype(dtype, default=feature_map.dtype)
        self.input_size = self.feature_map.input_size

        if self.feature_map.circuit.m != len(input_state):
            raise ValueError("Input state length does not match circuit size.")

        experiment = getattr(self.feature_map, "experiment", None)
        if experiment is None:
            experiment = pcvl.Experiment(self.feature_map.circuit)
            self.feature_map.experiment = experiment

        self._validate_experiment(experiment)
        self.experiment = experiment
        experiment_circuit = self.experiment.unitary_circuit()
        if experiment_circuit.m != self.feature_map.circuit.m:
            raise ValueError(
                "Experiment circuit must have the same number of modes as the feature map circuit."
            )

        if max(input_state) > 1 and self.computation_space is not ComputationSpace.FOCK:
            raise ValueError(
                f"Bunching must be enabled for an input state with"
                f"{max(input_state)} in one mode."
            )
        elif (
            all(x == 1 for x in input_state)
            and self.computation_space is not ComputationSpace.FOCK
        ):
            raise ValueError(
                "For non-FOCK computation_space, the kernel value will always be 1 "
                "for an input state with a photon in all modes."
            )

        m = len(input_state)
        _detectors, _empty_detectors = resolve_detectors(self.experiment, m)

        # Verify that no Detector was defined in experiment if using non-FOCK space.
        if not _empty_detectors and self.computation_space is not ComputationSpace.FOCK:
            raise RuntimeError(
                "computation_space must be FOCK if Experiment contains at least one Detector."
            )

        # Resolve noise model presence from the experiment before building the backend.
        _, empty_noise_model = resolve_photon_loss(self.experiment, m)
        self.has_custom_noise_model = not empty_noise_model

        self._quantum_layer = _CCInvQuantumLayer(
            experiment=self.experiment,
            input_state=input_state,
            input_size=self.input_size,
            input_parameters=[self.feature_map.input_parameters],
            trainable_parameters=self.feature_map.trainable_parameters,
            computation_space=self.computation_space,
            dtype=self.dtype,
            device=self.device,
        )

        self.is_trainable = feature_map.is_trainable

    def forward(
        self,
        x1: float | np.ndarray | torch.Tensor,
        x2: float | np.ndarray | torch.Tensor | None = None,
    ):
        """Calculate the quantum kernel for input data ``x1`` and ``x2``.

        If ``x1`` and ``x2`` are datapoints, a scalar value is returned. For
        input datasets the kernel matrix is computed.

        Parameters
        ----------
        x1 : float | numpy.ndarray | torch.Tensor
            First input datapoint or dataset.
        x2 : float | numpy.ndarray | torch.Tensor | None
            Second input datapoint or dataset. If omitted, the training kernel
            matrix for ``x1`` is computed.

        Returns
        -------
        torch.Tensor
            Scalar kernel value for datapoints, or a kernel matrix for datasets.
        """
        # Convert inputs to tensors and ensure they are on the correct device
        if not isinstance(x1, torch.Tensor):
            x1 = torch.as_tensor(x1, dtype=self.dtype)

        if x2 is not None:
            if isinstance(x2, np.ndarray):
                x2 = torch.from_numpy(x2).to(device=x1.device, dtype=self.dtype)
            elif isinstance(x2, torch.Tensor):
                x2 = x2.to(device=x1.device, dtype=self.dtype)

        # Return scalar value for input datapoints
        if self.feature_map.is_datapoint(x1):
            if x2 is None:
                raise ValueError("For input datapoints, please specify an x2 argument.")
            return self._return_kernel_scalar(x1, x2)

        # Ensure tensors before reshaping (satisfies mypy)
        if x2 is not None and not isinstance(x2, torch.Tensor):
            x2 = torch.as_tensor(x2, dtype=self.dtype, device=self.device)

        if isinstance(x2, torch.Tensor) or x2 is None:
            x1 = x1.reshape(-1, self.input_size)
            x2 = x2.reshape(-1, self.input_size) if x2 is not None else None
        else:
            raise TypeError("x2 is not None nor torch.Tensor")

        equal_inputs = self._check_equal_inputs(x1, x2)
        U_forward = self._quantum_layer._compute_unitary_batch(x1).to(x1.device)

        len_x1 = len(x1)
        if x2 is not None:
            U_adjoint = (
                self._quantum_layer
                ._compute_unitary_batch(x2)
                .conj()
                .transpose(1, 2)
                .to(x1.device)
            )
            # All (len_x1 × len_x2) pair unitaries
            all_circuits = U_forward.unsqueeze(1) @ U_adjoint.unsqueeze(0)
            all_circuits = all_circuits.view(-1, *all_circuits.shape[2:])
        else:
            U_adjoint = U_forward.conj().transpose(1, 2)
            # Upper-triangle pairs only to exploit symmetry
            upper_idx = torch.triu_indices(
                len_x1,
                len_x1,
                offset=1,
                device=x1.device,
            )
            all_circuits = U_forward[upper_idx[0]] @ U_adjoint[upper_idx[1]]

        transition_probs = self._quantum_layer._compute_transition_probs(
            all_circuits, self.input_state, self.shots, self.sampling_method
        )

        if x2 is None:
            kernel_matrix = torch.zeros(
                len_x1, len_x1, dtype=self.dtype, device=x1.device
            )
            upper_idx = upper_idx.to(x1.device)
            transition_probs = transition_probs.to(dtype=self.dtype, device=x1.device)
            kernel_matrix[upper_idx[0], upper_idx[1]] = transition_probs
            kernel_matrix[upper_idx[1], upper_idx[0]] = transition_probs
            kernel_matrix.fill_diagonal_(1)

            if self.force_psd:
                kernel_matrix = self._project_psd(kernel_matrix)
        else:
            transition_probs = transition_probs.to(dtype=self.dtype, device=x1.device)
            kernel_matrix = transition_probs.reshape(len_x1, len(x2))

            if self.force_psd and equal_inputs:
                kernel_matrix = 0.5 * (kernel_matrix + kernel_matrix.T)
                kernel_matrix = self._project_psd(kernel_matrix)

        return kernel_matrix

    def _return_kernel_scalar(
        self,
        x1: Tensor | np.ndarray | float | int,
        x2: Tensor | np.ndarray | float | int,
    ) -> float:
        """Return the scalar kernel value for a single pair of datapoints.

        Parameters
        ----------
        x1 : torch.Tensor | numpy.ndarray | float | int
            First datapoint.
        x2 : torch.Tensor | numpy.ndarray | float | int
            Second datapoint.

        Returns
        -------
        float
            Scalar kernel value.
        """
        if isinstance(x1, np.ndarray):
            x1_t: Tensor = torch.from_numpy(x1)
        elif isinstance(x1, (float, int)):
            x1_t = torch.tensor([x1])
        else:
            x1_t = x1
        if isinstance(x2, np.ndarray):
            x2_t: Tensor = torch.from_numpy(x2)
        elif isinstance(x2, (float, int)):
            x2_t = torch.tensor([x2])
        else:
            x2_t = x2

        x1_t = torch.as_tensor(x1_t, dtype=self.dtype, device=self.device).reshape(
            self.input_size
        )
        x2_t = torch.as_tensor(x2_t, dtype=self.dtype, device=self.device).reshape(
            self.input_size
        )

        kernel_unitary = self._quantum_layer._compute_kernel_unitary(x1_t, x2_t)
        transition_probs = self._quantum_layer._compute_transition_probs(
            kernel_unitary.unsqueeze(0),
            self.input_state,
            self.shots,
            self.sampling_method,
        )
        return transition_probs[0].item()

    @classmethod
    @sanitize_parameters
    def simple(
        cls,
        input_size: int,
        *,
        shots: int = 0,
        sampling_method: str = "multinomial",
        computation_space: ComputationSpace | str | None = None,
        force_psd: bool = True,
        dtype: str | torch.dtype = torch.float32,
        device: torch.device | None = None,
        angle_encoding_scale: float = 1.0,
        n_modes: int = None,
    ) -> "FidelityKernel":
        """Create a simple fidelity kernel with minimal configuration.

        Parameters
        ----------
        input_size : int
            Classical feature dimension.
        shots : int
            Number of pseudo-sampling shots. Default is ``0``.
        sampling_method : str
            Sampling method used when ``shots`` is positive. Default is
            ``"multinomial"``.
        computation_space : ComputationSpace | str | None
            Logical computation subspace.
        force_psd : bool
            Whether to project the training kernel matrix to the nearest
            positive semi-definite matrix. Default is ``True``.
        dtype : str | torch.dtype
            Target dtype for internal tensors. Default is ``torch.float32``.
        device : torch.device | None
            Device on which to execute computations.
        angle_encoding_scale : float
            Global scaling applied to angle encoding features. Default is ``1.0``.
        n_modes : int | None
            Number of photonic modes used by the helper construction.

        Returns
        -------
        FidelityKernel
            Configured fidelity kernel.
        """
        feature_map = FeatureMap.simple(
            input_size=input_size,
            n_modes=n_modes,
            dtype=dtype,
            device=device,
            angle_encoding_scale=angle_encoding_scale,
        )

        if n_modes is None:
            state_size = input_size + 1
        else:
            state_size = n_modes

        input_state = state_size * [0]
        for i in range(state_size):
            if i % 2 == 0:
                input_state[i] = 1

        return cls(
            feature_map=feature_map,
            input_state=input_state,
            shots=shots,
            sampling_method=sampling_method,
            computation_space=computation_space,
            force_psd=force_psd,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _project_psd(matrix: Tensor) -> Tensor:
        """Projects a symmetric matrix to closest positive semi-definite"""
        eigenvals, eigenvecs = torch.linalg.eigh(matrix)
        eigenvals = torch.diag(torch.where(eigenvals > 0, eigenvals, 0))
        return eigenvecs @ eigenvals @ eigenvecs.T

    @staticmethod
    def _check_equal_inputs(x1, x2) -> bool:
        """Checks whether x1 and x2 are equal."""
        if x2 is None:
            return True
        elif x1.shape != x2.shape:
            return False
        elif isinstance(x1, Tensor):
            return torch.allclose(x1, x2)
        elif isinstance(x1, np.ndarray):
            return np.allclose(x1, x2)
        return False

    @staticmethod
    def _validate_experiment(experiment: pcvl.Experiment) -> None:
        """Validate that the provided experiment is compatible with fidelity kernels."""
        if (
            not experiment.is_unitary
            or experiment.post_select_fn is not None
            or experiment.heralds
            or experiment.in_heralds
        ):
            raise ValueError(
                "The provided experiment must be unitary, and must not have post-selection or heralding."
            )
        if experiment.min_photons_filter:
            warnings.warn(
                "The 'min_photons_filter' from the experiment is currently ignored.",
                UserWarning,
                stacklevel=2,
            )
