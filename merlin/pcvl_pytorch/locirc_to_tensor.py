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

from __future__ import annotations

import torch
from multipledispatch import dispatch
from perceval.components import (
    BS,
    PERM,
    PS,
    AComponent,
    Barrier,
    BSConvention,
    Circuit,
    Unitary,
)

from ..utils.dtypes import resolve_float_complex

SUPPORTED_COMPONENTS = (PS, BS, PERM, Unitary, Barrier)
"""Tuple of quantum components supported by CircuitConverter.

Components:
    PS: Phase shifter with single phi parameter
    BS: Beam splitter with theta and four phi parameters
    PERM: Mode permutation (no parameters)
    Unitary: Generic unitary matrix (no parameters)
    Barrier: Synchronization barrier (removed during compilation)
"""


class CircuitConverter:
    """Convert a parameterized Perceval circuit into a differentiable PyTorch unitary matrix.

    This class converts Perceval quantum circuits into PyTorch tensors that can be used
    in neural network training with automatic differentiation. It supports batch processing
    for efficient training and handles various quantum components like beam splitters,
    phase shifters, and unitary operations.

    Parameters
    ----------
    circuit : pcvl.Circuit
        Perceval circuit to convert.
    input_specs : list[str] | None
        Parameter name prefixes used to group parameters into input tensors.
    dtype : torch.dtype
        Target tensor dtype.
    device : torch.device
        Device used for tensor operations.
    phase_imprecision : float
        Deterministic quantization step applied to every phase shifter before
        building the unitary. This models finite phase-setting resolution: a
        commanded phase ``phi`` is mapped to
        ``round(phi / phase_imprecision) * phase_imprecision`` with a
        straight-through estimator, so the forward pass uses the quantized
        value while the backward pass keeps the identity gradient through the
        commanded phase. This is nearest-grid rounding, not truncation. Exact
        half-step ties follow ``torch.round`` behavior. For example,
        ``phi = pi / 8`` with ``phase_imprecision = pi / 4`` quantizes to
        ``0`` because ``round(0.5) == 0``. Default value is 0.0.
    phase_error : float
        Stochastic uniform perturbation half-width in radians. This models
        random phase noise around the quantized phase after any
        ``phase_imprecision`` step. When active, the effective sampled phase is
        ``round(phi / phase_imprecision) * phase_imprecision + epsilon`` with
        ``epsilon ~ Uniform(-phase_error, phase_error)``. If
        ``phase_imprecision`` is inactive, the sampled phase is
        ``phi + epsilon``. Fresh samples are drawn only when :meth:`to_tensor`
        is called with ``apply_phase_error=True``; otherwise the converter
        remains deterministic. Default value is 0.0.

    Notes
    -----
    Supported Components:
        - PS (Phase Shifter)
        - BS (Beam Splitter)
        - PERM (Permutation)
        - Unitary (Generic unitary matrix)
        - Barrier (no-op, removed during compilation)

    Phase Noise Parameter Flow:
        Phase noise parameters (`phase_imprecision` and `phase_error`) are
        configured at converter initialization and automatically applied during
        unitary generation. The flow is:

        1. **Initialization**: User passes `phase_imprecision` and/or
           `phase_error` to CircuitConverter via QuantumLayer
           (Step 4: through InitializationContext → ComputationProcessFactory
           → ComputationProcess → CircuitConverter).

        2. **Compilation**: During `_compile_circuit()`, constant phase shifters
           are marked as dynamic if `phase_error > 0.0`, ensuring fresh
           perturbations on each call. Quantization-only noise allows
           precomputation since it is deterministic.

        3. **Conversion**: Each call to `to_tensor(*params, apply_phase_error=bool)`
           applies both quantization (always, if configured) and perturbations
           (only if `apply_phase_error=True`). Monte Carlo sampling is done by
           calling `to_tensor()` multiple times with `apply_phase_error=True`
           and averaging the resulting probability distributions.

        4. **Gradient Flow**: Phase quantization uses straight-through estimators
           to preserve gradients to the commanded phase. Perturbations use
           `torch.empty_like(phase)` to ensure proper device/dtype handling
           and do NOT require gradients (they are stochastic noise, not
           learnable parameters).

        5. **Effective Phase**: For a phase shifter commanded to phase ``phi``,
           the forward phase is:

           - ``phi`` when both phase noises are inactive;
           - ``round(phi / phase_imprecision) * phase_imprecision`` when only
             ``phase_imprecision`` is active;
           - ``phi + epsilon`` when only ``phase_error`` is active;
           - ``round(phi / phase_imprecision) * phase_imprecision + epsilon``
             when both are active.

           The quantization uses nearest-grid rounding through
           :func:`torch.round`; it is not floor or truncation.

    Example:
        Basic usage with a single phase shifter:

        >>> import torch
        >>> import perceval as pcvl
        >>> from merlin.pcvl_pytorch.locirc_to_tensor import CircuitConverter
        >>>
        >>> # Create a simple circuit with one phase shifter
        >>> circuit = pcvl.Circuit(1) // pcvl.PS(pcvl.P("phi"))
        >>>
        >>> # Convert to PyTorch with gradient tracking
        >>> converter = CircuitConverter(circuit, input_specs=["phi"])
        >>> phi_params = torch.tensor([0.5], requires_grad=True)
        >>> unitary = converter.to_tensor(phi_params)
        >>> print(unitary.shape)  # torch.Size([1, 1])

        Multiple parameters with grouping:

        >>> # Circuit with multiple phase shifters
        >>> circuit = (pcvl.Circuit(2)
        ...            // pcvl.PS(pcvl.P("theta1"))
        ...            // (1, pcvl.PS(pcvl.P("theta2"))))
        >>>
        >>> converter = CircuitConverter(circuit, input_specs=["theta"])
        >>> theta_params = torch.tensor([0.1, 0.2], requires_grad=True)
        >>> unitary = converter.to_tensor(theta_params)
        >>> print(unitary.shape)  # torch.Size([2, 2])

        Batch processing for training:

        >>> # Batch of parameter values
        >>> batch_params = torch.tensor([[0.1], [0.2], [0.3]], requires_grad=True)
        >>> converter = CircuitConverter(circuit, input_specs=["phi"])
        >>> batch_unitary = converter.to_tensor(batch_params)
        >>> print(batch_unitary.shape)  # torch.Size([3, 1, 1])

        Training integration:

        >>> # Training loop with beam splitter
        >>> circuit = pcvl.Circuit(2) // pcvl.BS.Rx(pcvl.P("theta"))
        >>> converter = CircuitConverter(circuit, ["theta"])
        >>> theta = torch.tensor([0.5], requires_grad=True)
        >>> optimizer = torch.optim.Adam([theta], lr=0.01)
        >>>
        >>> for step in range(10):
        ...     optimizer.zero_grad()
        ...     unitary = converter.to_tensor(theta)
        ...     loss = some_loss_function(unitary)
        ...     loss.backward()
        ...     optimizer.step()
    """

    def __init__(
        self,
        circuit: Circuit,
        input_specs: list[str] = None,
        dtype: torch.dtype = torch.complex64,
        device: torch.device = torch.device("cpu"),
        phase_imprecision: float = 0.0,
        phase_error: float = 0.0,
    ):
        """Initialize the CircuitConverter with a Perceval circuit.

        Parameters
        ----------
        circuit : pcvl.Circuit
            Parameterized Perceval circuit to convert.
        input_specs : list[str] | None
            Parameter name prefixes used to group parameters into separate
            tensors. If ``None``, all parameters go into a single tensor.
        dtype : torch.dtype
            Tensor dtype.
        device : torch.device
            PyTorch device for tensor operations.
        phase_imprecision : float
            Deterministic quantization step applied to every phase shifter
            before building the unitary. This models finite phase-setting
            resolution: a commanded phase ``phi`` is mapped to
            ``round(phi / phase_imprecision) * phase_imprecision`` with a
            straight-through estimator, so the forward pass uses the quantized
            value while the backward pass keeps the identity gradient through
            the commanded phase. This is nearest-grid rounding, not
            truncation. If omitted, no phase quantization is applied. Default
            value is 0.0.
        phase_error : float
            Stochastic uniform perturbation half-width in radians. This models
            random phase noise around the configured phase after any
            ``phase_imprecision`` quantization. The sampled perturbation is
            added after quantization, so both noises compose as
            ``round(phi / phase_imprecision) * phase_imprecision + epsilon``.
            Fresh samples are drawn only when :meth:`to_tensor` is called with
            ``apply_phase_error=True``; otherwise the converter remains
            deterministic. If omitted, no stochastic phase perturbation is
            configured. Default value is 0.0.

        Raises
        ------
        ValueError
            If ``input_specs`` do not match circuit parameters, or if a phase
            noise value is negative.
        TypeError
            If ``circuit`` is not a Perceval circuit.
        """

        # device is the device where the tensors will be allocated, default is set with torch.device('xxx')
        # in pytorch module, there is no discovery of the device from parameters, so it is the user's responsibility to
        # set the device, with .to() before calling the generation function
        self.device = device
        self.input_params = None
        self.batch_size = 1

        self.set_dtype(dtype)
        self._phase_imprecision = float(phase_imprecision)
        self._phase_error = float(phase_error)
        self._apply_phase_error = False

        # Validate that phase noise parameters are non-negative
        if self._phase_imprecision < 0.0:
            raise ValueError("phase_imprecision must be non-negative.")
        if self._phase_error < 0.0:
            raise ValueError("phase_error must be non-negative.")

        assert isinstance(circuit, Circuit), (
            f"Expected a Perceval LO circuit, but got {type(circuit).__name__}"
        )
        self.circuit = circuit

        # Create parameter mapping - it will map parameter names to their index in the input tensors
        self.param_mapping = {}
        self.spec_mappings = {}  # Track the mapping of input specs to parameter names

        self.nb_input_tensor = input_specs and len(input_specs) or 0
        param_names = [p.name for p in circuit.get_parameters()]

        if input_specs is None:
            self.param_mapping = {
                p.name: (0, idx) for idx, p in enumerate(self.circuit.get_parameters())
            }
        else:
            # Now create the mappings for parameters
            for i, spec in enumerate(input_specs):
                matching_params = [p for p in param_names if p.startswith(spec)]
                self.spec_mappings[spec] = matching_params

                if not matching_params:
                    raise ValueError(
                        f"No parameters found matching the input spec '{spec}'."
                    )
                for j, param in enumerate(matching_params):
                    self.param_mapping[param] = (i, j)

            # Check if all parameters are covered
            for param in param_names:
                if param not in self.param_mapping:
                    raise ValueError(
                        f"Parameter '{param}' not covered by any input spec"
                    )

        self.list_rct = self._compile_circuit()

    def set_dtype(self, dtype: torch.dtype):
        """Set the tensor data types for float and complex operations.

        Parameters
        ----------
        dtype : torch.dtype
            Target dtype (float32/complex64 or float64/complex128).

        Raises
        ------
        TypeError
            If ``dtype`` is not supported.
        """
        float_dtype, complex_dtype = resolve_float_complex(dtype)
        self.tensor_fdtype = float_dtype
        self.tensor_cdtype = complex_dtype

    def to(self, dtype: torch.dtype, device: str | torch.device):
        """Move the converter to a specific device and dtype.

        Parameters
        ----------
        dtype : torch.dtype
            Target tensor dtype (float32/complex64 or float64/complex128).
        device : str | torch.device
            Target device (string or torch.device).

        Returns
        -------
        CircuitConverter
            ``self`` for method chaining.

        Raises
        ------
        TypeError
            If ``device`` type is not supported.
        """
        if isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            raise TypeError(
                f"Expected a string or torch.device, but got {type(device).__name__}"
            )
        self.set_dtype(dtype)

        for idx, (r, c) in enumerate(self.list_rct):
            if isinstance(c, torch.Tensor):
                self.list_rct[idx] = (
                    r,
                    c.to(dtype=self.tensor_cdtype, device=self.device),
                )

        return self

    def _compile_circuit(self):
        """Precompile the circuit to optimize performance.

        This method:
        1. Removes barrier components (no-ops)
        2. Precomputes tensors for components without parameters
        3. Merges adjacent non-parameterized components to reduce computation

        Returns
        -------
        list[tuple[range | object, torch.Tensor | AComponent]]
            List of (mode_range, component_or_tensor) tuples for the compiled circuit

        Raises
        ------
        TypeError
            If the circuit contains unsupported component types.
        """

        # we are building a list of components or precompiled tensors or dimension (1, m, m)
        list_rct = []
        for r, c in self.circuit:
            if not isinstance(c, SUPPORTED_COMPONENTS):
                raise TypeError(
                    f"{c} type not supported for conversion to PyTorch tensor."
                )
            if isinstance(c, Barrier):
                continue
            # Check if this PS component requires dynamic handling due to phase_error.
            # Phase shifters with active phase_error must remain dynamic (not precomputed)
            # because fresh perturbations must be drawn on each call to to_tensor().
            # In contrast, phase_imprecision-only PS can still be precomputed since
            # quantization is deterministic.
            is_phase_error_sensitive = isinstance(c, PS) and self._phase_error > 0.0
            if not c.get_parameters(all_params=False) and not is_phase_error_sensitive:
                # we can already compute the tensor for this component
                curr_comp_tensor = self._compute_tensor(c)
                list_rct.append((r, curr_comp_tensor))
            else:
                list_rct.append((r, c))

        # in second pass, we will be fusing the adjacent numeric components together
        for idx, (r, ct) in enumerate(list_rct):
            if ct is None:
                # this component has been merged with a previous one, skip it
                continue
            if isinstance(ct, torch.Tensor):
                # let us check all the following components that could be merged with this one
                merge_group = [(r, ct)]
                min_group = r[0]
                max_group = r[-1]
                blocked_modes = set()
                for j in range(idx + 1, len(list_rct)):
                    r2, c2 = list_rct[j]
                    if c2 is None:
                        continue
                    if not isinstance(c2, torch.Tensor) or any(
                        mode in blocked_modes for mode in r2
                    ):
                        for ir in r2:
                            blocked_modes.add(ir)
                        if len(blocked_modes) == self.circuit.m:
                            # all modes are blocked, we cannot merge anymore
                            break
                    else:
                        # we can merge this component with the previous one
                        merge_group.append((r2, c2))
                        if r2[0] < min_group:
                            min_group = r2[0]
                        if r2[-1] > max_group:
                            max_group = r2[-1]
                        # remove the component from the list
                        list_rct[j] = (r2, None)  # noqa: B909
                if len(merge_group) > 1:
                    # we have a group of components that can be merged
                    # we will compute the tensor for the whole group
                    merged_tensor = torch.eye(
                        max_group - min_group + 1,
                        dtype=self.tensor_cdtype,
                        device=self.device,
                    )
                    for r, c in merge_group:
                        c = c.to(self.device)
                        merged_tensor[r[0] - min_group : (r[-1] - min_group + 1), :] = (
                            c
                            @ merged_tensor[
                                r[0] - min_group : (r[-1] - min_group + 1), :
                            ]
                        )
                    list_rct[idx] = (range(min_group, max_group + 1), merged_tensor)

        # Remove None entries from the list
        return [item for item in list_rct if item[1] is not None]

    def to_tensor(
        self,
        *input_params: torch.Tensor,
        batch_size: int | None = None,
        apply_phase_error: bool = False,
    ) -> torch.Tensor:
        r"""Convert the parameterized circuit to a PyTorch unitary tensor.

        Phase Noise Processing:
            This method applies configured phase noise to all phase shifters during
            unitary generation. The noise is applied in two stages:

            1. **phase_imprecision (deterministic, always applied)**:
               If configured, every phase is quantized to the nearest multiple of
               ``phase_imprecision`` using a straight-through estimator. This uses
               ``torch.round(phase / phase_imprecision) * phase_imprecision``:
               it is nearest-grid rounding, not truncation. Exact half-step ties
               follow ``torch.round`` behavior, so ``pi / 8`` with a ``pi / 4``
               step quantizes to ``0``. Gradients flow through the commanded phase,
               while the forward pass uses the quantized value. This is always
               active and does not require ``apply_phase_error=True``.

            2. **phase_error (stochastic, controlled by apply_phase_error flag)**:
               If configured and `apply_phase_error=True`, fresh samples from
               Uniform(-phase_error, phase_error) are drawn and added to each phase
               after quantization. The samples respect the phase tensor's device and
               dtype via `torch.empty_like()`. Each call with `apply_phase_error=True`
               produces a different unitary. For Monte Carlo averaging of probabilistic
               outputs, call this method multiple times with `apply_phase_error=True`,
               collect the resulting probability distributions, and average them.

            Parameter Flow (see class Notes for full context):
            - layer_utils.classify_noise() → extracts phase settings to NoiseGroups
            - ComputationProcess.__init__() → stores phase settings from NoiseGroups
            - ComputationProcess._setup_computation_graphs() → passes to CircuitConverter
            - CircuitConverter.to_tensor() ← receives apply_phase_error flag each call

        Parameters
        ----------
        input_params : torch.Tensor
            Variable number of parameter tensors. Each tensor has shape
            ``(num_params,)`` or ``(batch_size, num_params)`` in the order of
            ``input_specs``.
        batch_size : int | None
            Explicit batch size. If ``None``, it is inferred from the input
            tensors.
        apply_phase_error : bool
            Whether to draw fresh stochastic perturbations for configured
            ``phase_error`` values during this conversion. This flag does not
            affect deterministic ``phase_imprecision`` quantization, which is
            applied whenever ``phase_imprecision`` is positive. The perturbation
            is added after quantization. Default value is False.

        Returns
        -------
        torch.Tensor
            Complex unitary tensor of shape ``(circuit.m, circuit.m)`` for a
            single sample or ``(batch_size, circuit.m, circuit.m)`` for batched
            inputs.

        Raises
        ------
        ValueError
            If the wrong number of input tensors is provided.
        TypeError
            If ``input_params`` is not a list or tuple.
        """
        if len(input_params) == 1 and isinstance(input_params[0], list):
            input_params = input_params[0]  # type: ignore[assignment]
        if len(input_params) != self.nb_input_tensor:
            raise ValueError(
                f"Expected {self.nb_input_tensor} input tensors, but got {len(input_params)}."
            )
        if not isinstance(input_params, list) and not isinstance(input_params, tuple):
            raise TypeError(
                f"Expected a list of input tensors, but got {type(input_params).__name__}."
            )

        self.torch_params = input_params

        if batch_size is None:
            if input_params and input_params[0].dim() > 1:
                has_batch = True
                batch_size = input_params[0].shape[0]
            else:
                has_batch = False
                batch_size = 1
        else:
            has_batch = True
        self.batch_size = batch_size

        previous_apply_phase_error = self._apply_phase_error
        self._apply_phase_error = apply_phase_error
        try:
            converted_tensor = (
                torch
                .eye(self.circuit.m, dtype=self.tensor_cdtype, device=self.device)
                .unsqueeze(0)
                .repeat(batch_size, 1, 1)
            )
            # Build unitary tensor by composing component unitaries
            for r, c in self.list_rct:
                if isinstance(c, torch.Tensor):
                    # If the component is already a tensor, use it directly, just move it to the correct device and dtype
                    # and expand it to the batch size
                    curr_comp_tensor = c.to(
                        dtype=self.tensor_cdtype, device=self.device
                    ).expand(batch_size, -1, -1)
                else:
                    curr_comp_tensor = self._compute_tensor(c)

                # Compose unitaries
                contribution = converted_tensor[..., r[0] : (r[-1] + 1), :].clone()
                converted_tensor[..., r[0] : (r[-1] + 1), :] = (
                    curr_comp_tensor @ contribution.to(curr_comp_tensor.device)
                )
        finally:
            self._apply_phase_error = previous_apply_phase_error

        if not has_batch:
            # If no batch dimension was provided, remove the batch dimension
            converted_tensor = converted_tensor.squeeze(0)

        return converted_tensor

    @dispatch((Unitary, PERM))
    def _compute_tensor(self, comp: AComponent) -> torch.Tensor:
        """Compute tensor for Unitary and Permutation components.

        Args:
            comp: Unitary or PERM component (no parameters)

        Returns:
            Batched unitary tensor of shape (batch_size, comp_size, comp_size)
        """
        return (
            torch
            .tensor(
                comp.compute_unitary(), dtype=self.tensor_cdtype, device=self.device
            )
            .unsqueeze(0)
            .expand(self.batch_size, -1, -1)
        )

    @dispatch(BS)
    def _compute_tensor(self, comp: AComponent) -> torch.Tensor:  # type: ignore[no-redef]
        """Compute tensor for Beam Splitter component.

        Handles different BS conventions (Rx, Ry, H) and processes 5 parameters:
        theta, phi_tl, phi_bl, phi_tr, phi_br

        Args:
            comp: BS component with parameters

        Returns:
            Batched 2x2 unitary tensor of shape (batch_size, 2, 2)

        Raises:
            NotImplementedError: If BS convention is not supported
        """
        param_values = []

        for _index, param in enumerate(comp.get_parameters(all_params=True)):
            if param.is_variable:
                (tensor_id, idx_in_tensor) = self.param_mapping[param.name]
                param_values.append(self.torch_params[tensor_id][..., idx_in_tensor])
            else:
                param_values.append(
                    torch.tensor(
                        float(param), dtype=self.tensor_fdtype, device=self.device
                    )
                )

        cos_theta = torch.cos(param_values[0] / 2)
        sin_theta = torch.sin(param_values[0] / 2)
        phi_tl_tr = param_values[1] + param_values[3]  # phi_tl_val + phi_tr_val
        u00_mul = torch.cos(phi_tl_tr) + 1j * torch.sin(phi_tl_tr)

        phi_tr_bl = param_values[3] + param_values[2]  # phi_tr_val + phi_bl_val
        u01_mul = torch.cos(phi_tr_bl) + 1j * torch.sin(phi_tr_bl)

        phi_tl_br = param_values[1] + param_values[4]  # phi_tl_val + phi_br_val
        u10_mul = torch.cos(phi_tl_br) + 1j * torch.sin(phi_tl_br)

        phi_bl_br = param_values[2] + param_values[4]  # phi_bl_val + phi_br_val
        u11_mul = torch.cos(phi_bl_br) + 1j * torch.sin(phi_bl_br)

        bs_convention = comp._convention
        if bs_convention == BSConvention.Rx:
            unitary_tensor = torch.tensor(
                [[1, 1j], [1j, 1]], dtype=self.tensor_cdtype, device=self.device
            )
        elif bs_convention == BSConvention.Ry:
            unitary_tensor = torch.tensor(
                [[1, -1], [1, 1]], dtype=self.tensor_cdtype, device=self.device
            )
        elif bs_convention == BSConvention.H:
            unitary_tensor = torch.tensor(
                [[1, 1], [1, -1]], dtype=self.tensor_cdtype, device=self.device
            )
        else:
            raise NotImplementedError(
                f"BS convention : {comp._convention.name} not supported."
            )

        unitary_tensor = (
            unitary_tensor
            .unsqueeze(0)
            .repeat(self.batch_size, 1, 1)
            .to(cos_theta.device)
        )
        unitary_tensor[..., 0, 0] *= u00_mul.to(self.device) * cos_theta
        unitary_tensor[..., 0, 1] *= u01_mul.to(self.device) * sin_theta
        unitary_tensor[..., 1, 1] *= u11_mul.to(self.device) * cos_theta
        unitary_tensor[..., 1, 0] *= u10_mul.to(self.device) * sin_theta
        return unitary_tensor

    @dispatch(PS)
    def _compute_tensor(self, comp: AComponent) -> torch.Tensor:  # type: ignore[no-redef]
        """Compute tensor for Phase Shifter component.

        Applies phase noise to the phase value before constructing the phase
        unitary exp(1j * phase). This method is called by to_tensor() for each
        phase shifter in the circuit.

        Phase Noise Processing (in order):
            1. **Read Phase**: Retrieve the phase value from the PS component.
               This can be a constant, a trainable parameter, or an input-driven
               parameter. The value is converted to real dtype for noise application.

            2. **Quantization (phase_imprecision)**:
               If self._phase_imprecision > 0.0, apply deterministic STE
               quantization. The commanded phase is rounded to the nearest
               multiple of ``phase_imprecision`` with
               ``torch.round(phase / phase_imprecision) * phase_imprecision``,
               while gradients pass through the original commanded phase
               unchanged. This models finite phase resolution in hardware.
               It is not truncation. For example, ``pi / 8`` with a
               ``pi / 4`` imprecision is exactly half a step, and
               ``torch.round(0.5)`` sends it to ``0``.

            3. **Perturbation (phase_error)**:
               If self._apply_phase_error and self._phase_error > 0.0, draw
               fresh ``Uniform(-phase_error, phase_error)`` samples and add them
               to the quantized phase. If quantization is inactive, samples are
               added to the commanded phase. Samples are drawn using
               ``torch.empty_like(phase)`` to respect the phase tensor's device,
               dtype, and batch shape. Perturbations do NOT require gradients;
               they are stochastic noise, not learnable. Optimization updates
               the commanded phase, not the noise.

            4. **Complex Conversion**: Convert the noisy phase to the complex
               phase unitary exp(1j * phase).

        Gradient Flow:
            - Quantization: STE ensures gradients bypass the quantization step
            - Perturbations: .detach() on phase_error ensures no gradients flow
              through noise samples, only through the commanded phase
            - Result: dL/d(phase_commanded) is well-defined and updates the
              circuit parameters during backprop

        Device and Dtype Safety:
            - Perturbations use torch.empty_like(phase) to match device/dtype
            - Batch handling is automatic via broadcasting
            - Results are in the converter's configured complex_dtype

        Args:
            comp: PS component with phi parameter

        Returns:
            Batched 1x1 phase tensor of shape (batch_size, 1, 1) in complex dtype
        """
        if comp.param("phi").is_variable:
            (tensor_id, idx_in_tensor) = self.param_mapping[comp.param("phi").name]
            phase = self.torch_params[tensor_id][..., idx_in_tensor].to(
                dtype=self.tensor_fdtype, device=self.device
            )
        else:
            phase = torch.tensor(
                comp.param("phi")._value, dtype=self.tensor_fdtype, device=self.device
            )

        if phase.ndim == 0 and self.batch_size > 1:
            phase = phase.expand(self.batch_size)

        # Apply finite phase resolution with nearest-grid quantization. This is
        # not truncation: a phase is mapped to
        # round(phase / phase_imprecision) * phase_imprecision. torch.round
        # decides exact half-step ties, so pi/8 with a pi/4 step maps to 0.
        # The STE keeps gradients attached to the commanded phase.
        if self._phase_imprecision > 0.0:
            phase_imprecision = phase.new_tensor(self._phase_imprecision)
            phase_quantized = torch.round(phase / phase_imprecision) * phase_imprecision
            # Straight-through estimator: adding a detached delta makes the
            # forward value equal to phase_quantized, while autograd sees
            # d phase / d commanded_phase = 1 because the delta is constant.
            phase = phase + (phase_quantized - phase).detach()

        # Apply stochastic phase perturbation after quantization. Each call
        # draws fresh Uniform(-phase_error, phase_error) samples. The samples
        # are noise, not trainable parameters; gradients flow only through the
        # commanded phase value.
        if self._apply_phase_error and self._phase_error > 0.0:
            # Use torch.empty_like() to ensure perturbations follow the same
            # device, dtype, and batch structure as the phase tensor.
            phase_error = torch.empty_like(phase).uniform_(
                -self._phase_error,
                self._phase_error,
            )
            phase = phase + phase_error

        unitary_tensor = torch.exp(1j * phase.to(self.tensor_cdtype)).reshape(
            -1, 1
        )  # reshape so that in any case, we have 2 dim
        return unitary_tensor.unsqueeze(-1)  # to change shape of tensor to (b, 1, 1)
