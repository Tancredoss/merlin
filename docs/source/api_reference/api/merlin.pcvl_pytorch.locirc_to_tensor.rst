merlin.pcvl\_pytorch.locirc\_to\_tensor module
==============================================

.. automodule:: merlin.pcvl_pytorch.locirc_to_tensor
   :no-members:

.. currentmodule:: merlin.pcvl_pytorch.locirc_to_tensor

.. autoclass:: CircuitConverter
   :members:
   :undoc-members:
   :show-inheritance:

.. note::

   ``CircuitConverter`` applies circuit phase noise directly to phase-shifter
   values before constructing the unitary tensor. ``phase_imprecision`` uses
   nearest-grid quantization with ``torch.round(phase / phase_imprecision) *
   phase_imprecision``. It does not floor or truncate phases. Exact half-step
   ties follow ``torch.round`` behavior; for example, ``pi / 8`` with a
   ``pi / 4`` imprecision step quantizes to ``0``.

   ``phase_error`` is added after any quantization and only when
   ``to_tensor(..., apply_phase_error=True)`` is used. The sampled effective
   phase is ``phase_quantized + epsilon`` with ``epsilon`` drawn from
   ``Uniform(-phase_error, phase_error)``.
