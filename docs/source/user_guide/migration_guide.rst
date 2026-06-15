.. _user_guide_migration_guide:

Migration guide (v0.3 to v.04)
===============================

Migrating from removed ``computation_space`` argument if the ``QuantumLayer``
-------------------------------------------------------------------------------

.. warning::
   *Deprecated since version 0.4:* The use of the ``computation_space`` argument in the QuantumLayer's constructor is no longer supported as 0.4.0.
   Use the ``computation_space`` flag inside ``measurement_strategy`` instead. For example, ``MeasurementStrategy.probs(computation_space=...)``.

To define the ``computation_space`` of a :class:`~merlin.algorithms.layer.QuantumLayer`, a measurement strategy factory method must be used. For example, to use the probabilities measurement strategy with the Fock space computation space:


Computation space now lives inside the strategy.

.. code-block:: python

   # Deprecated (legacy enum + separate computation_space)
   # QuantumLayer(..., measurement_strategy=MeasurementStrategy.PROBABILITIES,
   #             computation_space=ComputationSpace.FOCK)

   # Recommended
   QuantumLayer(..., measurement_strategy=MeasurementStrategy.probs(ComputationSpace.FOCK))


Migrating from removed ``amplitude_encoding`` flag in the ``QuantumLayer``
-------------------------------------------------------------------------------

.. warning::
   The ``amplitude_encoding=True`` constructor parameter was removed in
   **0.4**. Pass a :class:`~merlin.core.state_vector.StateVector` or a complex
   ``torch.Tensor`` to ``forward()`` instead.

To use amplitude encoding in a :class:`~merlin.algorithms.layer.QuantumLayer`, you just need to pass a class:`~merlin.core.state_vector.StateVector` or a complex ``torch.Tensor``  a the forward call. Here is how to use amplitude encoding in MerLin v.0.4.

.. code-block:: python

  import torch
  from merlin.core import StateVector, EncodingSpace
  from merlin import CircuitBuilder, QuantumLayer, MeasurementStrategy, ComputationSpace

  builder = CircuitBuilder(n_modes=4)
  builder.add_entangling_layer()

  layer = QuantumLayer(
      input_size=0,
      builder=builder,
      n_photons=2,
      measurement_strategy=MeasurementStrategy.probs(
          computation_space=ComputationSpace.FOCK
      ),
  )

  # Option 1: StateVector object
  # Basic object initialization
  input_state = StateVector(
      tensor=torch.rand(1, layer.output_size),
      n_modes=4,
      n_photons=2,
      encoding=EncodingSpace.FOCK,
  )
  # From tensor method
  StateVector.from_tensor(
      tensor=torch.rand(1, layer.output_size),
      n_modes=4,
      n_photons=2,
      encoding=EncodingSpace.FOCK,
  )
  layer(input_state)

  # Option 2: complex tensor
  input_state = torch.rand(1, layer.output_size, dtype=torch.complex64)
  layer(input_state)


v.0.3 deprecations are not errors
--------------------------------------

Deprecations in the next section are now errors. Please consult the following section for other migration notes.



Migration guide (v0.2 to v.0.3)
===============================

Migrating from removed ``no_bunching``
--------------------------------------

.. warning:: *Removed in version 0.4:*
   ``no_bunching`` is removed in version 0.4. Use explicit
   ``computation_space`` configuration instead.
   
The ``no_bunching`` flag is removed.

If you are using a ``QuantumLayer`` and you need to control how Fock states are
truncated or encoded, define the ``computation_space`` inside the ``measurement_strategy``
instead. 

If you are using a Kernel, define the encoding or truncation of the Fock states in the
``computation_space`` parameter.


Map the legacy intent as follows:

- ``no_bunching=False`` → ``computation_space=ComputationSpace.FOCK`` (full Fock space)
- ``no_bunching=True`` → ``computation_space=ComputationSpace.UNBUNCHED`` (one photon per mode)
- Dual-rail encodings → ``computation_space=ComputationSpace.DUAL_RAIL``

This keeps measurement strategy selection orthogonal to simulation space configuration.

Migrating from legacy ``MeasurementStrategy``
---------------------------------------------

.. warning:: *Deprecated since version 0.3:*
   Enum-style and string access is deprecated and will be removed from `MeasurementStrategy` in v0.4.
   Use the new factory methods instead: ``MeasurementStrategy.probs(computation_space=...)``, ``MeasurementStrategy.mode_expectations(computation_space=...)``, ``MeasurementStrategy.amplitudes(computation_space=...)``.
   See this migration section for the mapping.

Old to new mappings
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Deprecated
     - Recommended replacement
   * - ``MeasurementStrategy.PROBABILITIES``
     - ``MeasurementStrategy.probs(computation_space=...)``
   * - ``MeasurementStrategy.MODE_EXPECTATIONS``
     - ``MeasurementStrategy.mode_expectations(computation_space=...)``
   * - ``MeasurementStrategy.AMPLITUDES``
     - ``MeasurementStrategy.amplitudes(computation_space=...)``
   * - ``MeasurementStrategy.NONE``
     - ``MeasurementStrategy.amplitudes(computation_space=...)``
   * - ``"PROBABILITIES"`` (string)
     - ``MeasurementStrategy.probs(computation_space=...)``

Computation space now lives inside the strategy.

.. code-block:: python

   # Deprecated (legacy enum + separate computation_space)
   # QuantumLayer(..., measurement_strategy=MeasurementStrategy.PROBABILITIES,
   #             computation_space=ComputationSpace.FOCK)

   # Recommended
   QuantumLayer(..., measurement_strategy=MeasurementStrategy.probs(ComputationSpace.FOCK))
