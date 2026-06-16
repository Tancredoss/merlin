.. _encoding_space:

===============
Encoding Spaces
===============

``EncodingSpace`` describes how a compact logical amplitude tensor maps into
Merlin's canonical Fock basis. Use it when your data naturally lives in a
smaller structured basis, but the photonic circuit still needs a full
``StateVector`` at execution time.

This page covers the input encoding API:

- :class:`~merlin.core.encoding_space.EncodingSpace`
- :meth:`~merlin.core.state_vector.StateVector.from_tensor`
- :meth:`~merlin.core.state_vector.StateVector.logical_to_fock_map`

Logical basis and Fock basis
----------------------------

Merlin simulates photonic states in the Fock basis. A Fock basis state is an
occupation tuple such as ``(1, 0, 1, 0)``, meaning one photon in mode 0 and one
photon in mode 2.

Many machine-learning inputs are easier to express in a logical basis:

- A collision-free two-photon input over four modes has only six unbunched
  states, not the full ten-state Fock basis.
- A dual-rail two-qubit state has four logical states, embedded into four
  modes with two photons.
- A categorical feature pair with three values for the first feature and two
  for the second has six logical states, embedded as one photon in each
  feature block.

``EncodingSpace`` stores that mapping. ``StateVector.from_tensor(...,
encoding=...)`` validates the logical tensor, embeds it into full Fock order,
and returns a ``StateVector`` that can be passed to
:class:`~merlin.algorithms.layer.QuantumLayer.forward`.

The usual workflow is:

.. code-block:: python

   import torch
   from merlin.core import EncodingSpace, StateVector

   encoding = EncodingSpace.DUAL_RAIL
   logical = torch.tensor([1.0, 0.0, 0.0, 1.0])

   state = StateVector.from_tensor(logical, encoding=encoding)
   mapping = state.logical_to_fock_map()

   assert state.n_modes == 4
   assert state.n_photons == 2
   assert mapping == {(0, 0): 2, (0, 1): 3, (1, 0): 5, (1, 1): 6}

Use :meth:`EncodingSpace.logical_to_fock_map()
<merlin.core.encoding_space.EncodingSpace.logical_to_fock_map>` when you want
occupation tuples, and use :meth:`StateVector.logical_to_fock_map()
<merlin.core.state_vector.StateVector.logical_to_fock_map>` when you want the
indices of those states in the stored Fock tensor.

Built-in encodings
------------------

FOCK
^^^^

``EncodingSpace.FOCK`` means the tensor is already in Merlin's full Fock
ordering. You must provide ``n_modes`` and ``n_photons`` because the same
tensor width can correspond to different physical systems.

.. code-block:: python

   import torch
   from merlin.core import EncodingSpace, StateVector

   amplitudes = torch.arange(1, 11, dtype=torch.float32)
   state = StateVector.from_tensor(
       amplitudes,
       n_modes=4,
       n_photons=2,
       encoding=EncodingSpace.FOCK,
   )

   assert state.tensor.shape[-1] == 10
   assert state.encoding is EncodingSpace.FOCK

Use this when you already have a full Fock-sized tensor, such as simulator
output from another tool.

UNBUNCHED
^^^^^^^^^

``EncodingSpace.UNBUNCHED`` accepts only collision-free logical states and
embeds them into the full Fock tensor. For four modes and two photons, the
logical tensor has six components instead of the full ten.

.. code-block:: python

   import torch
   from merlin.core import EncodingSpace, StateVector

   n_modes, n_photons = 4, 2
   logical_size = EncodingSpace.UNBUNCHED.logical_basis_size(
       n_modes=n_modes,
       n_photons=n_photons,
   )
   logical = torch.arange(1, logical_size + 1, dtype=torch.float32)

   state = StateVector.from_tensor(
       logical,
       n_modes=n_modes,
       n_photons=n_photons,
       encoding=EncodingSpace.UNBUNCHED,
   )

   mapping = EncodingSpace.UNBUNCHED.logical_to_fock_map(
       n_modes=n_modes,
       n_photons=n_photons,
   )
   assert all(max(fock_state) <= 1 for fock_state in mapping.values())
   assert state.tensor.shape[-1] == 10

Use this when your model should only put one photon in any mode, but the
downstream circuit or measurement still expects full Fock ordering.

DUAL_RAIL
^^^^^^^^^

``EncodingSpace.DUAL_RAIL`` represents each logical qubit with one photon
shared over two modes. A two-qubit logical tensor therefore has four
components, embedded into four modes with two photons.

.. code-block:: python

   import math
   import torch
   from merlin.core import EncodingSpace, StateVector

   logical = torch.tensor([1.0, 0.0, 0.0, 1.0]) / math.sqrt(2)
   state = StateVector.from_tensor(logical, encoding=EncodingSpace.DUAL_RAIL)

   assert state.n_modes == 4
   assert state.n_photons == 2
   assert EncodingSpace.DUAL_RAIL.logical_to_fock_map(
       n_modes=4,
       n_photons=2,
   )[(1, 1)] == (0, 1, 0, 1)

This is the most direct choice for qubit-like binary features or logical
two-level systems.

Partitioned encodings
---------------------

Use ``EncodingSpace(modes_per_photon=[...])`` when each photon has its own
local set of modes. This is useful for ML features with heterogeneous
cardinalities, such as a three-value category paired with a two-value category.

.. code-block:: python

   import torch
   from merlin.core import EncodingSpace, StateVector

   encoding = EncodingSpace(modes_per_photon=[3, 2])
   logical = torch.zeros(encoding.logical_basis_size())
   logical[encoding.logical_basis_states().index((2, 1))] = 1.0

   state = StateVector.from_tensor(logical, encoding=encoding)

   assert state.n_modes == 5
   assert state.n_photons == 2
   assert encoding.logical_to_fock_map()[(2, 1)] == (0, 0, 1, 0, 1)

Each entry in ``modes_per_photon`` reserves a block of modes for one photon.
The product of the block widths is the logical tensor length.

QLOQ encodings
--------------

``EncodingSpace.qloq(qubit_groups=[...])`` is a convenience constructor for
Qubit Logic on Qudits (QLOQ). A group of ``k`` logical qubits becomes one
photon delocalized over ``2**k`` modes. For example,
``qubit_groups=[2, 2]`` creates ``modes_per_photon=(4, 4)`` and a
16-component logical tensor.

QLOQ was introduced for quantum circuit compression in Lysaght et al.,
"Quantum circuit compression using qubit logic on qudits", arXiv:2411.03878.
The example below uses the same grouping idea for an ML latent state rather
than a chemistry VQE: a compact 16-component latent vector is embedded into an
8-mode, 2-photon photonic state and passed through a ``QuantumLayer``.

.. code-block:: python

   import torch
   from merlin import CircuitBuilder, MeasurementStrategy, QuantumLayer
   from merlin.core import EncodingSpace, StateVector
   from merlin.core.computation_space import ComputationSpace

   encoding = EncodingSpace.qloq(qubit_groups=[2, 2])
   latent = torch.zeros(encoding.logical_basis_size(), dtype=torch.complex64)
   latent[0] = 1.0
   latent[-1] = 1.0
   latent = latent / torch.linalg.vector_norm(latent)

   state = StateVector.from_tensor(latent, encoding=encoding)
   assert encoding.modes_per_photon == (4, 4)
   assert state.n_modes == 8
   assert state.n_photons == 2

   builder = CircuitBuilder(n_modes=state.n_modes)
   builder.add_entangling_layer(trainable=False)
   layer = QuantumLayer(
       input_size=0,
       builder=builder,
       n_photons=state.n_photons,
       measurement_strategy=MeasurementStrategy.probs(
           computation_space=ComputationSpace.FOCK,
       ),
   )

   probabilities = layer(state)
   assert probabilities.shape[-1] == layer.output_size

Migration from manual Fock tensors
----------------------------------

Older code sometimes created a full Fock-sized tensor manually, filled a small
subset of entries, and passed that full tensor to ``StateVector.from_tensor``.
That still works with ``EncodingSpace.FOCK``, but it is no longer necessary
when the data is naturally logical.

Manual full-Fock construction:

.. code-block:: python

   import torch
   from merlin.core import EncodingSpace, StateVector

   full = torch.zeros(10, dtype=torch.complex64)
   full[2] = 1.0
   full[6] = 1.0
   full = full / torch.linalg.vector_norm(full)

   state = StateVector.from_tensor(
       full,
       n_modes=4,
       n_photons=2,
       encoding=EncodingSpace.FOCK,
   )

Equivalent logical construction:

.. code-block:: python

   import torch
   from merlin.core import EncodingSpace, StateVector

   logical = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.complex64)
   logical = logical / torch.linalg.vector_norm(logical)

   state = StateVector.from_tensor(
       logical,
       encoding=EncodingSpace.DUAL_RAIL,
   )

Use the logical form when possible. It records the intended encoding in the
``StateVector``, validates the compact tensor shape, and keeps the mapping
available through ``logical_to_fock_map()``.

Testing status
--------------

The examples on this page are mirrored by
``tests/core/test_encoding_examples.py`` so that the documented workflows stay
aligned with Merlin's runtime behavior.
