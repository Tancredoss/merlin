merlin.core.encoding_space
==========================

.. automodule:: merlin.core.encoding_space
   :no-members:

.. currentmodule:: merlin.core.encoding_space

EncodingSpace
-------------

.. autoclass:: EncodingSpace
   :members: qloq, parameters, n_modes, n_photons, basis_size, resolved_modes_per_photon, logical_basis_size, embed, logical_basis_states, fock_basis_states, logical_to_fock_map, logical_to_fock_indices
   :member-order: bysource
   :show-inheritance:

Built-ins
---------

``EncodingSpace.FOCK``
   Tensor amplitudes are already in Merlin's canonical full-Fock ordering.

``EncodingSpace.UNBUNCHED``
   Logical amplitudes enumerate collision-free states and are embedded into
   the full Fock basis.

``EncodingSpace.DUAL_RAIL``
   Each logical qubit is represented by one photon shared over two modes.

Custom partitioned encodings are created with
``EncodingSpace(modes_per_photon=[...])``. QLOQ partitioned encodings are
created with :meth:`EncodingSpace.qloq`.

See :doc:`/user_guide/encoding_space` for runnable examples of each encoding.
