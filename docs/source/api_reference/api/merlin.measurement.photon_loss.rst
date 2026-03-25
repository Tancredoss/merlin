merlin.measurement.photon_loss module
=====================================

Photon loss utilities let you incorporate Perceval ``NoiseModel`` instances
directly inside a :class:`pcvl.Experiment`. The resulting
:class:`~merlin.algorithms.layer.QuantumLayer` automatically inserts a
:class:`~merlin.measurement.photon_loss.PhotonLossTransform` before any detector
mapping, expanding the classical basis with loss outcomes whilst keeping
probability normalisation intact.

.. automodule:: merlin.measurement.photon_loss
   :no-members:

.. currentmodule:: merlin.measurement.photon_loss

.. autoclass:: PhotonLossTransform
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: resolve_photon_loss
