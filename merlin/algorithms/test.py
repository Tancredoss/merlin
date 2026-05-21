import perceval as pcvl
# Correction de l'import : on passe par le package algorithms
from merlin.algorithms.layer import QuantumLayer
import merlin as ML
import torch
layer = QuantumLayer(
    input_size=0,
    circuit=pcvl.Circuit(3),
    input_state=torch.tensor([1, 1, 0]),
    n_photons=1,
    measurement_strategy=ML.MeasurementStrategy.probs(),
)

# # Construction succeeds even though sum([1, 0]) == 1, not 2.
# layer()
# # Raises: IndexError: list index out of range 