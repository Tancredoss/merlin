import pytest
import torch
import perceval as pcvl
import merlin as ML
from merlin.algorithms.layer import QuantumLayer
from merlin.algorithms import layer


def test_quantum_layer_photon_count_mismatch():
    """
    Vérifie que QuantumLayer lève une ValueError dès son initialisation (__init__)
    lorsque le nombre de photons ne correspond pas à l'état d'entrée.
    """
    
    # On s'assure que pytest intercepte la ValueError avec le message exact
    with pytest.raises(ValueError, match="number of photons doesn't fit input state"):
        
        # L'erreur DOIT être levée ici. 
        # Si elle est levée ici, le test s'arrête et passe au vert (PASSED).
        QuantumLayer(
            input_size=0,
            circuit=pcvl.Circuit(3),
            input_state=torch.tensor([1, 1, 0]),
            n_photons=1,
            measurement_strategy=ML.MeasurementStrategy.probs(),
        )