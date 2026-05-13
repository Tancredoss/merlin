import pytest
import torch
import perceval as pcvl
import numpy as np

from merlin import ComputationSpace, Combinadics
from merlin.pcvl_pytorch.noisy_slos import (
    _InputStateNoisySLOSComputeGraph,
    NoisySLOSComputeGraph,
)
from merlin.pcvl_pytorch.slos_torchscript import SLOSComputeGraph

from merlin.algorithms.layer_utils import NoiseGroups, classify_noise_model
from merlin.pcvl_pytorch.locirc_to_tensor import CircuitConverter


@pytest.fixture
def entangling_circuit():
    # Create a 5-mode photonic processor
    circuit = pcvl.Circuit(5, name="5-Mode Fully Entangled Circuit")

    # Layer 1: Initial beam splitter network
    circuit.add((0, 1), pcvl.BS.H())
    circuit.add((1, 2), pcvl.BS.H())
    circuit.add((2, 3), pcvl.BS.H())
    circuit.add((3, 4), pcvl.BS.H())

    # Phase shifts to generate nontrivial interference
    circuit.add(0, pcvl.PS(phi=0.3))
    circuit.add(1, pcvl.PS(phi=1.1))
    circuit.add(2, pcvl.PS(phi=2.0))
    circuit.add(3, pcvl.PS(phi=0.7))
    circuit.add(4, pcvl.PS(phi=1.7))

    # Layer 2: Cross-coupling for stronger entanglement
    circuit.add((0, 1), pcvl.BS.H())
    circuit.add((1, 2), pcvl.BS.H())
    circuit.add((3, 4), pcvl.BS.H())

    # Additional phase tuning
    circuit.add(0, pcvl.PS(phi=2.4))
    circuit.add(2, pcvl.PS(phi=1.3))
    circuit.add(4, pcvl.PS(phi=0.9))

    # Final mixing layer
    circuit.add((0, 1), pcvl.BS.H())
    circuit.add((1, 2), pcvl.BS.H())
    circuit.add((2, 3), pcvl.BS.H())
    circuit.add((3, 4), pcvl.BS.H())

    return circuit


def test_identify_hom_one(entangling_circuit):

    circuit_to_analyze = entangling_circuit
    unitary = CircuitConverter(circuit_to_analyze).to_tensor([])

    # noisy computation
    groups = NoiseGroups(
        source={"indistinguishability": 1.0}, circuit=None, post_measurement=None
    )
    noisy_slos = NoisySLOSComputeGraph(
        groups, m=5, n_photons=2, computation_space=ComputationSpace.FOCK
    )

    # Noiseless computation
    slos = SLOSComputeGraph(m=5, n_photons=2, computation_space=ComputationSpace.FOCK)

    for state in Combinadics(scheme="fock", n=2, m=5).enumerate_states():
        noisy_output = noisy_slos.compute_probs(unitary, state)
        clean_output = slos.compute_probs(unitary, state)
        # Check that the output order follows combinatics
        assert noisy_output[0] == clean_output[0]
        assert (
            noisy_output[0] == Combinadics(scheme="fock", n=2, m=5).enumerate_states()
        )

        assert torch.allclose(noisy_output[1], clean_output[1])


def test_fully_distinguishable_hom_zero():
    circuit = pcvl.Circuit(2)
    circuit.add((0, 1), pcvl.BS.H())
    unitary = CircuitConverter(circuit).to_tensor([])

    noise_groups = classify_noise_model(pcvl.NoiseModel(indistinguishability=0.0))
    noisy_slos = NoisySLOSComputeGraph(
        noise_groups,
        m=2,
        n_photons=2,
        computation_space=ComputationSpace.FOCK,
        keep_keys=True,
        device=None,
        dtype=torch.float32,
    )

    keys, probs = noisy_slos.compute_probs(unitary, [1, 1])
    p = {k: probs[0, i].item() for i, k in enumerate(keys)}

    # Fully distinguishable photons -> classical split:
    # P(2,0)=0.25, P(1,1)=0.50, P(0,2)=0.25
    assert p[(2, 0)] == pytest.approx(0.25, abs=1e-6)
    assert p[(1, 1)] == pytest.approx(0.50, abs=1e-6)
    assert p[(0, 2)] == pytest.approx(0.25, abs=1e-6)
    assert sum(p.values()) == pytest.approx(1.0, abs=1e-7)


# Also works for the test_bunched_input_state and test_normalization functions
def test_against_perceval(entangling_circuit):
    circuit_to_analyze = entangling_circuit
    unitary = CircuitConverter(circuit_to_analyze).to_tensor([])

    for ind in torch.arange(0.1, 1.0, 9):
        noise_model = pcvl.NoiseModel(indistinguishability=ind.item())
        groups = classify_noise_model(noise_model)
        noisy_slos = NoisySLOSComputeGraph(
            groups, m=5, n_photons=3, computation_space=ComputationSpace.FOCK
        )

        source = pcvl.Source.from_noise_model(noise_model)

        backend = pcvl.BackendFactory.get_backend("SLOS")
        sim = pcvl.Simulator(backend)

        sim.set_circuit(circuit_to_analyze)

        for state in Combinadics(scheme="fock", n=3, m=5).enumerate_states():

            noisy_output = noisy_slos.compute_probs(unitary, state)
            # Normalisation check
            assert torch.sum(noisy_output[1]).item() == pytest.approx(1.0, abs=1e-6)

            input_state = pcvl.BasicState(state)

            perceval_probs = sim.probs_svd((source, input_state))

            for out_state, noisy_slos_probability in zip(
                noisy_output[0], noisy_output[1][0]
            ):
                assert np.allclose(
                    perceval_probs["results"][pcvl.FockState(out_state)],
                    noisy_slos_probability.item(),
                    atol=1e-4,
                )


def test_deduplication():
    """
    Bunched input partitions are not double-counted (compare with manual weight)
    """
    test = _InputStateNoisySLOSComputeGraph(
        input_state=[3, 0, 2, 1, 0],
        indistinguishability=0.5,
        computation_space=ComputationSpace.FOCK,
    )
    partitions = test._generate_obb_partition(input_state=test.input_state, order=1)
    assert torch.allclose(
        torch.unique(partitions[0], return_counts=True, dim=0)[1],
        torch.ones_like(partitions[1]),
    )
    assert torch.allclose(
        partitions[0],
        torch.tensor(
            [
                [[2, 0, 2, 1, 0], [1, 0, 0, 0, 0]],
                [[3, 0, 1, 1, 0], [0, 0, 1, 0, 0]],
                [[3, 0, 2, 0, 0], [0, 0, 0, 1, 0]],
            ],
            dtype=torch.int32,
        ),
    )
    assert torch.allclose(
        partitions[1],
        torch.tensor(
            [3, 2, 1],
        ),
    )


def test_gradient_flows_through_hom():
    assert False
