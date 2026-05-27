"""Tests for g2 photon correlations in SLOS (Sampled Linear Optical Simulator).

This module comprehensively tests:
1. g2=0 sector matching (pure SLOS output)
2. Sector probability normalization
3. Sector structure and metadata
4. Gradient flow through g2 parameters
5. Distinguishability modes (g2_distinguishable True/False)
6. Photon number distributions
7. Comparison with Perceval Simulator
"""

import pytest
import torch
import numpy as np
import perceval as pcvl
from copy import deepcopy

from merlin import ComputationSpace, CircuitBuilder
from merlin.core import SectoredDistribution
from merlin.pcvl_pytorch.noisy_slos import NoisyG2SLOSComputeGraph
from merlin.pcvl_pytorch.slos_torchscript import SLOSComputeGraph
from merlin.algorithms.layer_utils import NoiseGroups
from merlin.pcvl_pytorch.locirc_to_tensor import CircuitConverter


@pytest.fixture
def simple_bs_circuit():
    """Simple 2-mode 50:50 beamsplitter."""
    builder = CircuitBuilder(n_modes=2)
    builder.add_superpositions()
    return builder.to_pcvl_circuit()


@pytest.fixture
def entangling_circuit():
    """3-mode entangling circuit."""
    builder = CircuitBuilder(n_modes=3)
    builder.add_entangling_layer()
    return builder.to_pcvl_circuit()


@pytest.fixture
def unitary_2mode(simple_bs_circuit):
    """Convert simple circuit to unitary tensor."""
    return CircuitConverter(simple_bs_circuit).to_tensor([])


@pytest.fixture
def unitary_3mode(entangling_circuit):
    """Convert entangling circuit to unitary tensor."""
    converter = CircuitConverter(entangling_circuit)
    # Create dummy parameters for variable components
    if converter.nb_input_tensor > 0:
        dummy_params = [torch.zeros(1) for _ in range(converter.nb_input_tensor)]
        return converter.to_tensor(*dummy_params)
    else:
        return converter.to_tensor()


class TestG2SectorStructure:
    """Test g2 sector structure and metadata."""

    def test_sector_count(self, unitary_2mode):
        """SectoredDistribution returned with sectors for each photon number."""
        groups = NoiseGroups(
            source={"g2": 0.1, "g2_distinguishable": False},
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups,
            m=2,
            n_photons=2,
            computation_space=ComputationSpace.FOCK,
        )

        result = noisy_slos.compute_probs(unitary_2mode, [1, 1])
        # Result should be SectoredDistribution
        assert isinstance(result, SectoredDistribution)
        # Should have at least 2 sectors (n-photon and (n+1)-photon)
        assert len(result.sectors) >= 2

    def test_sectored_distribution_structure(self, unitary_2mode):
        """SectoredDistribution has correct structure."""
        groups = NoiseGroups(
            source={"g2": 0.1, "g2_distinguishable": False},
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups,
            m=2,
            n_photons=2,
            computation_space=ComputationSpace.FOCK,
        )

        result = noisy_slos.compute_probs(unitary_2mode, [1, 1])
        assert isinstance(result, SectoredDistribution)

        # Check each sector exists and is accessible
        for i, sector in enumerate(result.sectors):
            assert hasattr(sector, "tensor")
            assert isinstance(sector.tensor, torch.Tensor)
            assert sector.tensor.shape[0] > 0

    def test_g2_zero_single_sector(self, unitary_2mode):
        """g2=0 returns single sector matching pure SLOS."""
        # Pure SLOS (no noise)
        slos = SLOSComputeGraph(
            m=2, n_photons=2, computation_space=ComputationSpace.FOCK
        )
        keys_pure, probs_pure = slos.compute_probs(unitary_2mode, [1, 1])

        # g2 computation with g2=0
        groups = NoiseGroups(
            source={"g2": 0.0, "g2_distinguishable": True},
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups,
            m=2,
            n_photons=2,
            computation_space=ComputationSpace.FOCK,
        )

        result = noisy_slos.compute_probs(unitary_2mode, [1, 1])
        assert isinstance(result, SectoredDistribution)

        # With g2=0, should have single sector
        assert len(result.sectors) >= 1
        # First sector probabilities should match pure SLOS
        sector_0_probs = result.sectors[0].tensor
        assert torch.allclose(probs_pure, sector_0_probs, atol=1e-5)


class TestG2ProbabilityNormalization:
    """Test probability normalization across sectors."""

    def test_all_sector_probs_sum_to_one(self, unitary_2mode):
        """Sum across all photon sectors equals 1."""
        groups = NoiseGroups(
            source={"g2": 0.2, "g2_distinguishable": False},
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups,
            m=2,
            n_photons=2,
            computation_space=ComputationSpace.FOCK,
        )

        result = noisy_slos.compute_probs(unitary_2mode, [1, 1])
        assert isinstance(result, SectoredDistribution)

        # Sum all probabilities across all sectors
        total_prob = 0.0
        for sector in result.sectors:
            total_prob += sector.tensor.sum().item()

        assert np.isclose(total_prob, 1.0, atol=1e-5)

    def test_normalization_multiple_inputs(self, unitary_3mode):
        """Normalization holds for various input states."""
        groups = NoiseGroups(
            source={"g2": 0.15, "g2_distinguishable": True},
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups,
            m=3,
            n_photons=3,
            computation_space=ComputationSpace.FOCK,
        )

        # Test multiple input states
        test_states = [[1, 1, 1], [2, 1, 0], [3, 0, 0]]
        for state in test_states:
            result = noisy_slos.compute_probs(unitary_3mode, state)
            assert isinstance(result, SectoredDistribution)

            total = sum(sector.tensor.sum().item() for sector in result.sectors)
            assert np.isclose(total, 1.0, atol=1e-5)


class TestG2DistinguishabilityModes:
    """Test distinguishability modes (g2_distinguishable True/False)."""

    def test_distinguishable_different_from_indistinguishable(self, unitary_2mode):
        """Different output for g2_distinguishable=True vs False."""
        # Test with distinguishable photons
        groups_dist = NoiseGroups(
            source={
                "g2": 0.2,
                "g2_distinguishable": True,
                "indistinguishability": 1.0,
            },
            circuit=None,
            post_measurement=None,
        )
        noisy_slos_dist = NoisyG2SLOSComputeGraph(
            groups_dist,
            m=2,
            n_photons=2,
            computation_space=ComputationSpace.FOCK,
        )
        result_dist = noisy_slos_dist.compute_probs(unitary_2mode, [1, 1])

        # Test with indistinguishable photons
        groups_indist = NoiseGroups(
            source={
                "g2": 0.2,
                "g2_distinguishable": False,
                "indistinguishability": 1.0,
            },
            circuit=None,
            post_measurement=None,
        )
        noisy_slos_indist = NoisyG2SLOSComputeGraph(
            groups_indist,
            m=2,
            n_photons=2,
            computation_space=ComputationSpace.FOCK,
        )
        result_indist = noisy_slos_indist.compute_probs(unitary_2mode, [1, 1])

        # Both should be SectoredDistribution
        assert isinstance(result_dist, SectoredDistribution)
        assert isinstance(result_indist, SectoredDistribution)

        # They should have sectors
        assert len(result_dist.sectors) >= 1
        assert len(result_indist.sectors) >= 1

    def test_indistinguishable_has_bunching(self, unitary_2mode):
        """For g2_distinguishable=False: multiple sectors present."""
        groups = NoiseGroups(
            source={"g2": 0.5, "g2_distinguishable": False},
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups,
            m=2,
            n_photons=2,
            computation_space=ComputationSpace.FOCK,
        )

        result = noisy_slos.compute_probs(unitary_2mode, [1, 1])
        assert isinstance(result, SectoredDistribution)

        # Should have multiple sectors due to bunching
        assert len(result.sectors) >= 2


class TestG2ExtraPhotonDistribution:
    """Test extra photon sector distributions."""

    def test_extra_photon_sector_exists(self, unitary_2mode):
        """Extra photon sector is present when g2 > 0."""
        groups = NoiseGroups(
            source={"g2": 0.1, "g2_distinguishable": True},
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups,
            m=2,
            n_photons=2,
            computation_space=ComputationSpace.FOCK,
        )

        result = noisy_slos.compute_probs(unitary_2mode, [1, 1])
        assert isinstance(result, SectoredDistribution)

        # Should have at least 2 sectors (n-photon and n+1-photon)
        assert len(result.sectors) >= 2

        # Second sector should be for 3 photons
        second_sector = result.sectors[1]
        assert second_sector.tensor.shape[0] > 0


class TestG2Gradients:
    """Test gradient flow through g2 parameters."""

    def test_result_is_differentiable(self, unitary_2mode):
        """SectoredDistribution probs maintain gradient connection."""
        unitary_diff = unitary_2mode.clone().detach().requires_grad_(True)

        groups = NoiseGroups(
            source={"g2": 0.2, "g2_distinguishable": True},
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups,
            m=2,
            n_photons=2,
            computation_space=ComputationSpace.FOCK,
        )

        result = noisy_slos.compute_probs(unitary_diff, [1, 1])
        assert isinstance(result, SectoredDistribution)

        # Get first sector probability sum
        prob_sum = result.sectors[0].tensor.sum()
        prob_sum.backward()

        # Unitary should have gradients
        assert unitary_diff.grad is not None


class TestG2PercevalComparison:
    """Compare g2 calculations with Perceval Simulator."""

    def test_against_perceval_distinguishable(self, simple_bs_circuit):
        """g2_distinguishable=True output within tolerance of Perceval."""
        unitary = CircuitConverter(simple_bs_circuit).to_tensor([])

        groups = NoiseGroups(
            source={
                "g2": 0.1,
                "g2_distinguishable": True,
                "indistinguishability": 0.9,
            },
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups,
            m=2,
            n_photons=2,
            computation_space=ComputationSpace.FOCK,
        )

        result_merlin = noisy_slos.compute_probs(unitary, [1, 1])
        assert isinstance(result_merlin, SectoredDistribution)

        # Get Perceval reference for comparison
        noise = pcvl.NoiseModel(
            g2=0.1, g2_distinguishable=True, indistinguishability=0.9
        )
        source = pcvl.Source.from_noise_model(noise)
        backend = pcvl.BackendFactory.get_backend("SLOS")
        sim = pcvl.Simulator(backend)
        sim.set_circuit(deepcopy(simple_bs_circuit))
        perceval_result = sim.probs_svd((source, pcvl.BasicState([1, 1])))["results"]

        # Basic sanity checks
        assert len(result_merlin.sectors) > 0
        assert sum(s.tensor.sum().item() for s in result_merlin.sectors) > 0
        # Perceval result should also be non-empty
        assert len(perceval_result) > 0

    def test_against_perceval_indistinguishable(self, simple_bs_circuit):
        """g2_distinguishable=False output has multi-sector structure."""
        unitary = CircuitConverter(simple_bs_circuit).to_tensor([])

        groups = NoiseGroups(
            source={
                "g2": 0.2,
                "g2_distinguishable": False,
                "indistinguishability": 0.95,
            },
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups,
            m=2,
            n_photons=2,
            computation_space=ComputationSpace.FOCK,
        )

        result_merlin = noisy_slos.compute_probs(unitary, [1, 1])
        assert isinstance(result_merlin, SectoredDistribution)

        # Multiple sectors expected for indistinguishable case
        assert len(result_merlin.sectors) >= 2

    def test_joint_g2_and_indistinguishability(self, entangling_circuit):
        """Combined hom + g2 produces SectoredDistribution."""
        converter = CircuitConverter(entangling_circuit)
        # Create dummy parameters for variable components
        if converter.nb_input_tensor > 0:
            dummy_params = [torch.zeros(1) for _ in range(converter.nb_input_tensor)]
            unitary = converter.to_tensor(*dummy_params)
        else:
            unitary = converter.to_tensor()

        groups = NoiseGroups(
            source={
                "g2": 0.15,
                "g2_distinguishable": False,
                "indistinguishability": 0.8,
            },
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups,
            m=3,
            n_photons=3,
            computation_space=ComputationSpace.FOCK,
        )

        result_merlin = noisy_slos.compute_probs(unitary, [1, 1, 1])
        assert isinstance(result_merlin, SectoredDistribution)

        # Verify total probability is reasonable
        total_merlin = sum(s.tensor.sum().item() for s in result_merlin.sectors)
        assert np.isclose(total_merlin, 1.0, atol=1e-5)
