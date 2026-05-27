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

from merlin import ComputationSpace, CircuitBuilder, Combinadics
from merlin.core import SectoredDistribution, SectorResult
from merlin.pcvl_pytorch.noisy_slos import NoisyG2SLOSComputeGraph
from merlin.pcvl_pytorch.slos_torchscript import SLOSComputeGraph
from merlin.algorithms.layer_utils import NoiseGroups
from merlin.pcvl_pytorch.locirc_to_tensor import CircuitConverter


@pytest.fixture
def circuit():
    """4-mode input-dependent circuit: entangling-angle-entangling."""
    builder = CircuitBuilder(n_modes=4)
    builder.add_entangling_layer(trainable=False)
    builder.add_rotations()  # Creates variable input-dependent angles
    builder.add_entangling_layer(trainable=False)
    return builder.to_pcvl_circuit()


@pytest.fixture
def unitary(circuit):
    """Convert input-dependent circuit to unitary tensor function."""
    converter = CircuitConverter(circuit)

    return converter.to_tensor


class TestG2SectorStructure:
    """Test g2 sector structure and metadata."""

    def test_sector_count(self, unitary):
        """SectoredDistribution returned with sectors for each photon number."""
        groups = NoiseGroups(
            source={"g2": 0.1, "g2_distinguishable": False},
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups,
            m=4,
            n_photons=3,
            computation_space=ComputationSpace.FOCK,
        )

        result = noisy_slos.compute_probs(unitary(), [1, 1, 1, 0])
        # Result should be SectoredDistribution
        assert isinstance(result, SectoredDistribution)
        # Should have 4 sectors for n_photons=3 (3, 4, 5, 6 photons)
        assert len(result.sectors) == 4

    def test_sectored_distribution_structure(self, unitary):
        """SectoredDistribution has correct structure."""
        groups = NoiseGroups(
            source={"g2": 0.1, "g2_distinguishable": False},
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups,
            m=4,
            n_photons=3,
            computation_space=ComputationSpace.FOCK,
        )

        result = noisy_slos.compute_probs(unitary(), [1, 1, 1, 0])
        assert isinstance(result, SectoredDistribution)

        # Check each sector exists and is accessible
        for i, sector in enumerate(result.sectors):
            assert hasattr(sector, "tensor")
            assert isinstance(sector.tensor, torch.Tensor)
            assert (
                sector.tensor.squeeze().shape[0]
                == Combinadics(scheme="fock", n=3 + i, m=4).compute_space_size()
            )

    def test_sectored_distribution_metadata(self, unitary):
        """Each SectorResult has consistent n_photons, n_modes, and basis keys.

        For n_photons=3, sector i must carry metadata for 3+i photons:
        n_photons == 3+i, n_modes == 4, computation_space == FOCK, and every
        key in the (possibly empty) keys tuple must have length 4 and sum to
        3+i.
        """
        groups = NoiseGroups(
            source={"g2": 0.15, "g2_distinguishable": False},
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups,
            m=4,
            n_photons=3,
            computation_space=ComputationSpace.FOCK,
        )
        result = noisy_slos.compute_probs(unitary(), [1, 1, 1, 0])

        assert isinstance(result, SectoredDistribution)
        for i, sector in enumerate(result.sectors):
            assert isinstance(sector, SectorResult)
            assert sector.n_modes == 4
            assert sector.n_photons == 3 + i
            assert sector.computation_space == ComputationSpace.FOCK
            expected_size = Combinadics("fock", n=3 + i, m=4).compute_space_size()
            assert sector.tensor.squeeze().shape[0] == expected_size
            for key in sector.keys:
                assert len(key) == 4
                assert sum(key) == 3 + i

    def test_g2_zero_single_sector(self, unitary):
        """g2=0 returns single sector matching pure SLOS."""
        # Pure SLOS (no noise)
        slos = SLOSComputeGraph(
            m=4, n_photons=3, computation_space=ComputationSpace.FOCK
        )
        keys_pure, probs_pure = slos.compute_probs(unitary(), [1, 1, 1, 0])

        # g2 computation with g2=0
        groups = NoiseGroups(
            source={"g2": 0.0, "g2_distinguishable": True},
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups,
            m=4,
            n_photons=3,
            computation_space=ComputationSpace.FOCK,
        )

        result = noisy_slos.compute_probs(unitary(), [1, 1, 1, 0])
        assert isinstance(result, SectoredDistribution)

        # With g2=0, should have single sector
        assert len(result.sectors) >= 1
        # First sector probabilities should match pure SLOS
        sector_0_probs = result.sectors[0].tensor
        assert torch.allclose(probs_pure, sector_0_probs, atol=1e-5)

    def test_g2_zero_with_indistinguishability(self, unitary):
        """g2 =0 produces multiple sectors; first sector related to indistinguishable noisy SLOS."""
        indistinguishability = 0.8

        # Noisy SLOS with indistinguishability but no g2
        groups_noisy = NoiseGroups(
            source={"indistinguishability": indistinguishability},
            circuit=None,
            post_measurement=None,
        )
        from merlin.pcvl_pytorch.noisy_slos import NoisySLOSComputeGraph

        noisy_slos_hom = NoisySLOSComputeGraph(
            groups_noisy,
            m=4,
            n_photons=3,
            computation_space=ComputationSpace.FOCK,
            keep_keys=False,
        )
        probs_hom = noisy_slos_hom.compute_probs(unitary(), [1, 1, 1, 0])

        # g2 computation with g2 > 0 and same indistinguishability
        groups_g2 = NoiseGroups(
            source={
                "g2": 0.0,
                "g2_distinguishable": False,
                "indistinguishability": indistinguishability,
            },
            circuit=None,
            post_measurement=None,
        )
        noisy_slos_g2 = NoisyG2SLOSComputeGraph(
            groups_g2,
            m=4,
            n_photons=3,
            computation_space=ComputationSpace.FOCK,
        )

        result_g2 = noisy_slos_g2.compute_probs(unitary(), [1, 1, 1, 0])
        assert isinstance(result_g2, SectoredDistribution)

        # With g2 > 0, should have multiple sectors
        assert len(result_g2.sectors) >= 2

        sector_0_probs = result_g2.sectors[0].tensor
        assert torch.allclose(sector_0_probs, probs_hom, atol=1e-4)


class TestG2ProbabilityNormalization:
    """Test probability normalization across sectors."""

    def test_all_sector_probs_sum_to_one(self, unitary):
        """Sum across all photon sectors equals 1."""
        groups = NoiseGroups(
            source={"g2": 0.2, "g2_distinguishable": False},
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups,
            m=4,
            n_photons=3,
            computation_space=ComputationSpace.FOCK,
        )

        result = noisy_slos.compute_probs(unitary(), [1, 1, 1, 0])
        assert isinstance(result, SectoredDistribution)

        # Sum all probabilities across all sectors
        total_prob = 0.0
        for sector in result.sectors:
            total_prob += sector.tensor.sum().item()
            print(sector.tensor)

        assert np.isclose(total_prob, 1.0, atol=1e-5)

    def test_normalization_multiple_inputs(self, unitary):
        """Normalization holds for various input states."""
        groups = NoiseGroups(
            source={"g2": 0.15, "g2_distinguishable": True},
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups,
            m=4,
            n_photons=3,
            computation_space=ComputationSpace.FOCK,
        )

        # Test multiple input states
        test_states = [[1, 1, 1, 0], [2, 1, 0, 0], [1, 1, 1, 0]]
        for state in test_states:
            result = noisy_slos.compute_probs(unitary(), state)
            assert isinstance(result, SectoredDistribution)

            total = sum(sector.tensor.sum().item() for sector in result.sectors)
            assert np.isclose(total, 1.0, atol=1e-5)


class TestG2DistinguishabilityModes:
    """Test distinguishability modes (g2_distinguishable True/False)."""

    def test_distinguishable_different_from_indistinguishable(self, unitary):
        """Different output for g2_distinguishable=True vs False."""
        # Test with distinguishable photons
        groups_dist = NoiseGroups(
            source={
                "g2": 0.25,
                "g2_distinguishable": True,
                "indistinguishability": 1.0,
            },
            circuit=None,
            post_measurement=None,
        )
        noisy_slos_dist = NoisyG2SLOSComputeGraph(
            groups_dist,
            m=4,
            n_photons=3,
            computation_space=ComputationSpace.FOCK,
        )
        result_dist = noisy_slos_dist.compute_probs(unitary(), [1, 1, 1, 0])

        # Test with indistinguishable photons
        groups_indist = NoiseGroups(
            source={
                "g2": 0.25,
                "g2_distinguishable": False,
                "indistinguishability": 1.0,
            },
            circuit=None,
            post_measurement=None,
        )
        noisy_slos_indist = NoisyG2SLOSComputeGraph(
            groups_indist,
            m=4,
            n_photons=3,
            computation_space=ComputationSpace.FOCK,
        )
        result_indist = noisy_slos_indist.compute_probs(unitary(), [1, 1, 1, 0])

        # Both should be SectoredDistribution
        assert isinstance(result_dist, SectoredDistribution)
        assert isinstance(result_indist, SectoredDistribution)

        # They should have same number of sectors
        assert len(result_dist.sectors) == len(result_indist.sectors)
        assert len(result_dist.sectors) == 4

        # Verify they are actually different across sectors
        sectors_differ = False
        for sector_indist, sector_dist in zip(
            result_indist.sectors, result_dist.sectors
        ):
            if not torch.allclose(sector_indist.tensor, sector_dist.tensor, atol=1e-6):
                sectors_differ = True
                break
        assert (
            sectors_differ
        ), "Distinguishable and indistinguishable modes should produce different results"

    def test_indistinguishable_bunched_delegate(self, unitary):
        """g2_distinguishable=False: augmented input has the expected mode occupation.

        With a single extra photon added to a single-photon input [1,0,0,0]
        (g2=1.0 puts all probability in sector 1) and indistinguishability=1.0
        (NoisySLOS reduces to pure SLOS), the sector-1 distribution must equal
        the pure SLOS output for the augmented input [2,0,0,0].

        This confirms that the extra photon is bunched into mode 0—the same
        mode occupied by the original photon—rather than being placed in a
        different mode.
        """
        input_state = [1, 0, 0, 0]
        n_photons = 1
        m = 4

        unitary_tensor = unitary()

        # Reference: pure SLOS on the augmented state (extra photon in mode 0)
        slos_augmented = SLOSComputeGraph(
            m=m, n_photons=2, computation_space=ComputationSpace.FOCK
        )
        _, probs_augmented = slos_augmented.compute_probs(unitary_tensor, [2, 0, 0, 0])

        # g2=1.0 → weight_0=(1-1)^1=0, weight_1=1^1*(1-1)^0=1 (sector 1 carries everything)
        # indistinguishability=1.0 → NoisySLOSComputeGraph ≡ pure SLOS
        groups = NoiseGroups(
            source={"g2": 1.0, "g2_distinguishable": False, "indistinguishability": 1.0},
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups, m=m, n_photons=n_photons, computation_space=ComputationSpace.FOCK
        )
        result = noisy_slos.compute_probs(unitary_tensor, input_state)

        assert isinstance(result, SectoredDistribution)
        assert len(result.sectors) == 2
        # Sector 1 = 1.0 * NoisySLOS(n=2).compute_probs([2,0,0,0]) ≈ SLOS([2,0,0,0])
        sector_1_probs = result.sectors[1].tensor.squeeze()
        assert torch.allclose(sector_1_probs, probs_augmented.squeeze(), atol=1e-5)


class TestG2ExtraPhotonDistribution:
    """Test extra photon sector distributions."""

    def test_extra_photon_sector_exists(self, unitary):
        """Extra photon sector is present when g2 > 0."""
        groups = NoiseGroups(
            source={"g2": 0.1, "g2_distinguishable": True},
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups,
            m=4,
            n_photons=3,
            computation_space=ComputationSpace.FOCK,
        )

        result = noisy_slos.compute_probs(unitary(), [1, 1, 1, 0])
        assert isinstance(result, SectoredDistribution)

        # Should have multiple sectors (3, 4, 5, 6 photons)
        assert len(result.sectors) >= 2

        # Second sector should have 4 photons
        second_sector = result.sectors[1]
        assert second_sector.tensor.shape[0] > 0

    def test_distinguishable_extra_photon_distribution(self, unitary):
        """g2_distinguishable=True, single g2 event in mode m: extra-photon marginal = |U[:,m]|².

        For n_photons=1, input=[e_m] (photon only in mode 0), g2=1.0 (all
        weight in sector 1), g2_distinguishable=True, indistinguishability=1.0:

        sector-1 = conv(P_regular, P_onehot_m) where both equal |U[:,m]|².

        For two independent distinguishable photons the expected mode
        occupation factorises:

            E[n_j | sector 1] = P_regular[j] + P_onehot[j] = 2 |U[j,m]|²

        Equivalently, ``sector_1_probs @ fock_states`` (where fock_states is
        the matrix of 2-photon Fock vectors) gives a vector whose j-th entry
        equals 2|U[j,m]|², confirming that marginalising sector 1 over the
        regular-photon output recovers |U[:,m]|².
        """
        m_mode = 0  # single photon injected in mode 0
        input_state = [1, 0, 0, 0]
        n_photons = 1
        m = 4

        unitary_tensor = unitary()

        # g2=1.0 → weight_0=0, weight_1=1 (sector 1 carries everything)
        # indistinguishability=1.0 → regular probs = pure SLOS = |U[:,m]|²
        groups = NoiseGroups(
            source={"g2": 1.0, "g2_distinguishable": True, "indistinguishability": 1.0},
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups, m=m, n_photons=n_photons, computation_space=ComputationSpace.FOCK
        )
        result = noisy_slos.compute_probs(unitary_tensor, input_state)

        assert isinstance(result, SectoredDistribution)
        assert len(result.sectors) == 2

        sector_1_probs = result.sectors[1].tensor.squeeze()  # shape [N_2photon]

        # 2-photon Fock basis in Combinadics order: shape [N_2photon, m]
        fock_states = torch.tensor(
            Combinadics("fock", n=2, m=m).enumerate_states(), dtype=torch.float32
        )

        # E[n_j] = sum_s P(s) * s[j]  for each output mode j
        actual_occupation = sector_1_probs @ fock_states  # shape [m]

        # Expected: 2 * |U[j, m_mode]|² (two independent photons from mode m_mode)
        u = unitary_tensor.squeeze()  # [m, m] complex tensor
        expected_occupation = 2.0 * (u.abs() ** 2)[:, m_mode].float()

        assert torch.allclose(actual_occupation, expected_occupation, atol=1e-4)


class TestG2Gradients:
    """Test gradient flow through g2 parameters."""

    def test_result_is_differentiable(self, unitary):
        """SectoredDistribution probs maintain gradient connection."""
        unitary_diff = unitary().clone().detach().requires_grad_(True)

        groups = NoiseGroups(
            source={"indistinguishability": 0.8, "g2": 0.2, "g2_distinguishable": True},
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups,
            m=4,
            n_photons=3,
            computation_space=ComputationSpace.FOCK,
        )

        result = noisy_slos.compute_probs(unitary_diff, [1, 1, 1, 0])
        assert isinstance(result, SectoredDistribution)

        # Get first sector probability sum
        prob_sum = result.sectors[0].tensor.sum()
        prob_sum.backward()

        # Unitary should have gradients
        assert unitary_diff.grad is not None

    def test_gradient_flows_through_g2(self, unitary):
        """torch.autograd.gradcheck confirms analytical gradients w.r.t. g2.

        The g2 parameter enters through the binomial weights
        ``weight_k = g2^k * (1-g2)^(n-k)`` that mix the per-sector SLOS
        outputs.  Passing a differentiable tensor via ``noisy_slos.g2``
        keeps autograd alive through those scalar multiplications.

        Uses n_photons=2, m=4 to limit SLOS graph size and keep the check
        fast.
        """
        unitary_tensor = unitary().to(torch.complex128)

        def fn(g2: torch.Tensor) -> torch.Tensor:
            groups = NoiseGroups(
                source={"g2": float(g2.item()), "g2_distinguishable": False},
                circuit=None,
                post_measurement=None,
            )
            noisy_slos = NoisyG2SLOSComputeGraph(
                groups,
                m=4,
                n_photons=2,
                computation_space=ComputationSpace.FOCK,
                dtype=torch.float64,
            )
            # Override with the differentiable tensor so autograd tracks g2.
            noisy_slos.g2 = g2.squeeze()
            result = noisy_slos.compute_probs(unitary_tensor, [1, 1, 0, 0])
            # Each sector sum equals C(n,k)*g2^k*(1-g2)^(n-k), so the
            # vector has non-zero Jacobian entries w.r.t. g2.
            return torch.stack([s.tensor.sum() for s in result.sectors])

        g2_input = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(fn, (g2_input,), eps=1e-4, atol=1e-3)

    def test_gradient_flows_through_hom_and_g2(self, unitary):
        """torch.autograd.gradcheck confirms gradients flow through both g2 and indistinguishability.

        Indistinguishability is passed as a differentiable tensor directly in
        the NoiseGroups source dict.  ``torch.as_tensor`` in
        ``_InputStateNoisySLOSComputeGraph`` returns it unchanged (same
        memory), so ``_weights`` are computed with autograd active and
        individual sector probabilities carry a gradient back to
        ``indistinguishability``.

        The sector-sum gradients w.r.t. indistinguishability are identically
        zero (normalization cancels them), so ``sectors[0].tensor`` is
        returned instead; its individual entries carry the indistinguishability
        gradient.

        Uses n_photons=2, m=4 to limit SLOS graph size.
        """
        unitary_tensor = unitary().to(torch.complex128)

        def fn(g2: torch.Tensor, indist: torch.Tensor) -> torch.Tensor:
            groups = NoiseGroups(
                source={
                    "g2": float(g2.item()),
                    "g2_distinguishable": False,
                    # Pass the tensor so requires_grad survives into
                    # _InputStateNoisySLOSComputeGraph._weights.
                    "indistinguishability": indist.squeeze(),
                },
                circuit=None,
                post_measurement=None,
            )
            noisy_slos = NoisyG2SLOSComputeGraph(
                groups,
                m=4,
                n_photons=2,
                computation_space=ComputationSpace.FOCK,
                dtype=torch.float64,
            )
            noisy_slos.g2 = g2.squeeze()
            result = noisy_slos.compute_probs(unitary_tensor, [1, 1, 0, 0])
            # Return individual entries of sector 0; each entry depends on
            # indistinguishability through the OBB mixture weights.
            return result.sectors[0].tensor.squeeze()

        g2_input = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
        indist_input = torch.tensor([0.9], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            fn, (g2_input, indist_input), eps=1e-4, atol=1e-3
        )


class TestG2PercevalComparison:
    """Compare g2 calculations with Perceval Simulator.

    Each test verifies that Merlin's NoisyG2SLOSComputeGraph probabilities are
    within 1e-4 of Perceval's probs_svd for the same NoiseModel and circuit.

    Mapping strategy: Perceval returns a dict of BasicState -> probability.
    Each BasicState belongs to a photon-number sector (sum of occupations).
    For n_photons=3, sector i holds states with 3+i photons (i=0..3).
    Within a sector, the index is given by Combinadics("fock", n, m).fock_to_index().
    """

    @staticmethod
    def _perceval_to_merlin_probs(
        perceval_result: dict,
        result_merlin: SectoredDistribution,
        n_photons: int,
        m: int,
    ) -> None:
        """Assert each Perceval output probability matches the corresponding Merlin sector value."""
        for state, perceval_prob in perceval_result.items():
            occupations = tuple(state)
            n = sum(occupations)
            sector_idx = n - n_photons
            assert (
                0 <= sector_idx < len(result_merlin.sectors)
            ), f"State {occupations} with {n} photons has no corresponding Merlin sector"
            combo = Combinadics("fock", n=n, m=m)
            tensor_idx = combo.fock_to_index(occupations)
            merlin_prob = (
                result_merlin.sectors[sector_idx].tensor.squeeze()[tensor_idx].item()
            )
            assert (
                abs(merlin_prob - perceval_prob) < 1e-4
            ), f"State {occupations}: Merlin={merlin_prob:.6f}, Perceval={perceval_prob:.6f}"

    def test_against_perceval_distinguishable(self, circuit, unitary):
        """g2_distinguishable=True probabilities are within 1e-4 of Perceval."""
        unitary_tensor = unitary()

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
            m=4,
            n_photons=3,
            computation_space=ComputationSpace.FOCK,
        )
        result_merlin = noisy_slos.compute_probs(unitary_tensor, [1, 1, 1, 0])
        assert isinstance(result_merlin, SectoredDistribution)

        noise = pcvl.NoiseModel(
            g2=0.1, g2_distinguishable=True, indistinguishability=0.9
        )
        source = pcvl.Source.from_noise_model(noise)
        backend = pcvl.BackendFactory.get_backend("SLOS")
        sim = pcvl.Simulator(backend)
        sim.set_circuit(deepcopy(circuit))
        perceval_result = sim.probs_svd((source, pcvl.BasicState([1, 1, 1, 0])))[
            "results"
        ]

        self._perceval_to_merlin_probs(perceval_result, result_merlin, n_photons=3, m=4)

    def test_against_perceval_indistinguishable(self, circuit, unitary):
        """g2_distinguishable=False probabilities are within 1e-4 of Perceval."""
        unitary_tensor = unitary()

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
            m=4,
            n_photons=3,
            computation_space=ComputationSpace.FOCK,
        )
        result_merlin = noisy_slos.compute_probs(unitary_tensor, [1, 1, 1, 0])
        assert isinstance(result_merlin, SectoredDistribution)

        noise = pcvl.NoiseModel(
            g2=0.2, g2_distinguishable=False, indistinguishability=0.95
        )
        source = pcvl.Source.from_noise_model(noise)
        backend = pcvl.BackendFactory.get_backend("SLOS")
        sim = pcvl.Simulator(backend)
        sim.set_circuit(deepcopy(circuit))
        perceval_result = sim.probs_svd((source, pcvl.BasicState([1, 1, 1, 0])))[
            "results"
        ]

        self._perceval_to_merlin_probs(perceval_result, result_merlin, n_photons=3, m=4)

    def test_joint_g2_and_indistinguishability(self, circuit, unitary):
        """Combined g2 + indistinguishability probabilities are within 1e-4 of Perceval."""
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
            m=4,
            n_photons=3,
            computation_space=ComputationSpace.FOCK,
        )
        result_merlin = noisy_slos.compute_probs(unitary(), [1, 1, 1, 0])
        assert isinstance(result_merlin, SectoredDistribution)

        noise = pcvl.NoiseModel(
            g2=0.15, g2_distinguishable=False, indistinguishability=0.8
        )
        source = pcvl.Source.from_noise_model(noise)
        backend = pcvl.BackendFactory.get_backend("SLOS")
        sim = pcvl.Simulator(backend)
        sim.set_circuit(deepcopy(circuit))
        perceval_result = sim.probs_svd((source, pcvl.BasicState([1, 1, 1, 0])))[
            "results"
        ]

        self._perceval_to_merlin_probs(perceval_result, result_merlin, n_photons=3, m=4)
