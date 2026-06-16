"""Tests for SectoredDistribution and SectorResult classes."""

import pytest
import torch

from merlin.core.computation_space import ComputationSpace
from merlin.core.sectored_distribution import (
    SectorResult,
    SectoredDistribution,
    clean_sectored_distribution,
)
from merlin.utils.combinadics import Combinadics


class TestSectorResult:
    """Tests for the SectorResult class."""

    def test_sector_result_creation_with_keys(self):
        """Test creating a SectorResult with explicit keys."""
        tensor = torch.tensor([0.5, 0.3, 0.2])
        keys = ((1, 0), (0, 1), (0, 0))
        sector = SectorResult(
            tensor=tensor,
            n_modes=2,
            n_photons=1,
            keys=keys,
        )
        assert sector.n_modes == 2
        assert sector.n_photons == 1
        assert sector.keys == keys
        assert torch.allclose(sector.tensor, tensor)

    def test_sector_result_creation_without_keys(self):
        """Test creating a SectorResult with auto-generated keys."""
        tensor = torch.tensor([0.5, 0.3, 0.2])
        sector = SectorResult(
            tensor=tensor,
            n_modes=3,
            n_photons=1,
        )
        assert sector.keys is not None
        assert len(sector.keys) == 3
        # Keys should be generated from Combinadics
        expected_keys = tuple(Combinadics(scheme="fock", n=1, m=3).enumerate_states())
        assert sector.keys == expected_keys

    def test_sector_result_default_computation_space(self):
        """Test that default computation space is FOCK."""
        tensor = torch.tensor([0.5, 0.5])
        sector = SectorResult(
            tensor=tensor,
            n_modes=2,
            n_photons=1,
        )
        assert sector.computation_space == ComputationSpace.FOCK

    def test_sector_result_to_cpu(self):
        """Test moving SectorResult to CPU."""
        tensor = torch.tensor([0.5, 0.3, 0.2])
        keys = ((1, 0), (0, 1), (0, 0))
        sector = SectorResult(
            tensor=tensor,
            n_modes=2,
            n_photons=1,
            keys=keys,
        )
        moved_sector = sector.to("cpu")
        assert moved_sector.tensor.device.type == "cpu"
        assert moved_sector.n_modes == sector.n_modes
        assert moved_sector.n_photons == sector.n_photons
        assert moved_sector.keys == sector.keys

    def test_sector_result_clone(self):
        """Test cloning a SectorResult."""
        tensor = torch.tensor([0.5, 0.3, 0.2])
        keys = ((1, 0), (0, 1), (0, 0))
        sector = SectorResult(
            tensor=tensor,
            n_modes=2,
            n_photons=1,
            keys=keys,
        )
        cloned = sector.clone()
        assert torch.allclose(cloned.tensor, sector.tensor)
        assert cloned.n_modes == sector.n_modes
        assert cloned.n_photons == sector.n_photons
        assert cloned.keys == sector.keys
        # Ensure it's a true clone, not a reference
        cloned.tensor[0] = 0.0
        assert not torch.allclose(cloned.tensor, sector.tensor)

    def test_sector_result_detach(self):
        """Test detaching a SectorResult."""
        tensor = torch.tensor([0.5, 0.3, 0.2], requires_grad=True)
        sector = SectorResult(
            tensor=tensor,
            n_modes=2,
            n_photons=1,
        )
        detached = sector.detach()
        assert not detached.tensor.requires_grad
        assert detached.n_modes == sector.n_modes

    def test_sector_result_requires_grad(self):
        """Test setting requires_grad on SectorResult."""
        tensor = torch.tensor([0.5, 0.3, 0.2])
        sector = SectorResult(
            tensor=tensor,
            n_modes=2,
            n_photons=1,
        )
        sector.requires_grad_(True)
        assert sector.tensor.requires_grad
        sector.requires_grad_(False)
        assert not sector.tensor.requires_grad

    def test_sector_result_batched_tensor(self):
        """Test SectorResult with batched tensors."""
        tensor = torch.rand(4, 3)  # batch size 4, 3 output states
        sector = SectorResult(
            tensor=tensor,
            n_modes=2,
            n_photons=1,
        )
        assert sector.tensor.shape == (4, 3)

    def test_sector_result_out_of_place_keys(self):
        """Test SectorResult with keys representing different photon counts."""
        # Keys represent different photon numbers than n_photons
        tensor = torch.tensor([0.2, 0.3, 0.3, 0.2])
        keys = ((2, 0), (1, 1), (1, 0), (0, 0))  # Photon counts: 2, 2, 1, 0
        sector = SectorResult(
            tensor=tensor,
            n_modes=2,
            n_photons=2,  # Sector declared as 2 photons
            keys=keys,
        )
        # Verify keys are stored as-is
        assert sector.keys == keys
        # Each key should be summed to get actual photon count
        assert sum(keys[0]) == 2
        assert sum(keys[1]) == 2
        assert sum(keys[2]) == 1
        assert sum(keys[3]) == 0


class TestSectoredDistribution:
    """Tests for the SectoredDistribution class."""

    def _create_simple_distribution(self):
        """Helper to create a simple SectoredDistribution for testing."""
        # Create two sectors: 1-photon and 2-photon
        sector1 = SectorResult(
            tensor=torch.tensor([0.5, 0.5]),
            n_modes=2,
            n_photons=1,
            keys=((1, 0), (0, 1)),
        )
        sector2 = SectorResult(
            tensor=torch.tensor([0.3, 0.3, 0.4]),
            n_modes=2,
            n_photons=2,
            keys=((2, 0), (1, 1), (0, 2)),
        )
        return SectoredDistribution(sectors=(sector1, sector2))

    def test_sectored_distribution_creation(self):
        """Test creating a SectoredDistribution."""
        dist = self._create_simple_distribution()
        assert len(dist.sectors) == 2
        assert dist.sectors[0].n_photons == 1
        assert dist.sectors[1].n_photons == 2

    def test_sectored_distribution_photon_map(self):
        """Test that photon map is created correctly."""
        dist = self._create_simple_distribution()
        assert 1 in dist._photon_map
        assert 2 in dist._photon_map
        assert dist._photon_map[1] == 0
        assert dist._photon_map[2] == 1

    def test_get_sector_valid(self):
        """Test getting a sector by photon number."""
        dist = self._create_simple_distribution()
        sector1 = dist.get_sector(1)
        assert sector1.n_photons == 1
        assert torch.allclose(sector1.tensor, torch.tensor([0.5, 0.5]))

        sector2 = dist.get_sector(2)
        assert sector2.n_photons == 2
        assert torch.allclose(sector2.tensor, torch.tensor([0.3, 0.3, 0.4]))

    def test_get_sector_invalid(self):
        """Test that getting a non-existent sector raises ValueError."""
        dist = self._create_simple_distribution()
        with pytest.raises(
            ValueError, match="No SectorResult with that number of photons"
        ):
            dist.get_sector(3)

    def test_total_probability_simple(self):
        """Test total probability calculation for simple distribution."""
        dist = self._create_simple_distribution()
        total_prob = dist.total_probability()
        # Expected: 0.5 + 0.5 + 0.3 + 0.3 + 0.4 = 2.0
        assert torch.allclose(total_prob, torch.tensor(2.0))

    def test_total_probability_normalized(self):
        """Test total probability for normalized distribution."""
        sector1 = SectorResult(
            tensor=torch.tensor([0.25, 0.25]),
            n_modes=2,
            n_photons=1,
        )
        sector2 = SectorResult(
            tensor=torch.tensor([0.25, 0.125, 0.125]),
            n_modes=2,
            n_photons=2,
        )
        dist = SectoredDistribution(sectors=(sector1, sector2))
        total_prob = dist.total_probability()
        assert torch.allclose(total_prob, torch.tensor(1.0))

    def test_total_probability_batched(self):
        """Test total probability with batched tensors."""
        sector1 = SectorResult(
            tensor=torch.tensor([[0.5, 0.5], [0.3, 0.7]]),
            n_modes=2,
            n_photons=1,
        )
        dist = SectoredDistribution(sectors=(sector1,))
        total_prob = dist.total_probability()
        # Expected: (0.5+0.5) + (0.3+0.7) = 2.0
        assert torch.allclose(total_prob, torch.tensor(2.0))

    def test_sectored_distribution_to_cpu(self):
        """Test moving SectoredDistribution to CPU."""
        dist = self._create_simple_distribution()
        moved_dist = dist.to("cpu")
        assert len(moved_dist.sectors) == 2
        assert moved_dist.sectors[0].tensor.device.type == "cpu"
        assert moved_dist.sectors[1].tensor.device.type == "cpu"

    def test_sectored_distribution_clone(self):
        """Test cloning a SectoredDistribution."""
        dist = self._create_simple_distribution()
        cloned = dist.clone()
        assert len(cloned.sectors) == len(dist.sectors)
        for orig_sector, cloned_sector in zip(dist.sectors, cloned.sectors):
            assert torch.allclose(cloned_sector.tensor, orig_sector.tensor)
            assert cloned_sector.n_photons == orig_sector.n_photons

    def test_sectored_distribution_detach(self):
        """Test detaching a SectoredDistribution."""
        sector = SectorResult(
            tensor=torch.tensor([0.5, 0.5], requires_grad=True),
            n_modes=2,
            n_photons=1,
        )
        dist = SectoredDistribution(sectors=(sector,))
        detached = dist.detach()
        assert not detached.sectors[0].tensor.requires_grad

    def test_sectored_distribution_requires_grad(self):
        """Test setting requires_grad on SectoredDistribution."""
        dist = self._create_simple_distribution()
        dist.requires_grad_(True)
        assert dist.sectors[0].tensor.requires_grad
        assert dist.sectors[1].tensor.requires_grad

    def test_to_tensor_simple_unbatched(self):
        """Test converting SectoredDistribution to a single tensor (unbatched)."""
        dist = self._create_simple_distribution()
        tensor = dist.to_tensor()
        # Expected concatenation: [0.5, 0.5, 0.3, 0.3, 0.4]
        assert tensor.shape == (5,)
        assert torch.allclose(tensor, torch.tensor([0.5, 0.5, 0.3, 0.3, 0.4]))

    def test_to_tensor_with_keys(self):
        """Test to_tensor with return_keys=True."""
        dist = self._create_simple_distribution()
        keys, tensor = dist.to_tensor(return_keys=True)
        assert len(keys) == 5
        assert keys == [(1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
        assert torch.allclose(tensor, torch.tensor([0.5, 0.5, 0.3, 0.3, 0.4]))

    def test_to_tensor_batched(self):
        """Test converting batched SectoredDistribution to tensor."""
        sector1 = SectorResult(
            tensor=torch.tensor([[0.5, 0.5], [0.3, 0.7]]),
            n_modes=2,
            n_photons=1,
        )
        sector2 = SectorResult(
            tensor=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            n_modes=2,
            n_photons=2,
        )
        dist = SectoredDistribution(sectors=(sector1, sector2))
        tensor = dist.to_tensor()
        assert tensor.shape == (2, 5)
        expected = torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0], [0.3, 0.7, 0.0, 0.0, 0.0]])
        assert torch.allclose(tensor, expected)

    def test_to_tensor_batched_with_keys(self):
        """Test to_tensor with batched data and return_keys=True."""
        sector1 = SectorResult(
            tensor=torch.tensor([[0.5, 0.5], [0.3, 0.7]]),
            n_modes=2,
            n_photons=1,
        )
        dist = SectoredDistribution(sectors=(sector1,))
        keys, tensor = dist.to_tensor(return_keys=True)
        assert len(keys) == 2
        assert tensor.shape == (2, 2)

    def test_to_tensor_preserves_multiple_batch_dimensions(self):
        """Test to_tensor preserves all leading batch dimensions."""
        sector1 = SectorResult(
            tensor=torch.arange(12, dtype=torch.float32).reshape(2, 3, 2),
            n_modes=2,
            n_photons=1,
        )
        sector2 = SectorResult(
            tensor=torch.arange(18, dtype=torch.float32).reshape(2, 3, 3),
            n_modes=2,
            n_photons=2,
        )

        tensor = SectoredDistribution(sectors=(sector1, sector2)).to_tensor()

        assert tensor.shape == (2, 3, 5)
        assert torch.allclose(tensor[..., :2], sector1.tensor)
        assert torch.allclose(tensor[..., 2:], sector2.tensor)

    def test_to_tensor_with_out_of_place_keys(self):
        """Test to_tensor with out-of-place keys (photon loss)."""
        sector2 = SectorResult(
            tensor=torch.tensor([0.2, 0.3, 0.2, 0.15, 0.15]),
            n_modes=2,
            n_photons=2,
            keys=(
                (2, 0),  # 2 photons
                (1, 1),  # 2 photons
                (0, 2),  # 2 photons
                (0, 1),  # 1 photon (out of place)
                (0, 0),  # 0 photons (out of place)
            ),
        )
        dist = SectoredDistribution(sectors=(sector2,))
        tensor = dist.to_tensor()
        assert tensor.shape == (5,)
        assert torch.allclose(tensor, torch.tensor([0.2, 0.3, 0.2, 0.15, 0.15]))

    def test_to_tensor_with_out_of_place_keys_and_return_keys(self):
        """Test to_tensor with return_keys=True and out-of-place keys."""
        sector2 = SectorResult(
            tensor=torch.tensor([0.2, 0.3, 0.2, 0.15, 0.15]),
            n_modes=2,
            n_photons=2,
            keys=(
                (2, 0),
                (1, 1),
                (0, 2),
                (0, 1),
                (0, 0),
            ),
        )
        dist = SectoredDistribution(sectors=(sector2,))
        keys, tensor = dist.to_tensor(return_keys=True)
        assert len(keys) == 5
        assert keys == [(2, 0), (1, 1), (0, 2), (0, 1), (0, 0)]
        assert torch.allclose(tensor, torch.tensor([0.2, 0.3, 0.2, 0.15, 0.15]))


class TestCleanSectoredDistribution:
    """Tests for the clean_sectored_distribution function."""

    def test_clean_simple_distribution(self):
        """Test cleaning a simple distribution without photon loss."""
        sector1 = SectorResult(
            tensor=torch.tensor([0.5, 0.5]),
            n_modes=2,
            n_photons=1,
            keys=((1, 0), (0, 1)),
        )
        dist = SectoredDistribution(sectors=(sector1,))
        cleaned = clean_sectored_distribution(dist)
        assert len(cleaned.sectors) == 1
        assert cleaned.get_sector(1).n_photons == 1

    def test_clean_distribution_with_photon_loss_simple(self):
        """Test cleaning a distribution where output states have different photon counts."""
        # Sector declared as 2 photons, but keys represent 0, 1, and 2 photons
        sector2 = SectorResult(
            tensor=torch.tensor([0.3, 0.4, 0.3]),
            n_modes=2,
            n_photons=2,
            keys=((2, 0), (1, 1), (0, 2)),
        )
        dist = SectoredDistribution(sectors=(sector2,))
        cleaned = clean_sectored_distribution(dist)
        # All states should be in the same sector since all have 2 photons
        assert len(cleaned.sectors) == 1
        assert cleaned.get_sector(2).n_photons == 2
        assert torch.allclose(
            cleaned.get_sector(2).tensor, torch.tensor([0.3, 0.4, 0.3])
        )

    def test_clean_distribution_with_photon_loss_complex(self):
        """Test cleaning a distribution where states actually lose photons."""
        # Sector with 2 photons, but some output states have 0 or 1 photons
        sector2 = SectorResult(
            tensor=torch.tensor([0.2, 0.3, 0.2, 0.15, 0.15]),
            n_modes=2,
            n_photons=2,
            keys=(
                (2, 0),  # 2 photons
                (1, 1),  # 2 photons
                (0, 2),  # 2 photons
                (0, 1),  # 1 photon (lost one)
                (0, 0),  # 0 photons (lost both)
            ),
        )
        dist = SectoredDistribution(sectors=(sector2,))
        cleaned = clean_sectored_distribution(dist)
        # Should have 3 sectors: 2-photon, 1-photon, and 0-photon
        assert len(cleaned.sectors) == 3
        # Check that probabilities are correctly separated
        sector_0 = cleaned.get_sector(0)
        sector_1 = cleaned.get_sector(1)
        sector_2 = cleaned.get_sector(2)
        assert torch.allclose(sector_0.tensor, torch.tensor([0.15]))
        # 1-photon: Combinadics order in 2 modes is (1,0), (0,1)
        # (0,1) from original maps to (0,1)=0.15, so result is [0.0, 0.15]
        assert torch.allclose(sector_1.tensor, torch.tensor([0.0, 0.15]))
        # 2-photon: Combinadics order in 2 modes is (2,0), (1,1), (0,2)
        assert torch.allclose(sector_2.tensor, torch.tensor([0.2, 0.3, 0.2]))

    def test_clean_distribution_with_photon_loss_all_counts(self):
        """Test cleaning where states span a wide range of photon counts."""
        # Start with 3 photons but output can have 0, 1, 2, or 3
        sector3 = SectorResult(
            tensor=torch.tensor([0.1, 0.15, 0.2, 0.25, 0.15, 0.05, 0.05, 0.05]),
            n_modes=2,
            n_photons=3,
            keys=(
                (3, 0),  # 3 photons
                (2, 1),  # 3 photons
                (1, 2),  # 3 photons
                (0, 3),  # 3 photons
                (1, 1),  # 2 photons (lost 1)
                (1, 0),  # 1 photon (lost 2)
                (0, 1),  # 1 photon (lost 2)
                (0, 0),  # 0 photons (lost all)
            ),
        )
        dist = SectoredDistribution(sectors=(sector3,))
        cleaned = clean_sectored_distribution(dist)
        # Should have 4 sectors: 0, 1, 2, and 3 photons
        assert len(cleaned.sectors) == 4
        # Verify each sector has correct photon count
        for n in [0, 1, 2, 3]:
            sector = cleaned.get_sector(n)
            assert sector.n_photons == n

    def test_clean_batched_distribution_with_photon_loss(self):
        """Test cleaning batched distribution with photon loss."""
        # Batched sector with photon loss
        tensor = torch.tensor([[0.3, 0.4, 0.3, 0.0, 0.0], [0.2, 0.3, 0.2, 0.15, 0.15]])
        keys = (
            (2, 0),  # 2 photons
            (1, 1),  # 2 photons
            (0, 2),  # 2 photons
            (0, 1),  # 1 photon
            (0, 0),  # 0 photons
        )
        sector2 = SectorResult(
            tensor=tensor,
            n_modes=2,
            n_photons=2,
            keys=keys,
        )
        dist = SectoredDistribution(sectors=(sector2,))
        cleaned = clean_sectored_distribution(dist)
        # Should have 3 sectors
        assert len(cleaned.sectors) == 3
        # Check batch dimensions are preserved
        for sector in cleaned.sectors:
            assert sector.tensor.shape[0] == 2  # Batch size

    def test_clean_distribution_preserves_probabilities(self):
        """Test that cleaning preserves total probability."""
        sector1 = SectorResult(
            tensor=torch.tensor([0.25, 0.75]),
            n_modes=2,
            n_photons=1,
            keys=((1, 0), (0, 1)),
        )
        dist = SectoredDistribution(sectors=(sector1,))
        original_total = dist.total_probability()
        cleaned = clean_sectored_distribution(dist)
        cleaned_total = cleaned.total_probability()
        assert torch.allclose(original_total, cleaned_total)

    def test_clean_distribution_preserves_probabilities_with_loss(self):
        """Test that cleaning preserves total probability with photon loss."""
        sector2 = SectorResult(
            tensor=torch.tensor([0.2, 0.3, 0.2, 0.15, 0.15]),
            n_modes=2,
            n_photons=2,
            keys=(
                (2, 0),
                (1, 1),
                (0, 2),
                (0, 1),
                (0, 0),
            ),
        )
        dist = SectoredDistribution(sectors=(sector2,))
        original_total = dist.total_probability()
        cleaned = clean_sectored_distribution(dist)
        cleaned_total = cleaned.total_probability()
        assert torch.allclose(original_total, cleaned_total)

    def test_clean_distribution_complex_scenario(self):
        """Test cleaning with a more complex multi-sector distribution exploring full Fock basis."""
        # Sector with 1 photon in 3 modes (full basis: 3 states)
        sector1 = SectorResult(
            tensor=torch.tensor([0.2, 0.15, 0.15]),
            n_modes=3,
            n_photons=1,
            keys=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        )
        # Sector with 2 photons in 3 modes but with photon loss
        # Full Fock basis for 2 photons: 6 states
        # Includes out-of-place keys: (1, 0, 0), (0, 1, 0), (0, 0, 1) represent 1-photon loss
        sector2 = SectorResult(
            tensor=torch.tensor(
                [0.12, 0.12, 0.12, 0.08, 0.08, 0.08, 0.1, 0.05, 0.05, 0.05]
            ),
            n_modes=3,
            n_photons=2,
            keys=(
                (2, 0, 0),
                (0, 2, 0),
                (0, 0, 2),  # 2-photon states (6 full basis)
                (1, 1, 0),
                (1, 0, 1),
                (0, 1, 1),  # 2-photon states (continued)
                (1, 0, 0),  # 1 photon - same key as sector1!
                (0, 1, 0),  # 1 photon - same key as sector1!
                (0, 0, 1),  # 1 photon - same key as sector1!
                (0, 0, 0),  # 0 photons
            ),
        )
        dist = SectoredDistribution(sectors=(sector1, sector2))
        original_total = dist.total_probability()
        cleaned = clean_sectored_distribution(dist)
        cleaned_total = cleaned.total_probability()
        assert torch.allclose(original_total, cleaned_total)
        # Should have sectors for 0, 1, and 2 photons
        assert 0 in {s.n_photons for s in cleaned.sectors}
        assert 1 in {s.n_photons for s in cleaned.sectors}
        assert 2 in {s.n_photons for s in cleaned.sectors}

        # Verify probabilities are correctly combined
        # 0-photon sector: only from sector2 (0, 0, 0) = 0.05
        sector_0 = cleaned.get_sector(0)
        assert torch.allclose(sector_0.tensor.sum(), torch.tensor(0.05))
        assert torch.allclose(sector_0.tensor, torch.tensor([0.05]))

        # 1-photon sector: Combinadics order in 3 modes is (1,0,0), (0,1,0), (0,0,1)
        # From sector1: (1,0,0)=0.2, (0,1,0)=0.15, (0,0,1)=0.15
        # From sector2: (1,0,0)=0.1, (0,1,0)=0.05, (0,0,1)=0.05
        # Combined: (1,0,0)=0.3, (0,1,0)=0.2, (0,0,1)=0.2, total=0.7
        sector_1 = cleaned.get_sector(1)
        assert torch.allclose(sector_1.tensor.sum(), torch.tensor(0.7))
        assert (
            len(sector_1.tensor) == 3
        )  # Three unique keys in 1-photon sector (full basis)
        # Verify actual values: combined probabilities in canonical order
        assert torch.allclose(sector_1.tensor, torch.tensor([0.3, 0.2, 0.2]))

        # 2-photon sector: Combinadics order in 3 modes is
        # (2,0,0), (1,1,0), (1,0,1), (0,2,0), (0,1,1), (0,0,2)
        # From sector2 keys: (2,0,0)=0.12, (0,2,0)=0.12, (0,0,2)=0.12, (1,1,0)=0.08, (1,0,1)=0.08, (0,1,1)=0.08
        # In canonical order: 0.12, 0.08, 0.08, 0.12, 0.08, 0.12, total=0.60
        sector_2 = cleaned.get_sector(2)
        assert torch.allclose(sector_2.tensor.sum(), torch.tensor(0.60))
        assert (
            len(sector_2.tensor) == 6
        )  # Six unique keys in 2-photon sector (full Fock basis)
        # Verify actual values in canonical Combinadics order
        expected_2 = torch.tensor([0.12, 0.08, 0.08, 0.12, 0.08, 0.12])
        assert torch.allclose(sector_2.tensor, expected_2)

    def test_clean_distribution_complex_scenario_batched(self):
        """Test cleaning batched distribution with overlapping keys and full Fock basis."""
        # Batched sector1: 1 photon in 3 modes (3 states full basis), 2 batch items
        sector1 = SectorResult(
            tensor=torch.tensor([[0.2, 0.15, 0.15], [0.25, 0.25, 0.2]]),
            n_modes=3,
            n_photons=1,
            keys=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        )
        # Batched sector2: 2 photons in 3 modes with photon loss
        # Includes full 2-photon basis (6 states) plus overlapping 1-photon keys and 0-photon
        sector2 = SectorResult(
            tensor=torch.tensor(
                [
                    [0.12, 0.12, 0.12, 0.08, 0.08, 0.08, 0.1, 0.05, 0.05, 0.05],
                    [0.14, 0.14, 0.14, 0.09, 0.09, 0.09, 0.08, 0.04, 0.04, 0.06],
                ]
            ),
            n_modes=3,
            n_photons=2,
            keys=(
                (2, 0, 0),
                (0, 2, 0),
                (0, 0, 2),  # 2-photon states
                (1, 1, 0),
                (1, 0, 1),
                (0, 1, 1),  # 2-photon states
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),  # 1-photon - overlaps with sector1!
                (0, 0, 0),  # 0 photons
            ),
        )
        dist = SectoredDistribution(sectors=(sector1, sector2))
        original_total = dist.total_probability()
        cleaned = clean_sectored_distribution(dist)
        cleaned_total = cleaned.total_probability()
        assert torch.allclose(original_total, cleaned_total)

        # Verify batch dimension is preserved
        for sector in cleaned.sectors:
            assert len(sector.tensor.shape) == 2
            assert sector.tensor.shape[0] == 2  # Batch size

        sector_0 = cleaned.get_sector(0)
        sector_1 = cleaned.get_sector(1)
        sector_2 = cleaned.get_sector(2)

        # Batch item 0:
        # 0-photon: sector2[0][-1] = 0.05
        # 1-photon (canonical order): sector1[0] + sector2[0][6:9] = [0.2,0.15,0.15] + [0.1,0.05,0.05] = [0.3, 0.2, 0.2] -> sum 0.7
        # 2-photon (canonical order): [0.12, 0.08, 0.08, 0.12, 0.08, 0.12] -> sum 0.60
        assert torch.allclose(sector_0.tensor[0].sum(), torch.tensor(0.05))
        assert torch.allclose(sector_1.tensor[0].sum(), torch.tensor(0.7))
        assert torch.allclose(sector_2.tensor[0].sum(), torch.tensor(0.60))
        # Check batch item 0 - verify actual values
        assert torch.allclose(sector_0.tensor[0], torch.tensor([0.05]))
        assert torch.allclose(sector_1.tensor[0], torch.tensor([0.3, 0.2, 0.2]))
        assert torch.allclose(
            sector_2.tensor[0], torch.tensor([0.12, 0.08, 0.08, 0.12, 0.08, 0.12])
        )

        # Batch item 1:
        # 0-photon: sector2[1][-1] = 0.06
        # 1-photon: sector1[1] + sector2[1][6:9] = [0.25,0.25,0.2] + [0.08,0.04,0.04] = [0.33, 0.29, 0.24] -> sum 0.86
        # 2-photon (canonical): [0.14, 0.09, 0.09, 0.14, 0.09, 0.14] -> sum 0.69
        assert torch.allclose(sector_0.tensor[1].sum(), torch.tensor(0.06))
        assert torch.allclose(sector_1.tensor[1].sum(), torch.tensor(0.86))
        assert torch.allclose(sector_2.tensor[1].sum(), torch.tensor(0.69))
        # Check batch item 1 - verify actual values
        assert torch.allclose(sector_0.tensor[1], torch.tensor([0.06]))
        assert torch.allclose(sector_1.tensor[1], torch.tensor([0.33, 0.29, 0.24]))
        assert torch.allclose(
            sector_2.tensor[1], torch.tensor([0.14, 0.09, 0.09, 0.14, 0.09, 0.14])
        )

        # Verify each sector has correct number of unique keys (full Fock basis)
        assert (
            len(sector_1.keys) == 3
        )  # Three unique 1-photon keys (full basis in 3 modes)
        assert (
            len(sector_2.keys) == 6
        )  # Six unique 2-photon keys (full basis in 3 modes)

    def test_clean_batched_complex_distribution_with_loss(self):
        """Test cleaning batched multi-sector distribution with photon loss."""
        sector1 = SectorResult(
            tensor=torch.tensor([[0.5, 0.5], [0.3, 0.7]]),
            n_modes=2,
            n_photons=1,
            keys=((1, 0), (0, 1)),
        )
        sector2 = SectorResult(
            tensor=torch.tensor(
                [[0.2, 0.2, 0.2, 0.2, 0.2], [0.15, 0.2, 0.15, 0.25, 0.25]]
            ),
            n_modes=2,
            n_photons=2,
            keys=(
                (2, 0),
                (1, 1),
                (0, 2),
                (1, 0),
                (0, 0),
            ),
        )
        dist = SectoredDistribution(sectors=(sector1, sector2))
        original_total = dist.total_probability()
        cleaned = clean_sectored_distribution(dist)
        cleaned_total = cleaned.total_probability()
        assert torch.allclose(original_total, cleaned_total)
        # All sectors should maintain batch dimension
        for sector in cleaned.sectors:
            assert len(sector.tensor.shape) == 2
            assert sector.tensor.shape[0] == 2

    def test_clean_distribution_preserves_multiple_batch_dimensions(self):
        """Test cleaning preserves all leading batch dimensions."""
        tensor = torch.arange(30, dtype=torch.float32).reshape(2, 3, 5)
        sector2 = SectorResult(
            tensor=tensor,
            n_modes=2,
            n_photons=2,
            keys=((2, 0), (1, 1), (0, 2), (1, 0), (0, 0)),
        )

        cleaned = clean_sectored_distribution(
            SectoredDistribution(sectors=(sector2,))
        )

        assert cleaned.get_sector(2).tensor.shape == (2, 3, 3)
        assert torch.allclose(cleaned.get_sector(2).tensor, tensor[..., :3])

        expected_one_photon = torch.zeros(2, 3, 2)
        expected_one_photon[..., 0] = tensor[..., 3]
        assert cleaned.get_sector(1).tensor.shape == (2, 3, 2)
        assert torch.allclose(cleaned.get_sector(1).tensor, expected_one_photon)

        assert cleaned.get_sector(0).tensor.shape == (2, 3, 1)
        assert torch.allclose(cleaned.get_sector(0).tensor[..., 0], tensor[..., 4])

    def test_clean_single_state_per_photon_number(self):
        """Test cleaning where each photon count has a single state."""
        sector3 = SectorResult(
            tensor=torch.tensor([0.1, 0.2, 0.3, 0.4]),
            n_modes=1,
            n_photons=3,
            keys=((3,), (2,), (1,), (0,)),
        )
        dist = SectoredDistribution(sectors=(sector3,))
        cleaned = clean_sectored_distribution(dist)
        # Should have 4 sectors
        assert len(cleaned.sectors) == 4
        for n, expected_prob in [(0, 0.4), (1, 0.3), (2, 0.2), (3, 0.1)]:
            assert torch.allclose(
                cleaned.get_sector(n).tensor, torch.tensor([expected_prob])
            )

    def test_clean_multiple_states_same_photon_count(self):
        """Test cleaning where multiple output states have the same photon count."""
        # 2-photon input in 3 modes, exploring full Fock basis
        # 2-photon states: (2,0,0), (0,2,0), (0,0,2), (1,1,0), (1,0,1), (0,1,1)
        # 1-photon states: (1,0,0), (0,1,0), (0,0,1)
        sector2 = SectorResult(
            tensor=torch.tensor([0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.1, 0.05, 0.05]),
            n_modes=3,
            n_photons=2,
            keys=(
                (2, 0, 0),  # 2 photons
                (0, 2, 0),  # 2 photons
                (0, 0, 2),  # 2 photons
                (1, 1, 0),  # 2 photons
                (1, 0, 1),  # 2 photons
                (0, 1, 1),  # 2 photons
                (1, 0, 0),  # 1 photon
                (0, 1, 0),  # 1 photon
                (0, 0, 1),  # 1 photon
            ),
        )
        dist = SectoredDistribution(sectors=(sector2,))
        cleaned = clean_sectored_distribution(dist)
        # Should have 2 sectors: 1 and 2 photons
        assert len(cleaned.sectors) == 2
        # 2-photon sector should have 6 states (full Fock basis for 2 photons in 3 modes)
        sector_2 = cleaned.get_sector(2)
        assert sector_2.tensor.shape[0] == 6
        assert torch.allclose(sector_2.tensor.sum(), torch.tensor(0.75))
        # 1-photon sector should have 3 states (full Fock basis for 1 photon in 3 modes)
        sector_1 = cleaned.get_sector(1)
        assert sector_1.tensor.shape[0] == 3
        assert torch.allclose(sector_1.tensor.sum(), torch.tensor(0.2))


class TestIntegration:
    """Integration tests combining multiple operations."""

    def test_full_workflow(self):
        """Test a complete workflow: create, manipulate, and convert distribution."""
        # Create sectors
        sector1 = SectorResult(
            tensor=torch.tensor([0.5, 0.5]),
            n_modes=3,
            n_photons=1,
        )
        sector2 = SectorResult(
            tensor=torch.tensor([0.2, 0.3, 0.2, 0.3]),
            n_modes=3,
            n_photons=2,
        )

        # Create distribution
        dist = SectoredDistribution(sectors=(sector1, sector2))

        # Verify structure
        assert dist.get_sector(1).n_photons == 1
        assert dist.get_sector(2).n_photons == 2

        # Check total probability
        total = dist.total_probability()
        assert total.shape == ()

        # Convert to tensor
        tensor = dist.to_tensor()
        assert tensor.shape == (6,)  # 2 + 4 states

        # Clone and verify independence
        cloned = dist.clone()
        cloned.sectors[0].tensor[0] = 0.0
        assert not torch.allclose(dist.sectors[0].tensor, cloned.sectors[0].tensor)

    def test_distributed_operations_chain(self):
        """Test chaining multiple operations on distribution."""
        sector = SectorResult(
            tensor=torch.tensor([0.5, 0.5], requires_grad=True),
            n_modes=2,
            n_photons=1,
        )
        dist = SectoredDistribution(sectors=(sector,))

        # Chain operations
        cloned = dist.clone()
        moved = cloned.to("cpu")
        detached = moved.detach()

        assert not detached.sectors[0].tensor.requires_grad
        assert detached.sectors[0].n_modes == 2

    def test_workflow_with_out_of_place_keys(self):
        """Test complete workflow with out-of-place keys (photon loss scenario)."""
        # Simulate a scenario where a 2-photon input can result in 0, 1, or 2 photon outputs
        sector = SectorResult(
            tensor=torch.tensor([0.15, 0.25, 0.15, 0.3, 0.15]),
            n_modes=2,
            n_photons=2,
            keys=(
                (2, 0),  # 2 photons
                (1, 1),  # 2 photons
                (0, 2),  # 2 photons
                (1, 0),  # 1 photon
                (0, 0),  # 0 photons
            ),
        )
        dist = SectoredDistribution(sectors=(sector,))

        # Get sector should work with any photon number in keys
        sector_2 = dist.get_sector(2)
        assert sector_2.n_photons == 2

        # Total probability should sum all output states
        total = dist.total_probability()
        assert torch.allclose(total, torch.tensor(1.0))

        # Convert to tensor and check structure
        tensor = dist.to_tensor()
        assert tensor.shape == (5,)
        keys, tensor_with_keys = dist.to_tensor(return_keys=True)
        assert len(keys) == 5

        # Clean the distribution to separate by output photon numbers
        cleaned = clean_sectored_distribution(dist)
        assert len(cleaned.sectors) == 3  # 0, 1, 2 photons
        assert torch.allclose(cleaned.total_probability(), total)

    def test_batched_workflow_with_photon_loss(self):
        """Test batched workflow with photon loss across multiple inputs."""
        # 2 batch items, 2-photon input with photon loss
        tensor = torch.tensor(
            [
                [0.1, 0.2, 0.1, 0.3, 0.3],
                [0.15, 0.25, 0.15, 0.25, 0.2],
            ]
        )
        sector = SectorResult(
            tensor=tensor,
            n_modes=2,
            n_photons=2,
            keys=(
                (2, 0),
                (1, 1),
                (0, 2),
                (1, 0),
                (0, 0),
            ),
        )
        dist = SectoredDistribution(sectors=(sector,))

        # Check batched total probability
        total = dist.total_probability()
        assert torch.allclose(total, torch.tensor(2.0))  # Both rows sum to 1

        # Clean and verify structure is maintained
        cleaned = clean_sectored_distribution(dist)
        for sector in cleaned.sectors:
            assert len(sector.tensor.shape) == 2
            assert sector.tensor.shape[0] == 2  # Batch size preserved
        assert torch.allclose(cleaned.total_probability(), total)

    def test_multi_sector_workflow_with_mixed_photon_loss(self):
        """Test workflow with multiple sectors, some with photon loss."""
        # Sector 1: 1-photon (no loss possible)
        sector1 = SectorResult(
            tensor=torch.tensor([0.3, 0.2]),
            n_modes=2,
            n_photons=1,
            keys=((1, 0), (0, 1)),
        )
        # Sector 2: 2-photon with photon loss
        sector2 = SectorResult(
            tensor=torch.tensor([0.15, 0.1, 0.15, 0.1, 0.0]),
            n_modes=2,
            n_photons=2,
            keys=(
                (2, 0),
                (1, 1),
                (0, 2),
                (1, 0),
                (0, 0),
            ),
        )
        dist = SectoredDistribution(sectors=(sector1, sector2))

        # Verify structure before cleaning
        assert len(dist.sectors) == 2
        assert dist.get_sector(1).n_photons == 1
        assert dist.get_sector(2).n_photons == 2

        # Total probability
        original_total = dist.total_probability()
        assert torch.allclose(original_total, torch.tensor(1.0))

        # Clean and verify all photon numbers are represented
        cleaned = clean_sectored_distribution(dist)
        photon_numbers = {s.n_photons for s in cleaned.sectors}
        assert 0 in photon_numbers
        assert 1 in photon_numbers
        assert 2 in photon_numbers
        assert torch.allclose(cleaned.total_probability(), original_total)
