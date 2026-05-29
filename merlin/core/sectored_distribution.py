from __future__ import annotations

from dataclasses import dataclass

import torch

from .computation_space import ComputationSpace
from ..utils.combinadics import Combinadics


@dataclass
class SectorResult:
    """One photon-number sector of a probability output. If keys are not given, the Combinatics.enumerate_states will be used."""

    tensor: torch.Tensor
    n_modes: int
    n_photons: int
    computation_space: ComputationSpace = ComputationSpace.FOCK
    keys: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        """Create the photon number to sector index map."""
        self.keys = tuple(
            Combinadics(
                scheme="fock", n=self.n_photons, m=self.n_modes
            ).enumerate_states()
        )

    def to(self, *args, **kwargs) -> SectorResult:
        """Return a new state vector moved or cast via ``torch.Tensor.to``.

        Parameters
        ----------
        *args
            Positional arguments forwarded to :meth:`torch.Tensor.to`.
        **kwargs
            Keyword arguments forwarded to :meth:`torch.Tensor.to`.

        Returns
        -------
        SectorResult
            Converted sector result.
        """
        new_tensor = self.tensor.to(*args, **kwargs)
        return SectorResult(
            new_tensor,
            self.n_modes,
            self.n_photons,
            computation_space=self.computation_space,
            keys=self.keys,
        )

    def clone(self) -> SectorResult:
        """Return a cloned ``SectorResult`` with identical metadata and normalization flag.

        Returns
        -------
        SectorResult
            Cloned sector result.
        """
        return SectorResult(
            self.tensor.clone(),
            self.n_modes,
            self.n_photons,
            computation_space=self.computation_space,
            keys=self.keys,
        )

    def detach(self) -> SectorResult:
        """Return a detached ``SectorResult`` sharing data without gradients.

        Returns
        -------
        SectorResult
            Detached sector result.
        """
        return SectorResult(
            self.tensor.detach(),
            self.n_modes,
            self.n_photons,
            computation_space=self.computation_space,
            keys=self.keys,
        )

    def requires_grad_(self, requires_grad: bool = True) -> SectorResult:
        """Set ``requires_grad`` on the underlying tensor and return self.

        Parameters
        ----------
        requires_grad : bool
            Whether gradients should be tracked.

        Returns
        -------
        SectorResult
            The updated instance.
        """
        self.tensor.requires_grad_(requires_grad)
        return self


@dataclass
class SectoredDistribution:
    """Probability output spanning multiple photon-number sectors."""

    sectors: tuple[SectorResult, ...]

    def __post_init__(self) -> None:
        """Create the photon number to sector index map."""
        self._photon_map = {
            self.sectors[i].n_photons: i for i in range(len(self.sectors))
        }

    def get_sector(self, n_photons: int) -> SectorResult:
        """Return the SectorResult associated with n_photons."""
        if n_photons not in self._photon_map.keys():
            raise ValueError("No SectorResult with that number of photons")
        return self.sectors[self._photon_map[n_photons]]

    def total_probability(self) -> torch.Tensor:
        """Returns the total probability across the sectors."""
        total_prob = 0.0
        for sector in self.sectors:
            total_prob += sector.tensor.sum()
        return total_prob

    def to(self, *args, **kwargs) -> SectoredDistribution:
        """Return a new ``SectoredDistribution`` with SectorResults moved/cast via ``torch.Tensor.to``.

        Parameters
        ----------
        *args
            Positional arguments forwarded to :meth:`torch.Tensor.to`.
        **kwargs
            Keyword arguments forwarded to :meth:`torch.Tensor.to`.

        Returns
        -------
        SectoredDistribution
            Converted sectored distribution.
        """
        new_sectors = []
        for sector in self.sectors:
            new_sectors.append(sector.to(*args, **kwargs))
        return SectoredDistribution(tuple(new_sectors))

    def clone(self) -> SectoredDistribution:
        """Return a cloned SectoredDistribution with metadata and logical performance preserved.

        Returns
        -------
        SectoredDistribution
            Cloned sectored distribution.
        """
        new_sectors = []
        for sector in self.sectors:
            new_sectors.append(sector.clone())
        return SectoredDistribution(tuple(new_sectors))

    def detach(self) -> SectoredDistribution:
        """Return a detached ``SectoredDistribution`` sharing data without gradients.

        Returns
        -------
        SectoredDistribution
            Detached sectored distribution.
        """
        new_sectors = []
        for sector in self.sectors:
            new_sectors.append(sector.detach())
        return SectoredDistribution(tuple(new_sectors))

    def requires_grad_(self, requires_grad: bool = True) -> SectoredDistribution:
        """Set ``requires_grad`` on underlying tensors and return self.

        Parameters
        ----------
        requires_grad : bool
            Whether gradients should be tracked.

        Returns
        -------
        SectoredDistribution
            The updated instance.
        """
        for sector in self.sectors:
            sector.requires_grad_(requires_grad)
        return self
