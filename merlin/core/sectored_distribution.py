from __future__ import annotations

import torch
from dataclasses import dataclass

from .probability_distribution import ProbabilityDistribution
from .computation_space import ComputationSpace


@dataclass
class SectorResult:
    """One photon-number sector of a probability output."""

    tensor: torch.Tensor
    n_modes: int
    n_photons: int
    computation_space: ComputationSpace = ComputationSpace.FOCK
    keys: tuple[tuple[int, ...], ...] = ()

    # mirrors StateVector / ProbabilityDistribution:
    # to(), clone(), detach(), requires_grad_()

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
        StateVector
            Converted state vector.
        """
        new_tensor = self.tensor.to(*args, **kwargs)
        return SectorResult(
            new_tensor, self.n_modes, self.n_photons, computation_space=self.computation_space,keys=self.keys
        )

    def clone(self) -> SectorResult:
        """Return a cloned ``SectorResult`` with identical metadata and normalization flag.

        Returns
        -------
        StateVector
            Cloned state vector.
        """
        return SectorResult(
            self.tensor.clone(),
            self.n_modes, self.n_photons, computation_space=self.computation_space,keys=self.keys
        )

    def detach(self) -> SectorResult:
        """Return a detached ``SectorResult`` sharing data without gradients.

        Returns
        -------
        StateVector
            Detached state vector.
        """
        return SectorResult(
            self.tensor.detach(),
            self.n_modes, self.n_photons, computation_space=self.computation_space,keys=self.keys
        )

    def requires_grad_(self, requires_grad: bool = True) -> SectorResult:
        """Set ``requires_grad`` on the underlying tensor and return self.

        Parameters
        ----------
        requires_grad : bool
            Whether gradients should be tracked.

        Returns
        -------
        StateVector
            The updated instance.
        """
        self.tensor.requires_grad_(requires_grad)
        return self
    

@dataclass
class SectoredDistribution:
    """Probability output spanning multiple photon-number sectors."""
    sectors: tuple[SectorResult, ...]


     # mirrors StateVector / ProbabilityDistribution:
    # to(), clone(), detach(), requires_grad_()
    # get_sector(n_photons), photon_counts, total_probability()
