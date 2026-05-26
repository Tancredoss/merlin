# MIT License
#
# Copyright (c) 2025 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Quantum measurement sampling utilities.
"""

from collections.abc import Callable

import torch

from merlin.core.partial_measurement import DetectorTransformOutput, PartialMeasurement
from merlin.core.sectored_distribution import SectoredDistribution


class SamplingProcess:
    """Handle quantum measurement sampling with different methods.

    This class provides functionality to simulate quantum measurement noise
    by applying different sampling strategies to probability distributions.

    Parameters
    ----------
    method : str
        Sampling method to use.
    """

    def __init__(self, method: str = "multinomial"):
        """Initialize the sampling process with a specific method.

        Parameters
        ----------
        method : str
            Sampling method to use, one of ``"multinomial"``,
            ``"binomial"``, or ``"gaussian"``.

        Raises
        ------
        ValueError
            If ``method`` is not one of the valid options.
        """
        # Validate method
        self.valid_methods = ["multinomial", "binomial", "gaussian"]
        if method not in self.valid_methods:
            raise ValueError(
                f"Invalid sampling method: {method}. Valid options are: {self.valid_methods}"
            )
        self.method = method

    def pcvl_sampler(
        self, distribution: torch.Tensor, shots: int, method: str = None
    ) -> torch.Tensor:
        """Apply sampling noise to a probability distribution.

        Parameters
        ----------
        distribution : torch.Tensor
            Input probability distribution tensor.
        shots : int
            Number of measurement shots to simulate.
        method : str | None
            Sampling method to use ('multinomial', 'binomial', or 'gaussian'),
            defaults to the initialized method

        Returns
        -------
        torch.Tensor
            Noisy probability distribution after sampling.

        Raises
        ------
        ValueError
            If ``method`` is not one of the valid options.
        """
        if shots <= 0:
            return distribution

        if method is None:
            method = self.method

        if method == "multinomial":
            if distribution.dim() == 1:
                sampled_counts = torch.multinomial(
                    distribution, num_samples=shots, replacement=True
                )
                noisy_dist = torch.zeros_like(distribution)
                for idx in sampled_counts:
                    noisy_dist[idx] += 1
                return noisy_dist / shots
            else:
                batch_size = distribution.shape[0]
                noisy_dists = []
                for i in range(batch_size):
                    sampled_counts = torch.multinomial(
                        distribution[i], num_samples=shots, replacement=True
                    )
                    noisy_dist = torch.zeros_like(distribution[i])
                    for idx in sampled_counts:
                        noisy_dist[idx] += 1
                    noisy_dists.append(noisy_dist / shots)
                return torch.stack(noisy_dists)

        elif method == "binomial":
            return torch.distributions.Binomial(shots, distribution).sample() / shots

        elif method == "gaussian":
            std_dev = torch.sqrt(distribution * (1 - distribution) / shots)
            noise = torch.randn_like(distribution) * std_dev
            noisy_dist = distribution + noise
            noisy_dist = torch.clamp(noisy_dist, 0, 1)
            noisy_dist = noisy_dist / noisy_dist.sum(dim=-1, keepdim=True)
            return noisy_dist

        raise ValueError(
            f"Invalid sampling method: {method}. Valid options are: {self.valid_methods}"
        )

    def pcvl_sampler_g2(
        self, distribution: SectoredDistribution, shots: int, method: str = None
    ) -> SectoredDistribution:
        """Apply sampling noise to a SectoredDistribution.

        Parameters
        ----------
        distribution : SectoredDistribution
            Input probability distribution object.
        shots : int
            Number of measurement shots to simulate.
        method : str | None
            Sampling method to use ('multinomial', 'binomial', or 'gaussian'),
            defaults to the initialized method

        Returns
        -------
        SectoredDistribution
            Noisy probability distribution after sampling.

        Raises
        ------
        ValueError
            If ``method`` is not one of the valid options.
        """
        if shots <= 0:
            return distribution

        if method is None:
            method = self.method

        # Flattening the vectors into one sole vector
        one_dimension_vector = None
        one_dimension_counts = None
        indexes = []
        for sector in distribution.sectors:
            if one_dimension_vector is None:
                one_dimension_vector = sector.tensor
                if one_dimension_vector.dim() == 1:
                    augment_input = True
                    one_dimension_vector.unsqueeze(0)
                else:
                    augment_input = False
                indexes.append((0, one_dimension_vector.size(1)))
            else:
                if augment_input:
                    one_dimension_vector = torch.cat(
                        [one_dimension_vector, sector.tensor.unsqueeze(0)], dim=1
                    )
                else:
                    one_dimension_vector = torch.cat(
                        [one_dimension_vector, sector.tensor], dim=1
                    )
                indexes.append((indexes[-1][-1], one_dimension_vector.size(1)))

        # Sampling
        if method == "multinomial":
            batch_size = one_dimension_vector.shape[0]
            noisy_dists = []
            for i in range(batch_size):
                sampled_counts = torch.multinomial(
                    one_dimension_vector[i], num_samples=shots, replacement=True
                )
                noisy_dist = torch.zeros_like(one_dimension_vector[i])
                for idx in sampled_counts:
                    noisy_dist[idx] += 1
                noisy_dists.append(noisy_dist / shots)
            one_dimension_counts = torch.stack(noisy_dists)

        elif method == "binomial":
            one_dimension_counts = (
                torch.distributions.Binomial(shots, one_dimension_vector).sample()
                / shots
            )

        elif method == "gaussian":
            std_dev = torch.sqrt(
                one_dimension_vector * (1 - one_dimension_vector) / shots
            )
            noise = torch.randn_like(one_dimension_vector) * std_dev
            noisy_dist = one_dimension_vector + noise
            noisy_dist = torch.clamp(noisy_dist, 0, 1)
            noisy_dist = noisy_dist / noisy_dist.sum(dim=-1, keepdim=True)
            one_dimension_counts = noisy_dist

        if one_dimension_counts is None:
            raise ValueError(
                f"Invalid sampling method: {method}. Valid options are: {self.valid_methods}"
            )

        # Reformatting the output
        sectors = []
        for sector, index_splits in zip(distribution.sectors, indexes):
            sectors.append(sector.clone())
            if augment_input:
                sectors[-1].tensor = one_dimension_counts[
                    :, index_splits[0] : index_splits[1]
                ].squeeze(dim=0)
            else:
                sectors[-1].tensor = one_dimension_counts[
                    :, index_splits[0] : index_splits[1]
                ]

        return SectoredDistribution(sectors)


def partial_measurement(
    detector_output: DetectorTransformOutput,
    *,
    grouping: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> PartialMeasurement:
    """Build a PartialMeasurement from DetectorTransform(partial_measurement=True) output.

    Parameters
    ----------
    detector_output : :data:`merlin.core.partial_measurement.DetectorTransformOutput`
        Output of ``DetectorTransform(partial_measurement=True)``.
    grouping : Callable[[torch.Tensor], torch.Tensor] | None
        Optional callable used to group branch probabilities.

    Returns
    -------
    PartialMeasurement
        Partial-measurement wrapper built from the detector output.
    """
    return PartialMeasurement.from_detector_transform_output(
        detector_output, grouping=grouping
    )
