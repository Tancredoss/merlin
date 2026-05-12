import pytest
import torch
from perceval.utils import BasicState, StateVector

from merlin import ComputationSpace
from merlin.pcvl_pytorch.noisy_slos import _InputStateNoisySLOSComputeGraph


def test_noisy_slos():
    a = _InputStateNoisySLOSComputeGraph(
        [2, 0, 1, 3, 1],
        indistinguishability=0.2,
        computation_space=ComputationSpace.UNBUNCHED,
    )
    print(a.computation_space)
    assert False
