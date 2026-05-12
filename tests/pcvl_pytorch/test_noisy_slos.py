import pytest
import torch
from perceval.utils import BasicState, StateVector

from merlin import ComputationSpace
from merlin.pcvl_pytorch.noisy_slos import (
    _InputStateNoisySLOSComputeGraph,
    NoisySLOSComputeGraph,
)
from merlin.pcvl_pytorch.slos_torchscript import SLOSComputeGraph
