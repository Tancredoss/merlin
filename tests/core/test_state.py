import pytest

from merlin.core.computation_space import ComputationSpace
from merlin.core.state import _generate_default_input_state


def test_default_input_state_uses_one_photon_per_mode_prefix():
    assert _generate_default_input_state(
        n_modes=5,
        n_photons=3,
        computation_space=ComputationSpace.FOCK,
    ) == [1, 1, 1, 0, 0]


def test_default_input_state_rejects_more_photons_than_modes():
    with pytest.raises(ValueError, match="Provide an explicit input_state"):
        _generate_default_input_state(
            n_modes=3,
            n_photons=5,
            computation_space=ComputationSpace.FOCK,
        )


def test_default_input_state_dual_rail_pattern():
    assert _generate_default_input_state(
        n_modes=6,
        n_photons=3,
        computation_space=ComputationSpace.DUAL_RAIL,
    ) == [1, 0, 1, 0, 1, 0]


def test_default_input_state_rejects_invalid_dual_rail_counts():
    with pytest.raises(ValueError, match="2 \\* n_photons"):
        _generate_default_input_state(
            n_modes=6,
            n_photons=2,
            computation_space=ComputationSpace.DUAL_RAIL,
        )
