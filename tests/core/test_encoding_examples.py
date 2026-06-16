import torch

from merlin import CircuitBuilder, EncodingSpace, MeasurementStrategy, QuantumLayer
from merlin.core.computation_space import ComputationSpace
from merlin.core.state_vector import StateVector
from merlin.utils.combinadics import Combinadics


def _expected_embedded_tensor(
    logical: torch.Tensor,
    encoding: EncodingSpace,
    *,
    n_modes: int | None = None,
    n_photons: int | None = None,
) -> torch.Tensor:
    """Embed a one-dimensional logical tensor using the encoding index map."""
    resolved_modes = n_modes if n_modes is not None else encoding.n_modes
    resolved_photons = n_photons if n_photons is not None else encoding.n_photons
    if resolved_modes is None or resolved_photons is None:
        raise ValueError("n_modes and n_photons must be resolved for the example.")

    fock_size = Combinadics("fock", resolved_photons, resolved_modes).compute_space_size()
    expected = torch.zeros(fock_size, dtype=torch.complex64)
    mapping = encoding.logical_to_fock_indices(
        n_modes=resolved_modes,
        n_photons=resolved_photons,
    )
    for logical_index, fock_index in enumerate(mapping.values()):
        expected[fock_index] = logical[logical_index].to(torch.complex64)
    return expected


def test_fock_example_keeps_canonical_fock_ordering():
    n_modes, n_photons = 4, 2
    fock_basis = Combinadics("fock", n_photons, n_modes)
    amplitudes = torch.arange(1, fock_basis.compute_space_size() + 1)

    state = StateVector.from_tensor(
        amplitudes,
        n_modes=n_modes,
        n_photons=n_photons,
        encoding=EncodingSpace.FOCK,
    )

    assert state.encoding is EncodingSpace.FOCK
    assert torch.allclose(state.tensor, amplitudes.to(torch.complex64))
    assert state.logical_to_fock_map() == {
        tuple(fock_state): index for index, fock_state in enumerate(fock_basis)
    }


def test_unbunched_example_embeds_only_collision_free_states():
    n_modes, n_photons = 4, 2
    logical = torch.arange(
        1,
        EncodingSpace.UNBUNCHED.logical_basis_size(
            n_modes=n_modes,
            n_photons=n_photons,
        )
        + 1,
    )

    state = StateVector.from_tensor(
        logical,
        n_modes=n_modes,
        n_photons=n_photons,
        encoding=EncodingSpace.UNBUNCHED,
    )

    expected = _expected_embedded_tensor(
        logical,
        EncodingSpace.UNBUNCHED,
        n_modes=n_modes,
        n_photons=n_photons,
    )
    assert torch.allclose(state.tensor, expected)
    assert all(
        max(fock_state) <= 1
        for fock_state in EncodingSpace.UNBUNCHED.logical_to_fock_map(
            n_modes=n_modes,
            n_photons=n_photons,
        ).values()
    )


def test_dual_rail_bell_example_maps_logical_qubits_to_modes():
    logical = torch.tensor([1.0, 0.0, 0.0, 1.0])

    state = StateVector.from_tensor(logical, encoding=EncodingSpace.DUAL_RAIL)

    assert state.n_modes == 4
    assert state.n_photons == 2
    assert EncodingSpace.DUAL_RAIL.logical_to_fock_map(
        n_modes=4,
        n_photons=2,
    ) == {
        (0, 0): (1, 0, 1, 0),
        (0, 1): (1, 0, 0, 1),
        (1, 0): (0, 1, 1, 0),
        (1, 1): (0, 1, 0, 1),
    }
    assert torch.allclose(
        state.tensor,
        _expected_embedded_tensor(
            logical,
            EncodingSpace.DUAL_RAIL,
            n_modes=4,
            n_photons=2,
        ),
    )


def test_partitioned_categorical_example_maps_heterogeneous_features():
    encoding = EncodingSpace(modes_per_photon=[3, 2])
    logical = torch.zeros(encoding.logical_basis_size())
    logical[encoding.logical_basis_states().index((2, 1))] = 1.0

    state = StateVector.from_tensor(logical, encoding=encoding)

    assert state.n_modes == 5
    assert state.n_photons == 2
    assert encoding.logical_to_fock_map()[(2, 1)] == (0, 0, 1, 0, 1)
    assert torch.allclose(
        state.tensor,
        _expected_embedded_tensor(logical, encoding),
    )


def test_qloq_latent_example_forwards_through_quantum_layer():
    encoding = EncodingSpace.qloq(qubit_groups=[2, 2])
    latent = torch.zeros(encoding.logical_basis_size(), dtype=torch.complex64)
    latent[0] = 1.0
    latent[-1] = 1.0

    state = StateVector.from_tensor(latent, encoding=encoding)

    assert encoding.modes_per_photon == (4, 4)
    assert state.n_modes == 8
    assert state.n_photons == 2

    builder = CircuitBuilder(n_modes=state.n_modes)
    builder.add_entangling_layer(trainable=False)
    layer = QuantumLayer(
        input_size=0,
        builder=builder,
        n_photons=state.n_photons,
        measurement_strategy=MeasurementStrategy.probs(
            computation_space=ComputationSpace.FOCK,
        ),
    )

    probabilities = layer(state)

    assert probabilities.shape[-1] == layer.output_size
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(1), atol=1e-5)
