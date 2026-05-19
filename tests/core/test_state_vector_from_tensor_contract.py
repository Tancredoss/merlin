import pytest
import torch

from merlin.core import EncodingSpace
from merlin.core.state_vector import StateVector
from merlin.utils.combinadics import Combinadics


def _normalized_dense(tensor: torch.Tensor) -> torch.Tensor:
    dense = tensor.to_dense() if tensor.is_sparse else tensor
    norms = torch.linalg.vector_norm(dense, dim=-1, keepdim=True)
    norms = torch.where(norms == 0, torch.ones_like(norms), norms)
    return dense / norms


@pytest.mark.parametrize("batched", [False, True])
def test_from_tensor_default_contract_dense_preserves_fock_basis_and_raw_tensor(
    batched: bool,
):
    n_modes, n_photons = 4, 2
    basis = Combinadics("fock", n_photons, n_modes)
    basis_size = basis.compute_space_size()

    if batched:
        amps = torch.arange(1, 2 * basis_size + 1, dtype=torch.float32).reshape(
            2, basis_size
        )
    else:
        amps = torch.arange(1, basis_size + 1, dtype=torch.float32)
    amps = amps.to(torch.complex64)

    sv = StateVector.from_tensor(amps, n_modes=n_modes, n_photons=n_photons)

    assert sv.n_modes == n_modes
    assert sv.n_photons == n_photons
    assert sv.encoding is EncodingSpace.FOCK
    assert sv.basis_size == basis_size
    assert list(sv.basis) == list(basis)
    assert not sv.tensor.is_sparse
    assert not sv.is_normalized
    assert torch.allclose(sv.tensor, amps)
    assert torch.allclose(sv.to_dense(), _normalized_dense(amps))
    assert sv.is_normalized


@pytest.mark.parametrize("batched", [False, True])
def test_from_tensor_default_contract_sparse_preserves_layout_until_dense_view(
    batched: bool,
):
    n_modes, n_photons = 4, 2
    basis_size = Combinadics("fock", n_photons, n_modes).compute_space_size()

    if batched:
        indices = torch.tensor([[0, 0, 1], [0, basis_size - 1, 1]])
        values = torch.tensor([1 + 0j, 2 + 0j, 3 + 0j], dtype=torch.complex64)
        amps = torch.sparse_coo_tensor(indices, values, (2, basis_size))
    else:
        indices = torch.tensor([[0, basis_size - 1]])
        values = torch.tensor([1 + 0j, 2 + 0j], dtype=torch.complex64)
        amps = torch.sparse_coo_tensor(indices, values, (basis_size,))

    sv = StateVector.from_tensor(amps, n_modes=n_modes, n_photons=n_photons)

    original = amps.coalesce()
    stored = sv.tensor.coalesce()

    assert sv.n_modes == n_modes
    assert sv.n_photons == n_photons
    assert sv.encoding is EncodingSpace.FOCK
    assert sv.tensor.is_sparse
    assert not sv.is_normalized
    assert torch.equal(stored.indices(), original.indices())
    assert torch.allclose(stored.values(), original.values())
    assert torch.allclose(sv.to_dense(), _normalized_dense(amps))
    assert sv.is_normalized


def test_from_tensor_default_contract_rejects_basis_size_mismatch():
    tensor = torch.ones(9, dtype=torch.complex64)

    with pytest.raises(ValueError, match="basis size"):
        StateVector.from_tensor(tensor, n_modes=4, n_photons=2)


def test_from_tensor_default_contract_rejects_scalar_tensor():
    tensor = torch.tensor(1.0)

    with pytest.raises(ValueError, match="at least one-dimensional"):
        StateVector.from_tensor(tensor, n_modes=1, n_photons=1)


def test_from_tensor_default_contract_explicit_dtype_and_device_are_applied():
    tensor = torch.arange(1, 7, dtype=torch.float32)

    sv = StateVector.from_tensor(
        tensor,
        n_modes=3,
        n_photons=2,
        dtype=torch.complex128,
        device=torch.device("cpu"),
    )

    assert sv.tensor.device.type == "cpu"
    assert sv.tensor.dtype == torch.complex128
    assert torch.allclose(sv.tensor, tensor.to(torch.complex128))


def test_from_tensor_default_contract_real_inputs_default_to_complex64():
    tensor = torch.arange(1, 7, dtype=torch.float64)

    sv = StateVector.from_tensor(tensor, n_modes=3, n_photons=2)

    assert sv.tensor.dtype == torch.complex64
    assert torch.allclose(sv.tensor, tensor.to(torch.complex64))


def test_from_tensor_with_dual_rail_encoding_embeds_into_fock_space():
    logical = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    fock_basis_size = Combinadics("fock", 2, 4).compute_space_size()

    sv = StateVector.from_tensor(
        logical,
        n_modes=4,
        n_photons=2,
        encoding=EncodingSpace.DUAL_RAIL,
    )

    mapping = EncodingSpace.DUAL_RAIL.logical_to_fock_indices(n_modes=4, n_photons=2)
    expected = torch.zeros(fock_basis_size, dtype=torch.complex64)
    for logical_idx, fock_idx in enumerate(mapping.values()):
        expected[fock_idx] = complex(float(logical[logical_idx].item()))

    assert sv.encoding is EncodingSpace.DUAL_RAIL
    assert sv.tensor.shape == (fock_basis_size,)
    assert torch.allclose(sv.tensor, expected)

    loss = sv.tensor.real.sum()
    loss.backward()
    assert logical.grad is not None
    assert torch.allclose(logical.grad, torch.ones_like(logical))


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"n_modes": 4},
        {"n_photons": 2},
    ],
)
def test_from_tensor_with_dual_rail_encoding_infers_dimensions(kwargs):
    logical = torch.zeros(4, dtype=torch.complex64)

    sv = StateVector.from_tensor(
        logical,
        encoding=EncodingSpace.DUAL_RAIL,
        **kwargs,
    )

    assert sv.n_modes == 4
    assert sv.n_photons == 2
    assert sv.tensor.shape == (Combinadics("fock", 2, 4).compute_space_size(),)
    assert sv.logical_to_fock_map() == EncodingSpace.DUAL_RAIL.logical_to_fock_indices(
        n_modes=4,
        n_photons=2,
    )


def test_from_tensor_with_dual_rail_encoding_rejects_non_power_of_two_inference():
    logical = torch.zeros(3, dtype=torch.complex64)

    with pytest.raises(ValueError, match="power of two"):
        StateVector.from_tensor(logical, encoding=EncodingSpace.DUAL_RAIL)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_modes": 5}, "even n_modes"),
        ({"n_modes": 5, "n_photons": 2}, "n_modes == 2 \\* n_photons"),
        ({"n_modes": 6, "n_photons": 3}, "logical basis size"),
    ],
)
def test_from_tensor_with_dual_rail_encoding_rejects_invalid_dimensions(
    kwargs,
    match,
):
    logical = torch.zeros(4, dtype=torch.complex64)

    with pytest.raises(ValueError, match=match):
        StateVector.from_tensor(
            logical,
            encoding=EncodingSpace.DUAL_RAIL,
            **kwargs,
        )


def test_from_tensor_with_custom_partitioned_encoding_embeds_into_fock_space():
    encoding = EncodingSpace(modes_per_photon=[3, 2])
    logical = torch.arange(1, 7, dtype=torch.float32)
    fock_basis_size = Combinadics("fock", 2, 5).compute_space_size()

    sv = StateVector.from_tensor(
        logical,
        n_modes=5,
        n_photons=2,
        encoding=encoding,
    )

    expected = torch.zeros(fock_basis_size, dtype=torch.complex64)
    for logical_idx, fock_idx in enumerate(encoding.logical_to_fock_indices().values()):
        expected[fock_idx] = complex(float(logical[logical_idx].item()))

    assert sv.encoding == encoding
    assert sv.tensor.shape == (fock_basis_size,)
    assert torch.allclose(sv.tensor, expected)


def test_from_tensor_with_custom_partitioned_encoding_infers_dimensions():
    encoding = EncodingSpace(modes_per_photon=[3, 2])
    logical = torch.zeros(encoding.logical_basis_size(), dtype=torch.complex64)

    sv = StateVector.from_tensor(logical, encoding=encoding)

    assert sv.n_modes == 5
    assert sv.n_photons == 2
    assert sv.tensor.shape == (Combinadics("fock", 2, 5).compute_space_size(),)


def test_from_tensor_with_qloq_encoding_infers_dimensions():
    encoding = EncodingSpace.qloq(qubit_groups=[2, 1])
    logical = torch.zeros(encoding.logical_basis_size(), dtype=torch.complex64)

    sv = StateVector.from_tensor(logical, encoding=encoding)

    assert sv.n_modes == 6
    assert sv.n_photons == 2
    assert sv.tensor.shape == (Combinadics("fock", 2, 6).compute_space_size(),)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_modes": 6}, "expects n_modes=5"),
        ({"n_photons": 3}, "expects n_photons=2"),
    ],
)
def test_from_tensor_with_partitioned_encoding_rejects_invalid_dimensions(
    kwargs,
    match,
):
    encoding = EncodingSpace(modes_per_photon=[3, 2])
    logical = torch.zeros(encoding.logical_basis_size(), dtype=torch.complex64)

    with pytest.raises(ValueError, match=match):
        StateVector.from_tensor(logical, encoding=encoding, **kwargs)


def test_from_tensor_with_batched_encoding_preserves_batch_and_autograd():
    logical = torch.arange(1, 9, dtype=torch.float32).reshape(2, 4)
    logical.requires_grad_()
    fock_basis_size = Combinadics("fock", 2, 4).compute_space_size()

    sv = StateVector.from_tensor(
        logical,
        n_modes=4,
        n_photons=2,
        encoding=EncodingSpace.DUAL_RAIL,
    )

    assert sv.encoding is EncodingSpace.DUAL_RAIL
    assert sv.tensor.shape == (2, fock_basis_size)

    sv.tensor.real.sum().backward()
    assert logical.grad is not None
    assert torch.allclose(logical.grad, torch.ones_like(logical))


def test_from_tensor_with_partitioned_encoding_preserves_sparse_layout():
    indices = torch.tensor([[0, 3]])
    values = torch.tensor([2.0 + 0j, 5.0 + 0j], dtype=torch.complex64)
    logical = torch.sparse_coo_tensor(indices, values, (4,), dtype=torch.complex64)
    fock_basis_size = Combinadics("fock", 2, 4).compute_space_size()

    sv = StateVector.from_tensor(
        logical,
        n_modes=4,
        n_photons=2,
        encoding=EncodingSpace.DUAL_RAIL,
    )

    mapping = EncodingSpace.DUAL_RAIL.logical_to_fock_indices(n_modes=4, n_photons=2)
    expected_indices = torch.tensor([[mapping[(0, 0)], mapping[(1, 1)]]])
    expected = torch.sparse_coo_tensor(
        expected_indices,
        values,
        (fock_basis_size,),
        dtype=torch.complex64,
    ).coalesce()

    assert sv.encoding is EncodingSpace.DUAL_RAIL
    assert sv.tensor.is_sparse
    assert torch.equal(sv.tensor.coalesce().indices(), expected.indices())
    assert torch.allclose(sv.tensor.coalesce().values(), expected.values())


def test_from_tensor_with_encoding_rejects_logical_basis_size_mismatch():
    tensor = torch.ones(5, dtype=torch.complex64)

    with pytest.raises(ValueError, match="logical basis size"):
        StateVector.from_tensor(
            tensor,
            n_modes=4,
            n_photons=2,
            encoding=EncodingSpace.DUAL_RAIL,
        )


def test_statevector_tensor_helpers_preserve_encoding_metadata():
    logical = torch.zeros(4, dtype=torch.complex64)
    logical[0] = 1.0
    sv = StateVector.from_tensor(
        logical,
        n_modes=4,
        n_photons=2,
        encoding=EncodingSpace.DUAL_RAIL,
    )

    assert sv.clone().encoding is EncodingSpace.DUAL_RAIL
    assert sv.detach().encoding is EncodingSpace.DUAL_RAIL
    assert sv.to(dtype=torch.complex128).encoding is EncodingSpace.DUAL_RAIL


def test_logical_to_fock_map_for_fock_encoding_is_identity_by_fock_order():
    n_modes, n_photons = 4, 2
    amplitudes = torch.zeros(
        Combinadics("fock", n_photons, n_modes).compute_space_size()
    )
    sv = StateVector.from_tensor(amplitudes, n_modes=n_modes, n_photons=n_photons)

    expected = {
        tuple(state): index
        for index, state in enumerate(Combinadics("fock", n_photons, n_modes))
    }

    assert sv.encoding is EncodingSpace.FOCK
    assert sv.logical_to_fock_map() == expected


def test_logical_to_fock_map_for_unbunched_encoding():
    n_modes, n_photons = 4, 2
    logical = torch.zeros(
        EncodingSpace.UNBUNCHED.logical_basis_size(
            n_modes=n_modes,
            n_photons=n_photons,
        )
    )
    sv = StateVector.from_tensor(
        logical,
        n_modes=n_modes,
        n_photons=n_photons,
        encoding=EncodingSpace.UNBUNCHED,
    )

    assert sv.logical_to_fock_map() == EncodingSpace.UNBUNCHED.logical_to_fock_indices(
        n_modes=n_modes,
        n_photons=n_photons,
    )


def test_logical_to_fock_map_for_dual_rail_encoding():
    n_modes, n_photons = 4, 2
    logical = torch.zeros(
        EncodingSpace.DUAL_RAIL.logical_basis_size(
            n_modes=n_modes,
            n_photons=n_photons,
        )
    )
    sv = StateVector.from_tensor(
        logical,
        n_modes=n_modes,
        n_photons=n_photons,
        encoding=EncodingSpace.DUAL_RAIL,
    )

    assert sv.logical_to_fock_map() == EncodingSpace.DUAL_RAIL.logical_to_fock_indices(
        n_modes=n_modes,
        n_photons=n_photons,
    )


def test_logical_to_fock_map_for_custom_partitioned_encoding():
    encoding = EncodingSpace(modes_per_photon=[3, 2])
    logical = torch.zeros(encoding.logical_basis_size())
    sv = StateVector.from_tensor(
        logical,
        n_modes=5,
        n_photons=2,
        encoding=encoding,
    )

    assert sv.logical_to_fock_map() == encoding.logical_to_fock_indices()
