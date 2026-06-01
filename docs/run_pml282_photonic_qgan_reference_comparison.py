r"""Local PML-282 comparison against reproduced photonic QGAN behavior.

This script is optional validation support. It imports the reproduced
``PatchGenerator.dist_to_image`` implementation from ``external/reproduced_papers``
and compares it against the current MerLin ``PhotonicGenerator`` path:

* raw Fock probabilities are produced by a MerLin ``QuantumLayer``;
* the reproduced ``PatchGenerator`` maps those raw probabilities to image
  pixels;
* the current public MerLin path uses
  ``MeasurementStrategy.probs(..., occupancy_readout=True)`` plus
  ``ImageAdapter(headwise=True, normalize_patches=True)``;
* a short Adam GAN loop verifies that the generator is trainable through normal
  PyTorch optimizer ownership.

Run from the repository root when the local reproduced-papers checkout exists:

    .\.venv\Scripts\python.exe docs\run_pml282_photonic_qgan_reference_comparison.py
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
REPRO_PATH = ROOT / "external" / "reproduced_papers" / "papers" / "photonic_QGAN"
if not REPRO_PATH.exists():
    raise SystemExit(
        "Missing external/reproduced_papers checkout; clone it locally before "
        "running this optional comparison script."
    )
sys.path.insert(0, str(REPRO_PATH))

from lib.generators import PatchGenerator  # noqa: E402

import merlin as ML  # noqa: E402


def make_head(*, occupancy_readout: bool) -> ML.QuantumLayer:
    """Build one photonic QGAN-style generator head."""
    latent_dim = 2
    builder = ML.CircuitBuilder(n_modes=3)
    builder.add_entangling_layer(trainable=True, name="var_0")
    builder.add_angle_encoding(modes=list(range(latent_dim)), name="enc")
    builder.add_entangling_layer(trainable=True, name="var_1")
    return ML.QuantumLayer(
        input_size=latent_dim,
        builder=builder,
        input_state=[1, 1, 0],
        measurement_strategy=ML.MeasurementStrategy.probs(
            computation_space=ML.ComputationSpace.FOCK,
            occupancy_readout=occupancy_readout,
        ),
    )


def build_reproduced_patch(
    output_keys: list[tuple[int, ...]],
    *,
    image_size: int,
    gen_count: int,
) -> PatchGenerator:
    """Create a PatchGenerator shell using the reproduced mapping logic.

    The public MerLin occupancy readout corresponds to the reproduced paper's
    threshold-style ``pnr=False, lossy=False`` mapping: bunched Fock states are
    collapsed into the same occupied/unoccupied bin, and no lossy-state filter
    is applied.
    """
    patch = PatchGenerator.__new__(PatchGenerator)
    patch.image_size = image_size
    patch.gen_count = gen_count
    patch.output_keys = output_keys

    reverse_map: dict[int, list[tuple[int, ...]]] = {}
    possible_outputs: list[int] = []
    mode_count = len(output_keys[0])
    for key in output_keys:
        int_state = 0
        for index, count in enumerate(key):
            if count != 0:
                int_state += 2 ** (mode_count - index)
        reverse_map.setdefault(int_state, []).append(key)
        if int_state not in possible_outputs:
            possible_outputs.append(int_state)

    patch.output_map = {}
    for index, int_state in enumerate(sorted(possible_outputs)):
        for key in reverse_map[int_state]:
            patch.output_map[key] = index

    patch.bin_count = max(patch.output_map.values()) + 1
    patch.expected_size = image_size * image_size // gen_count
    patch._mapped_col_indices = torch.tensor(
        [index for index, key in enumerate(output_keys) if key in patch.output_map],
        dtype=torch.long,
    )
    patch._idx_cpu = torch.tensor(
        [patch.output_map[key] for key in output_keys if key in patch.output_map],
        dtype=torch.long,
    )
    return patch


def compare_generator_mapping() -> None:
    """Compare current MerLin generator outputs with reproduced mapping outputs."""
    torch.manual_seed(3)
    latent_dim = 2
    image_size = 4
    gen_count = 2

    raw_template = make_head(occupancy_readout=False)
    readout_template = make_head(occupancy_readout=True)
    readout_template.load_state_dict(raw_template.state_dict())

    raw_heads = [copy.deepcopy(raw_template) for _ in range(gen_count)]
    readout_heads = [copy.deepcopy(readout_template) for _ in range(gen_count)]

    current = ML.PhotonicGenerator(
        layers=readout_heads,
        output_adapter=ML.ImageAdapter(
            shape=(1, image_size, image_size),
            headwise=True,
            normalize_patches=True,
        ),
    )
    raw_reference = ML.PhotonicGenerator(
        layers=raw_heads,
        output_adapter=nn.Identity(),
    )

    z = torch.normal(0.0, 2 * math.pi, (4, latent_dim))
    raw_outputs = list(raw_reference.measure(z).outputs)
    patch = build_reproduced_patch(
        list(raw_heads[0].output_keys),
        image_size=image_size,
        gen_count=gen_count,
    )

    reproduced = patch.dist_to_image(raw_outputs)
    merlin = current(z).reshape(z.shape[0], -1)

    torch.testing.assert_close(merlin, reproduced, atol=1e-6, rtol=1e-6)
    print(
        "mapping comparison: match=True "
        f"shape={tuple(merlin.shape)} "
        f"max_abs_diff={(merlin - reproduced).abs().max().item():.6g}"
    )


def run_adam_smoke() -> None:
    """Run a short Adam loop proving the current generator is trainable."""
    torch.manual_seed(5)
    latent_dim = 2
    generator = ML.PhotonicGenerator(
        layers=make_head(occupancy_readout=True),
        count=2,
        output_adapter=ML.ImageAdapter(
            shape=(1, 4, 4),
            headwise=True,
            normalize_patches=True,
        ),
    )
    discriminator = nn.Sequential(
        nn.Linear(16, 16),
        nn.LeakyReLU(0.2),
        nn.Linear(16, 1),
    )

    criterion = nn.BCEWithLogitsLoss()
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.01, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(generator.parameters(), lr=0.01, betas=(0.5, 0.999))

    real = torch.rand(4, 16)
    real_labels = torch.full((4,), 0.9)
    fake_labels = torch.zeros(4)
    generator_labels = torch.full((4,), 0.9)

    fake = generator(torch.normal(0.0, 2 * math.pi, (4, latent_dim))).reshape(4, -1)
    opt_d.zero_grad()
    d_loss = criterion(discriminator(real).view(-1), real_labels)
    d_loss = d_loss + criterion(discriminator(fake.detach()).view(-1), fake_labels)
    d_loss.backward()
    opt_d.step()

    opt_g.zero_grad()
    generated = generator(torch.normal(0.0, 2 * math.pi, (4, latent_dim))).reshape(
        4, -1
    )
    g_loss = criterion(discriminator(generated).view(-1), generator_labels)
    g_loss.backward()
    opt_g.step()

    has_generator_gradient = any(
        param.grad is not None and param.grad.abs().sum() > 0
        for param in generator.parameters()
    )
    if not has_generator_gradient:
        raise AssertionError("Generator parameters did not receive gradients.")
    print(
        "adam smoke: "
        f"d_loss={float(d_loss.detach()):.6f} "
        f"g_loss={float(g_loss.detach()):.6f} "
        "generator_gradient=True"
    )


def main() -> None:
    """Run local mapping and optimizer comparisons."""
    compare_generator_mapping()
    run_adam_smoke()


if __name__ == "__main__":
    main()
