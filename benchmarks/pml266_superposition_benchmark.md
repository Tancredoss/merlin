# PML-266 Superposition Streaming Benchmark

Date: 2026-04-30

This benchmark compares the pre-PML-266 implementation on `origin/main`
against the current PML-266 working tree. The benchmark script is
`benchmarks/benchmark_superposition_streaming.py`.

## Compared revisions

| Label | Revision | Notes |
| --- | --- | --- |
| `origin-main-pre-pml266` | `5b36dc0f5ec877598ce65072f516b56800937ebd` | Base branch before PML-266 |
| `pml266-current-working-tree` | `0c49225382f1bd8ae2b4f3465f6eb4a1d4c1483f` + local changes | PML-266 plus the sparse `StateVector` no-densification fix |

Raw JSON results:

- `benchmarks/results/origin-main-pre-pml266.json`
- `benchmarks/results/pml266-current-working-tree.json`

## Internal path being checked

The intended user path is still ordinary dense amplitude tensors. Users do not
need to build PyTorch sparse COO tensors to get the memory benefit.

On the current branch, dense tensors, dense `StateVector` inputs, sparse
tensors, and sparse `StateVector` inputs are all normalized into the same
internal active-support representation before propagation:

```text
active basis indices + compact coefficient tensor [batch_size, nnz]
```

The simulator then propagates only those active basis states in chunks and
accumulates directly into the final dense output amplitude tensor. The dense
whole-support table is not part of the current execution path.

The benchmark intentionally passes dense tensors with sparse support because
that is the normal user-facing input shape and it can run unchanged on both
`origin/main` and the current branch. The allocation recorder verifies that the
current branch still reduces that dense input to active support internally,
because the largest tracked allocation follows `[parameter_batch,
output_states, chunk_size]` instead of `[parameter_batch, input_states,
output_states]`.

## Method

The benchmark uses a `QuantumLayer` with a deterministic rectangular Perceval
`GenericInterferometer`, Fock-space amplitude output, `complex64` amplitudes,
and `simultaneous_processes=8`. Each input vector is dense for compatibility
with both revisions, but only 16 entries are non-zero. This isolates the ticket
behavior: both implementations only need to simulate 16 active input
components, but the old implementation still materializes a dense
whole-support table.

Each case ran 1 warmup and 5 measured forwards. The script also wraps
`torch.zeros` as used by `merlin.core.process` and records whether a dense
whole-support allocation appears.

The comparison runner also stores the full complex output vector for each case
and compares the current branch against the baseline with `rtol=1e-5` and
`atol=1e-6`. This run passed for all 3 cases. The largest complex difference
was `3.07195e-08` at `fock_m28_p3_16nz_chunk8[1135]`, below the allowed
`2.54594e-06`.

Environment:

- WSL2 Linux
- Python `3.12.3`
- PyTorch `2.10.0+cu128`

## Results

| Case | Basis / output states | Old mean | New mean | Speedup | Old max whole-support allocation | New max tracked allocation | Allocation reduction | Old max RSS | New max RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fock_m16_p3_16nz_chunk8` | 816 | 0.0227 s | 0.0199 s | 1.1x | 5.08 MiB (`[1, 816, 816]`) | 0.050 MiB (`[1, 816, 8]`) | 102x | 696.8 MiB | 685.9 MiB |
| `fock_m28_p3_16nz_chunk8` | 4,060 | 0.1593 s | 0.0543 s | 2.9x | 125.76 MiB (`[1, 4060, 4060]`) | 0.248 MiB (`[1, 4060, 8]`) | 508x | 977.0 MiB | 720.4 MiB |
| `fock_m32_p3_16nz_chunk8` | 5,984 | 0.3350 s | 0.0765 s | 4.4x | 273.20 MiB (`[1, 5984, 5984]`) | 0.365 MiB (`[1, 5984, 8]`) | 748x | 1,296.1 MiB | 744.0 MiB |

## Findings

The base implementation allocates one dense whole-support tensor per forward
with shape `[parameter_batch, input_basis_size, output_basis_size]`. In these
Fock-space cases the input and output basis sizes are the same, so this grows
quadratically with the basis size.

The PML-266 implementation does not allocate the whole-support table. The
largest tracked allocation follows `[parameter_batch, output_basis_size,
chunk_size]`, matching the intended `chunk_size * num_output_states` scaling.
That is the observable sign that dense user input was reduced to compact active
support before simulation.

Wall time improves as the basis grows because the old code allocates and
normalizes the whole dense support table even though only 16 input amplitudes
are non-zero. The largest measured case went from 0.3350 s to 0.0765 s per
forward on this machine while producing matching amplitudes.

RSS is process-wide and less precise than the allocation instrumentation, but
it shows the same trend: the largest case peaked at about 1,296.1 MiB before
PML-266 and 744.0 MiB on the current branch.

This benchmark intentionally uses dense tensors with sparse support so the same
input can run on both revisions. The current branch converts that dense support
to compact active support internally. It also preserves raw sparse tensor and
sparse `StateVector` inputs; the base branch does not preserve the sparse
`StateVector` route.

## Reproduction

Generic comparison runner:

```bash
cd /mnt/c/Users/BenjaminSTOTT/PycharmProjects/merlin
source /home/benjamin/.virtualenvs/merlin/bin/activate
benchmarks/run_superposition_streaming_comparison.sh \
  --baseline-ref origin/main \
  --candidate-label pml266-current-working-tree \
  --baseline-label origin-main-pre-pml266 \
  --runs 5 \
  --warmups 1
```

The runner creates a temporary baseline worktree, runs
`benchmarks/benchmark_superposition_streaming.py` against both revisions, and
writes one JSON file per run into `benchmarks/results/`. By default it also
compares the full complex output vectors. Pass `--no-compare-output` to collect
timings and allocation data without the output comparison.

Current PML-266 working tree:

```bash
cd /mnt/c/Users/BenjaminSTOTT/PycharmProjects/merlin
source /home/benjamin/.virtualenvs/merlin/bin/activate
PYTHONPATH=$PWD python benchmarks/benchmark_superposition_streaming.py \
  --label pml266-current-working-tree \
  --json-out benchmarks/results/pml266-current-working-tree.json \
  --include-output \
  --runs 5 \
  --warmups 1
```

Pre-PML-266 base worktree:

```bash
cd /mnt/c/Users/BenjaminSTOTT/PycharmProjects/merlin/.worktrees/pml266-before
source /home/benjamin/.virtualenvs/merlin/bin/activate
PYTHONPATH=$PWD python ../../benchmarks/benchmark_superposition_streaming.py \
  --label origin-main-pre-pml266 \
  --commit 5b36dc0f5ec877598ce65072f516b56800937ebd \
  --branch origin/main \
  --dirty false \
  --json-out ../../benchmarks/results/origin-main-pre-pml266.json \
  --include-output \
  --runs 5 \
  --warmups 1
```
