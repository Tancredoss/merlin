#!/usr/bin/env bash
set -euo pipefail

# Run the superposition-streaming benchmark against two revisions:
#   1. the current checkout
#   2. a temporary git worktree at a baseline ref
#
# The script is intentionally environment-agnostic. It uses whichever Python
# executable is supplied via --python or the PYTHON environment variable, and it
# relies on that environment already having Merlin's benchmark dependencies
# installed.

usage() {
  cat <<'USAGE'
Usage:
  benchmarks/run_superposition_streaming_comparison.sh [options]

Options:
  --baseline-ref REF      Git ref for the baseline worktree (default: origin/main)
  --results-dir DIR       Directory for JSON outputs (default: benchmarks/results)
  --candidate-label NAME  Label for the current checkout run (default: candidate)
  --baseline-label NAME   Label for the baseline ref run (default: baseline)
  --runs N               Measured runs per case (default: 5)
  --warmups N            Warmup runs per case (default: 1)
  --case NAME            Benchmark case to run; can be repeated
  --rtol VALUE           Relative tolerance for output comparison (default: 1e-5)
  --atol VALUE           Absolute tolerance for output comparison (default: 1e-6)
  --python PATH          Python executable to use (default: $PYTHON or python)
  --no-compare-output    Skip full complex output comparison
  --keep-worktree        Do not delete the temporary baseline worktree
  -h, --help             Show this help text

Examples:
  benchmarks/run_superposition_streaming_comparison.sh

  PYTHON=/path/to/venv/bin/python \
    benchmarks/run_superposition_streaming_comparison.sh \
      --baseline-ref origin/main \
      --candidate-label my-branch \
      --baseline-label main
USAGE
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

baseline_ref="origin/main"
results_dir="$repo_root/benchmarks/results"
candidate_label="candidate"
baseline_label="baseline"
runs="5"
warmups="1"
python_bin="${PYTHON:-python}"
keep_worktree="false"
compare_output="true"
rtol="1e-5"
atol="1e-6"
cases=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline-ref)
      baseline_ref="$2"
      shift 2
      ;;
    --results-dir)
      results_dir="$2"
      shift 2
      ;;
    --candidate-label)
      candidate_label="$2"
      shift 2
      ;;
    --baseline-label)
      baseline_label="$2"
      shift 2
      ;;
    --runs)
      runs="$2"
      shift 2
      ;;
    --warmups)
      warmups="$2"
      shift 2
      ;;
    --case)
      cases+=("--case" "$2")
      shift 2
      ;;
    --rtol)
      rtol="$2"
      shift 2
      ;;
    --atol)
      atol="$2"
      shift 2
      ;;
    --python)
      python_bin="$2"
      shift 2
      ;;
    --no-compare-output)
      compare_output="false"
      shift
      ;;
    --keep-worktree)
      keep_worktree="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

benchmark_script="$repo_root/benchmarks/benchmark_superposition_streaming.py"
mkdir -p "$results_dir"

candidate_json="$results_dir/${candidate_label}.json"
baseline_json="$results_dir/${baseline_label}.json"
candidate_log="$results_dir/${candidate_label}.stdout.log"
baseline_log="$results_dir/${baseline_label}.stdout.log"
output_args=()
if [[ "$compare_output" == "true" ]]; then
  output_args+=("--include-output")
fi

baseline_commit="$(git -C "$repo_root" rev-parse "$baseline_ref")"
baseline_worktree="$(mktemp -d "${TMPDIR:-/tmp}/merlin-superposition-baseline.XXXXXX")"

cleanup() {
  if [[ "$keep_worktree" == "true" ]]; then
    echo "Keeping baseline worktree at: $baseline_worktree"
    return
  fi
  git -C "$repo_root" worktree remove --force "$baseline_worktree" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Benchmarking current checkout: $repo_root"
if ! PYTHONPATH="$repo_root" "$python_bin" "$benchmark_script" \
    --label "$candidate_label" \
    --json-out "$candidate_json" \
    --runs "$runs" \
    --warmups "$warmups" \
    "${output_args[@]}" \
    "${cases[@]}" >"$candidate_log" 2>&1; then
  echo "Candidate benchmark failed. Log: $candidate_log" >&2
  tail -n 80 "$candidate_log" >&2 || true
  exit 1
fi
rm -f "$candidate_log"

echo "Preparing baseline worktree: $baseline_ref ($baseline_commit)"
git -C "$repo_root" worktree add --quiet --detach "$baseline_worktree" "$baseline_commit"

echo "Benchmarking baseline worktree: $baseline_worktree"
if ! (
    cd "$baseline_worktree"
    PYTHONPATH="$baseline_worktree" "$python_bin" "$benchmark_script" \
      --label "$baseline_label" \
      --commit "$baseline_commit" \
      --branch "$baseline_ref" \
      --dirty false \
      --json-out "$baseline_json" \
      --runs "$runs" \
      --warmups "$warmups" \
      "${output_args[@]}" \
      "${cases[@]}"
  ) >"$baseline_log" 2>&1; then
  echo "Baseline benchmark failed. Log: $baseline_log" >&2
  tail -n 80 "$baseline_log" >&2 || true
  exit 1
fi
rm -f "$baseline_log"

if [[ "$compare_output" == "true" ]]; then
  echo "Comparing candidate and baseline outputs"
  "$python_bin" - "$candidate_json" "$baseline_json" "$rtol" "$atol" <<'PY'
import json
import math
import sys

candidate_path, baseline_path, rtol_text, atol_text = sys.argv[1:]
rtol = float(rtol_text)
atol = float(atol_text)

with open(candidate_path, encoding="utf-8") as handle:
    candidate = json.load(handle)
with open(baseline_path, encoding="utf-8") as handle:
    baseline = json.load(handle)

candidate_cases = {case["name"]: case for case in candidate["cases"]}
baseline_cases = {case["name"]: case for case in baseline["cases"]}
missing = sorted(set(candidate_cases) ^ set(baseline_cases))
if missing:
    raise SystemExit(f"Case mismatch between outputs: {missing}")

worst_case = None
worst_diff = -1.0
worst_allowed = 0.0

for name in sorted(candidate_cases):
    candidate_output = candidate_cases[name].get("output_values")
    baseline_output = baseline_cases[name].get("output_values")
    if candidate_output is None or baseline_output is None:
        raise SystemExit(
            "Missing output_values. Run benchmark with --include-output or "
            "disable comparison with --no-compare-output."
        )
    if candidate_output["shape"] != baseline_output["shape"]:
        raise SystemExit(
            f"Output shape mismatch for {name}: "
            f"{candidate_output['shape']} != {baseline_output['shape']}"
        )

    for idx, (cr, ci, br, bi) in enumerate(
        zip(
            candidate_output["real"],
            candidate_output["imag"],
            baseline_output["real"],
            baseline_output["imag"],
            strict=True,
        )
    ):
        diff = math.hypot(cr - br, ci - bi)
        reference = math.hypot(br, bi)
        allowed = atol + rtol * reference
        if diff > worst_diff:
            worst_case = f"{name}[{idx}]"
            worst_diff = diff
            worst_allowed = allowed
        if diff > allowed:
            raise SystemExit(
                f"Output mismatch for {name}[{idx}]: "
                f"diff={diff:.6g}, allowed={allowed:.6g}"
            )

print(
    "Output comparison passed: "
    f"{len(candidate_cases)} case(s), max_diff={worst_diff:.6g} "
    f"at {worst_case}, allowed={worst_allowed:.6g}"
)
PY
fi

echo "Wrote candidate results: $candidate_json"
echo "Wrote baseline results:  $baseline_json"
