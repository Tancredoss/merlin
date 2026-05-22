"""No-cloud characterization tests for MerlinProcessor remote-job helpers."""

from __future__ import annotations

import threading
import time
from concurrent.futures import CancelledError
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from perceval.runtime import RemoteProcessor
from perceval.runtime.session import ISession

import merlin.core.merlin_processor as merlin_processor_module
from merlin.core.merlin_processor import MerlinProcessor


class FakeCommand:
    """Sampler command that records how it was submitted."""

    def __init__(self) -> None:
        self.executed = False
        self.name = None
        self.execute_kwargs = None

    def execute_async(self, **kwargs):
        """Record async execution arguments and return this fake job."""
        self.executed = True
        self.execute_kwargs = kwargs
        return self


class FakeSampler:
    """Minimal sampler exposing the command attributes used by _submit_job."""

    def __init__(self) -> None:
        self.probs = FakeCommand()
        self.sample_count = FakeCommand()
        self.samples = FakeCommand()


@dataclass
class FakeStatus:
    """Small job status object with the fields read by _poll_job."""

    state: str = "SUCCESS"
    progress: float = 1.0
    stop_message: str | None = None


class FakeJob:
    """Remote job fake with deterministic status and result events."""

    def __init__(
        self,
        *,
        job_id: str = "job-1",
        is_complete: bool = True,
        is_failed: bool = False,
        status: FakeStatus | None = None,
        result_events: list | None = None,
    ) -> None:
        self.id = job_id
        self.status = status or FakeStatus()
        self.is_complete = is_complete
        self.is_failed = is_failed
        self.cancelled = False
        self.get_results_calls = 0
        self._result_events = (
            [{"results_list": []}] if result_events is None else list(result_events)
        )

    def cancel(self) -> None:
        """Record that remote cancellation was requested."""
        self.cancelled = True

    def get_results(self):
        """Return or raise the next configured result event."""
        self.get_results_calls += 1
        event = (
            self._result_events.pop(0)
            if len(self._result_events) > 1
            else self._result_events[0]
        )
        if isinstance(event, BaseException):
            raise event
        return event


@dataclass
class FakeComputationSpace:
    """Computation-space shim exposing the enum value used by MerlinProcessor."""

    value: str


class FakeLayer:
    """Layer shim providing just enough state-mapping data for result parsing."""

    def __init__(
        self,
        *,
        final_keys: list[tuple[int, ...]] | None = None,
        computation_scheme: str = "fock",
    ) -> None:
        graph = SimpleNamespace(
            final_keys=[(1, 0), (0, 1)] if final_keys is None else final_keys
        )
        self.computation_process = SimpleNamespace(simulation_graph=graph)
        self.computation_space = FakeComputationSpace(computation_scheme)


def make_processor(available_commands: list[str]) -> MerlinProcessor:
    """Build an uninitialized processor configured for unit helper tests."""
    proc = MerlinProcessor.__new__(MerlinProcessor)
    proc.available_commands = available_commands
    proc._lock = threading.Lock()
    proc._active_jobs = set()
    return proc


def make_poll_processor(output: torch.Tensor | None = None) -> MerlinProcessor:
    """Build a processor whose result parser records the raw payload."""
    proc = make_processor(["probs"])
    proc.processed_calls = []

    def process_results(raw_results, batch_size, layer, nsample):
        proc.processed_calls.append((raw_results, batch_size, layer, nsample))
        return torch.tensor([[1.0]]) if output is None else output

    proc._process_batch_results = process_results
    return proc


def make_state() -> dict:
    """Return the mutable polling state shape expected by _poll_job."""
    return {"cancel_requested": False, "job_ids": []}


def test_remote_processor_path_copies_available_commands():
    """RemoteProcessor construction freezes the current command-detection path."""
    remote_processor = MagicMock(spec=RemoteProcessor)
    remote_processor.name = "sim:slos"
    remote_processor.available_commands = ["probs", "sample_count"]
    remote_processor.proxies = None

    with patch.object(MerlinProcessor, "_extract_rp_token", return_value="token"):
        proc = MerlinProcessor(remote_processor=remote_processor)

    assert proc.remote_processor is remote_processor
    assert proc.session is None
    assert proc.available_commands == ["probs", "sample_count"]


def test_session_path_is_characterized_as_empty_commands_and_sampling_only():
    """ISession currently starts with no command list, so submission samples."""
    session = MagicMock(spec=ISession)
    session.platform_name = "scaleway-like"
    proc = MerlinProcessor(session=session)
    sampler = FakeSampler()

    proc._submit_job(
        sampler,
        nsample=None,
        job_base_label="job",
        _capped_name=lambda base, command: f"{base}:{command}",
    )

    assert proc.session is session
    assert proc.remote_processor is None
    assert proc.available_commands == []
    assert sampler.sample_count.executed is True
    assert sampler.sample_count.execute_kwargs == {
        "max_samples": MerlinProcessor.DEFAULT_SHOTS_PER_CALL
    }
    assert sampler.sample_count.name == "job:sample_count"
    assert sampler.probs.executed is False


def test_submit_job_prefers_probs_when_available_without_samples():
    """Probability-capable backends use probs for exact probability requests."""
    proc = make_processor(["probs", "sample_count"])
    sampler = FakeSampler()

    returned_job = proc._submit_job(
        sampler,
        nsample=None,
        job_base_label="job",
        _capped_name=lambda base, command: f"{base}:{command}",
    )

    assert returned_job is sampler.probs
    assert sampler.probs.executed is True
    assert sampler.probs.execute_kwargs == {}
    assert sampler.probs.name == "job:probs"
    assert sampler.sample_count.executed is False
    assert sampler.samples.executed is False


def test_submit_job_treats_zero_samples_as_exact_probabilities():
    """Zero requested samples currently follows the same path as nsample=None."""
    proc = make_processor(["probs", "sample_count"])
    sampler = FakeSampler()

    returned_job = proc._submit_job(
        sampler,
        nsample=0,
        job_base_label="job",
        _capped_name=lambda base, command: f"{base}:{command}",
    )

    assert returned_job is sampler.probs
    assert sampler.probs.executed is True
    assert sampler.probs.execute_kwargs == {}
    assert sampler.probs.name == "job:probs"
    assert sampler.sample_count.executed is False


def test_submit_job_uses_sample_count_when_sampling_requested():
    """A positive sample request uses sample_count when the backend exposes it."""
    proc = make_processor(["probs", "sample_count"])
    sampler = FakeSampler()

    returned_job = proc._submit_job(
        sampler,
        nsample=37,
        job_base_label="job",
        _capped_name=lambda base, command: f"{base}:{command}",
    )

    assert returned_job is sampler.sample_count
    assert sampler.sample_count.executed is True
    assert sampler.sample_count.execute_kwargs == {"max_samples": 37}
    assert sampler.sample_count.name == "job:sample_count"
    assert sampler.probs.executed is False


def test_submit_job_falls_back_to_samples_when_sample_count_is_unavailable():
    """Backends without sample_count currently use samples for sampled jobs."""
    proc = make_processor(["samples"])
    sampler = FakeSampler()

    returned_job = proc._submit_job(
        sampler,
        nsample=11,
        job_base_label="job",
        _capped_name=lambda base, command: f"{base}:{command}",
    )

    assert returned_job is sampler.samples
    assert sampler.samples.executed is True
    assert sampler.samples.execute_kwargs == {"max_samples": 11}
    assert sampler.samples.name == "job:samples"
    assert sampler.sample_count.executed is False


def test_submit_job_defaults_to_sample_count_when_commands_are_empty():
    """An empty command list currently means sampling through sample_count."""
    proc = make_processor([])
    sampler = FakeSampler()

    returned_job = proc._submit_job(
        sampler,
        nsample=None,
        job_base_label=None,
        _capped_name=lambda base, command: f"{base}:{command}",
    )

    assert returned_job is sampler.sample_count
    assert sampler.sample_count.executed is True
    assert sampler.sample_count.execute_kwargs == {
        "max_samples": MerlinProcessor.DEFAULT_SHOTS_PER_CALL
    }
    assert sampler.probs.executed is False
    assert sampler.samples.executed is False


def test_poll_job_success_processes_dict_payload_and_records_job_id():
    """Successful polling records the job id and delegates dict parsing."""
    output = torch.tensor([[0.25, 0.75]])
    proc = make_poll_processor(output=output)
    raw_results = {"results_list": [{"results": {"|1,0>": 1.0}}]}
    job = FakeJob(job_id="job-success", result_events=[raw_results])
    proc._active_jobs.add(job)
    state = make_state()
    layer = object()

    result = proc._poll_job(
        job,
        state,
        deadline=None,
        batch_size=3,
        layer=layer,
        nsample=None,
    )

    assert torch.equal(result, output)
    assert state["job_ids"] == ["job-success"]
    assert proc.processed_calls == [(raw_results, 3, layer, None)]
    assert job not in proc._active_jobs


def test_poll_job_failed_status_raises_with_stop_message_and_job_id():
    """Failed jobs surface the backend stop message and current job id."""
    proc = make_poll_processor()
    job = FakeJob(
        job_id="job-failed",
        is_complete=False,
        is_failed=True,
        status=FakeStatus(state="FAILED", stop_message="hardware rejected job"),
    )
    proc._active_jobs.add(job)

    with pytest.raises(RuntimeError, match="hardware rejected job.*job-failed"):
        proc._poll_job(job, make_state(), None, 1, object(), None)

    assert job not in proc._active_jobs


def test_poll_job_cancel_request_cancels_remote_job():
    """Caller cancellation asks the backend job to cancel before raising."""
    proc = make_poll_processor()
    job = FakeJob(is_complete=False)
    state = make_state()
    state["cancel_requested"] = True

    with pytest.raises(CancelledError, match="Remote call was cancelled"):
        proc._poll_job(job, state, None, 1, object(), None)

    assert job.cancelled is True


def test_poll_job_timeout_cancels_remote_job():
    """Timeouts request remote cancellation and raise TimeoutError."""
    proc = make_poll_processor()
    job = FakeJob(is_complete=False)

    with pytest.raises(TimeoutError, match="remote cancel issued"):
        proc._poll_job(job, make_state(), time.time() - 1.0, 1, object(), None)

    assert job.cancelled is True


def test_poll_job_cancel_requested_stop_message_raises_cancelled_error():
    """A failed job whose stop message says cancel maps to CancelledError."""
    proc = make_poll_processor()
    job = FakeJob(
        is_complete=False,
        is_failed=True,
        status=FakeStatus(state="FAILED", stop_message="Cancel requested by user"),
    )
    proc._active_jobs.add(job)

    with pytest.raises(CancelledError, match="Remote call was cancelled"):
        proc._poll_job(job, make_state(), None, 1, object(), None)

    assert job not in proc._active_jobs


def test_poll_job_cancel_requested_get_results_exception_raises_cancelled_error():
    """A completion-time cancel exception maps to CancelledError."""
    proc = make_poll_processor()
    job = FakeJob(result_events=[RuntimeError("Cancel requested on backend")])
    proc._active_jobs.add(job)

    with pytest.raises(CancelledError, match="Remote call was cancelled"):
        proc._poll_job(job, make_state(), None, 1, object(), None)

    assert job not in proc._active_jobs


def test_poll_job_retries_when_results_are_not_available(monkeypatch):
    """Completed jobs retry when Perceval says results are not available yet."""
    monkeypatch.setattr(merlin_processor_module.time, "sleep", lambda _seconds: None)
    output = torch.tensor([[1.0]])
    proc = make_poll_processor(output=output)
    raw_results = {"results_list": [{"results": {"|1,0>": 1.0}}]}
    job = FakeJob(
        result_events=[
            RuntimeError("Results are not available"),
            raw_results,
        ]
    )

    result = proc._poll_job(job, make_state(), None, 1, object(), None)

    assert torch.equal(result, output)
    assert job.get_results_calls == 2


def test_poll_job_retries_complete_non_dict_payloads_then_fails(monkeypatch):
    """Complete non-dict payloads are retried for the current bounded window."""
    monkeypatch.setattr(merlin_processor_module.time, "sleep", lambda _seconds: None)
    proc = make_poll_processor()
    job = FakeJob(job_id="job-nondict", result_events=[["not", "a", "dict"]])
    proc._active_jobs.add(job)

    with pytest.raises(RuntimeError, match="not a dict after 60 re-polls"):
        proc._poll_job(job, make_state(), None, 1, object(), None)

    assert job.get_results_calls == 60
    assert job not in proc._active_jobs


def test_process_batch_results_rejects_missing_payload():
    """A missing backend payload currently raises a runtime failure."""
    proc = make_processor(["probs"])

    with pytest.raises(RuntimeError, match="returned no results"):
        proc._process_batch_results(None, 1, FakeLayer())


def test_process_batch_results_rejects_non_dict_payload():
    """A non-dict backend payload currently raises a runtime failure."""
    proc = make_processor(["probs"])

    with pytest.raises(RuntimeError, match="Unexpected remote results type"):
        proc._process_batch_results(["not", "a", "dict"], 1, FakeLayer())


def test_process_batch_results_normalizes_counts():
    """Integer count payloads are normalized into probabilities per row."""
    proc = make_processor(["probs"])
    layer = FakeLayer()
    raw_results = {
        "results_list": [{"results": {"|1,0>": 3, "|0,1>": 1}}],
    }

    result = proc._process_batch_results(raw_results, 1, layer)

    assert torch.allclose(result, torch.tensor([[0.75, 0.25]]))


def test_process_batch_results_passes_probability_payloads_through():
    """Float payloads no larger than one are treated as probabilities."""
    proc = make_processor(["probs"])
    layer = FakeLayer()
    raw_results = {
        "results_list": [{"results": {"|1,0>": 0.2, "|0,1>": 0.8}}],
    }

    result = proc._process_batch_results(raw_results, 1, layer)

    assert torch.allclose(result, torch.tensor([[0.2, 0.8]]))


def test_process_batch_results_filters_invalid_states_for_non_fock_space():
    """Non-FOCK mappings currently filter states outside final_keys."""
    proc = make_processor(["probs"])
    layer = FakeLayer(
        final_keys=[(1, 0), (0, 1)],
        computation_scheme="unbunched",
    )
    raw_results = {
        "results_list": [{"results": {"|1,0>": 5, "|2,0>": 5}}],
    }

    result = proc._process_batch_results(raw_results, 1, layer)

    assert torch.allclose(result, torch.tensor([[1.0, 0.0]]))


def test_process_batch_results_zero_fills_missing_rows():
    """Missing result rows are padded with zero probability rows."""
    proc = make_processor(["probs"])
    layer = FakeLayer()
    raw_results = {
        "results_list": [
            {"results": {"|0,1>": 1.0}},
            {"metadata": "no results key"},
        ],
    }

    result = proc._process_batch_results(raw_results, 3, layer)

    assert torch.allclose(
        result,
        torch.tensor([
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]),
    )


def test_process_batch_results_probability_heuristic_renormalizes_float_rows():
    """The current heuristic treats first float <= 1 as probability payloads."""
    proc = make_processor(["probs"])
    layer = FakeLayer()
    raw_results = {
        "results_list": [{"results": {"|1,0>": 1.0, "|0,1>": 1.0}}],
    }

    result = proc._process_batch_results(raw_results, 1, layer)

    assert torch.allclose(result, torch.tensor([[0.5, 0.5]]))
