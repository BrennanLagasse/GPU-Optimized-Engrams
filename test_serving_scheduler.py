from scripts.serving_scheduler import make_static_batches, summarize_schedule
from scripts.serving_workload import build_serving_requests, summarize_requests


def test_serving_workload_has_requested_long_tail_shape():
    requests = build_serving_requests(seed=123)
    summary = summarize_requests(requests)

    assert summary["num_requests"] == 100
    assert summary["mean_input_tokens"] == 128
    assert summary["mean_output_tokens"] == 128
    assert summary["max_input_tokens"] == 1024
    assert summary["max_output_tokens"] == 1024
    assert any(request.input_length == 1024 for request in requests)
    assert any(request.output_length == 1024 for request in requests)


def test_static_batches_preserve_request_count_and_measure_padding():
    requests = build_serving_requests(count=10, mean_input=32, mean_output=32, max_input=128, max_output=128)
    batches = make_static_batches(
        requests,
        batch_size=4,
        num_replicas=2,
        policy="longest_output_first",
    )
    summary = summarize_schedule(batches)

    assert len(batches) == 3
    assert sum(batch.batch_size for batch in batches) == 10
    assert {batch.replica_id for batch in batches} == {0, 1}
    assert summary["requested_input_tokens"] == 320
    assert summary["requested_output_tokens"] == 320
    assert summary["padded_prefill_tokens"] >= summary["requested_input_tokens"]
    assert summary["padded_decode_tokens"] >= summary["requested_output_tokens"]
