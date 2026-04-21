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
        policy="longest_input_first",
    )
    summary = summarize_schedule(batches)

    assert len(batches) == 3
    assert sum(batch.batch_size for batch in batches) == 10
    assert {batch.replica_id for batch in batches} == {0, 1}
    assert summary["requested_input_tokens"] == 320
    assert summary["requested_output_tokens"] == 320
    assert summary["padded_prefill_tokens"] >= summary["requested_input_tokens"]
    assert summary["padded_decode_tokens"] >= summary["requested_output_tokens"]


def test_realistic_policy_does_not_sort_by_output_length():
    requests = build_serving_requests(count=8, mean_input=16, mean_output=16, max_input=64, max_output=64)
    batches = make_static_batches(
        requests,
        batch_size=4,
        num_replicas=1,
        policy="longest_input_first",
    )

    ordered = [request for batch in batches for request in batch.requests]
    assert ordered == sorted(
        requests,
        key=lambda request: (request.input_length, request.request_id),
        reverse=True,
    )
    assert ordered != sorted(
        requests,
        key=lambda request: (request.output_length, request.input_length),
        reverse=True,
    )


def test_random_policy_is_deterministic_and_differs_from_fifo():
    requests = build_serving_requests(count=10, mean_input=32, mean_output=32, max_input=128, max_output=128)

    first = make_static_batches(
        requests,
        batch_size=5,
        num_replicas=1,
        policy="random",
        seed=7,
    )
    second = make_static_batches(
        requests,
        batch_size=5,
        num_replicas=1,
        policy="random",
        seed=7,
    )

    first_order = [request.request_id for batch in first for request in batch.requests]
    second_order = [request.request_id for batch in second for request in batch.requests]

    assert first_order == second_order
    assert first_order != [request.request_id for request in requests]


def test_input_bucketed_random_policy_uses_only_input_buckets():
    requests = build_serving_requests(count=16, mean_input=32, mean_output=32, max_input=128, max_output=128)
    batches = make_static_batches(
        requests,
        batch_size=4,
        num_replicas=1,
        policy="input_bucketed_random",
        seed=11,
    )

    ordered = [request for batch in batches for request in batch.requests]
    buckets = [(request.input_length - 1) // 128 for request in ordered]

    assert buckets == sorted(buckets, reverse=True)
    assert ordered != sorted(
        requests,
        key=lambda request: (request.output_length, request.input_length),
        reverse=True,
    )


def test_greedy_prefill_assignment_balances_known_prefill_work():
    requests = build_serving_requests(count=24, mean_input=32, mean_output=32, max_input=128, max_output=128)

    batches = make_static_batches(
        requests,
        batch_size=4,
        num_replicas=2,
        policy="longest_input_first",
        replica_assignment="greedy_prefill",
    )

    loads = [0, 0]
    for batch in batches:
        loads[batch.replica_id] += batch.padded_prefill_tokens

    assert {batch.replica_id for batch in batches} == {0, 1}
    assert max(loads) - min(loads) <= max(batch.padded_prefill_tokens for batch in batches)
