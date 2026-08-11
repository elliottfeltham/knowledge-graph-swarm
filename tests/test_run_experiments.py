import pytest

from run_experiments import aggregate_seed_results, split_queries


def test_query_split_is_reproducible_and_keeps_sets_separate() -> None:
    queries = [{"query_id": str(index)} for index in range(10)]

    first_split = split_queries(
        queries, seed=10, adaptation_count=4, validation_count=3, evaluation_count=2
    )
    second_split = split_queries(
        queries, seed=10, adaptation_count=4, validation_count=3, evaluation_count=2
    )

    assert first_split == second_split
    adaptation_ids = {row["query_id"] for row in first_split[0]}
    validation_ids = {row["query_id"] for row in first_split[1]}
    evaluation_ids = {row["query_id"] for row in first_split[2]}
    assert adaptation_ids.isdisjoint(validation_ids)
    assert adaptation_ids.isdisjoint(evaluation_ids)
    assert validation_ids.isdisjoint(evaluation_ids)


def test_seed_results_are_aggregated_with_mean_and_standard_deviation() -> None:
    seed_results = []

    for recall_at_5 in (0.4, 0.5, 0.6):
        seed_results.append(
            {
                "systems": {
                    "faiss": {
                        "mean_recall_at_5": recall_at_5,
                        "mean_recall_at_10": 0.7,
                        "mean_mrr": 0.6,
                        "mean_retrieval_latency_ms": 1.0,
                    }
                }
            }
        )

    aggregate = aggregate_seed_results(seed_results)

    assert aggregate["faiss"]["recall_at_5"]["mean"] == pytest.approx(0.5)
    assert aggregate["faiss"]["recall_at_5"]["standard_deviation"] == pytest.approx(0.1)
