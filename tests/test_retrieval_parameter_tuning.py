import pytest

from evaluation.experiment_helpers import (
    compare_query_metrics,
    split_query_groups,
)
from evaluation.tune_retrieval_parameters import select_best_configuration


def test_query_groups_are_reproducible_and_disjoint() -> None:
    queries = [{"query_id": str(index)} for index in range(12)]

    first = split_query_groups(queries, 10, 4, 3, 2)
    second = split_query_groups(queries, 10, 4, 3, 2)

    assert first == second
    query_id_groups = [{row["query_id"] for row in group} for group in first]
    assert query_id_groups[0].isdisjoint(query_id_groups[1])
    assert query_id_groups[0].isdisjoint(query_id_groups[2])
    assert query_id_groups[1].isdisjoint(query_id_groups[2])


@pytest.mark.parametrize(
    ("agent_metrics", "expected"),
    [
        ({"recall_at_10": 1.0, "mrr": 0.5}, "improved"),
        ({"recall_at_10": 0.5, "mrr": 1.0}, "improved"),
        ({"recall_at_10": 0.5, "mrr": 0.5}, "unchanged"),
        ({"recall_at_10": 0.5, "mrr": 0.25}, "worsened"),
    ],
)
def test_query_comparison_uses_recall_then_mrr(agent_metrics, expected) -> None:
    faiss_metrics = {"recall_at_10": 0.5, "mrr": 0.5}

    assert compare_query_metrics(agent_metrics, faiss_metrics) == expected


def test_best_configuration_prefers_recall_then_mrr_then_lower_weight() -> None:
    configurations = [
        {
            "seed_k": 5,
            "graph_weight": 0.3,
            "adapted": {"recall_at_10": 0.8, "mrr": 0.9},
        },
        {
            "seed_k": 10,
            "graph_weight": 0.2,
            "adapted": {"recall_at_10": 0.81, "mrr": 0.85},
        },
        {
            "seed_k": 5,
            "graph_weight": 0.1,
            "adapted": {"recall_at_10": 0.81, "mrr": 0.85},
        },
    ]

    best = select_best_configuration(configurations)

    assert best["graph_weight"] == 0.1
