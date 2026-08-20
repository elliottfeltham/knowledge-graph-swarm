import pytest

from evaluation.evaluate_adaptation_over_time import (
    aggregate_checkpoints,
    calculate_ranking_changes,
    validate_checkpoints,
)


def test_checkpoints_are_sorted_and_must_fit_adaptation_queries() -> None:
    assert validate_checkpoints([100, 0, 25, 25], 100) == [0, 25, 100]

    with pytest.raises(ValueError):
        validate_checkpoints([25, 100], 100)

    with pytest.raises(ValueError):
        validate_checkpoints([0, 101], 100)


def test_checkpoint_results_are_aggregated_across_seeds() -> None:
    seed_results = []

    for recall, edge_weight in ((0.8, 0.1), (0.9, 0.2)):
        seed_results.append(
            {
                "checkpoints": [
                    {
                        "adaptation_queries": 25,
                        "metrics": {"recall_at_10": recall},
                        "ranking_changes": {"top_10_set_changes": 2},
                        "graph": {"average_edge_weight": edge_weight},
                        "updates": {"strengthened_edges": 75},
                    }
                ]
            }
        )

    aggregate = aggregate_checkpoints(seed_results)

    assert aggregate[0]["adaptation_queries"] == 25
    assert aggregate[0]["metrics"]["recall_at_10"]["mean"] == pytest.approx(0.85)
    assert aggregate[0]["ranking_changes"]["top_10_set_changes"]["mean"] == 2
    assert aggregate[0]["graph"]["average_edge_weight"]["mean"] == pytest.approx(0.15)
    assert aggregate[0]["updates"]["strengthened_edges"]["mean"] == 75


def test_ranking_changes_compare_membership_and_order() -> None:
    initial_rows = [
        {"query_id": "q1", "retrieved_doc_ids": ["a", "b", "c"]},
        {"query_id": "q2", "retrieved_doc_ids": ["d", "e", "f"]},
    ]
    current_rows = [
        {"query_id": "q1", "retrieved_doc_ids": ["b", "a", "c"]},
        {"query_id": "q2", "retrieved_doc_ids": ["d", "e", "g"]},
    ]

    changes = calculate_ranking_changes(initial_rows, current_rows)

    assert changes == {
        "top_10_set_changes": 1,
        "top_10_set_change_percentage": 50.0,
        "top_10_order_changes": 2,
        "top_10_order_change_percentage": 100.0,
    }
