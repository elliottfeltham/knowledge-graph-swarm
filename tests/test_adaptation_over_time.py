import pytest

from evaluation.evaluate_adaptation_over_time import (
    aggregate_checkpoints,
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
                        "graph": {"average_edge_weight": edge_weight},
                        "updates": {"strengthened_edges": 75},
                    }
                ]
            }
        )

    aggregate = aggregate_checkpoints(seed_results)

    assert aggregate[0]["adaptation_queries"] == 25
    assert aggregate[0]["metrics"]["recall_at_10"]["mean"] == pytest.approx(0.85)
    assert aggregate[0]["graph"]["average_edge_weight"]["mean"] == pytest.approx(0.15)
    assert aggregate[0]["updates"]["strengthened_edges"]["mean"] == 75
