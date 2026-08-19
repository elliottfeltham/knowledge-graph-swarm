import pytest

from evaluation.evaluate_repeated_adaptation import aggregate_condition_results


def test_repeated_adaptation_results_are_aggregated_across_seeds() -> None:
    seed_results = []

    for recall, changed in ((0.8, 2), (0.9, 4)):
        seed_results.append(
            {
                "conditions": {
                    "example": [
                        {
                            "pass": 1,
                            "interactions": 100,
                            "rankings_changed": changed,
                            "metrics": {"recall_at_10": recall},
                            "graph": {"undirected_edges": 100},
                            "updates": {"strengthened_edges": 300},
                        }
                    ]
                }
            }
        )

    aggregate = aggregate_condition_results(seed_results)
    result = aggregate["example"][0]

    assert result["metrics"]["recall_at_10"]["mean"] == pytest.approx(0.85)
    assert result["rankings_changed"]["mean"] == 3
    assert result["graph"]["undirected_edges"]["mean"] == 100
    assert result["updates"]["strengthened_edges"]["mean"] == 300
