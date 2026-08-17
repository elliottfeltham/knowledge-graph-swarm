from evaluation.tune_graph_updates import (
    build_parameter_grid,
    select_best_configuration,
)


def make_configuration(
    strengthen_amount: float,
    decay_rate: float,
    recall_at_10: float,
    mrr: float,
    recall_at_5: float,
) -> dict:
    return {
        "strengthen_amount": strengthen_amount,
        "decay_rate": decay_rate,
        "adapted": {
            "recall_at_10": recall_at_10,
            "mrr": mrr,
            "recall_at_5": recall_at_5,
        },
    }


def test_parameter_grid_contains_every_combination() -> None:
    grid = build_parameter_grid([0.01, 0.05], [0.99, 0.995, 1.0])

    assert grid == [
        (0.01, 0.99),
        (0.01, 0.995),
        (0.01, 1.0),
        (0.05, 0.99),
        (0.05, 0.995),
        (0.05, 1.0),
    ]


def test_selection_uses_metrics_before_update_tiebreaks() -> None:
    configurations = [
        make_configuration(0.01, 1.0, 0.80, 0.90, 0.75),
        make_configuration(0.1, 0.98, 0.81, 0.85, 0.70),
        make_configuration(0.075, 0.99, 0.81, 0.86, 0.74),
        make_configuration(0.05, 0.995, 0.81, 0.86, 0.74),
        make_configuration(0.05, 1.0, 0.81, 0.86, 0.74),
        make_configuration(0.01, 0.995, 0.81, 0.86, 0.74),
        make_configuration(0.01, 1.0, 0.81, 0.86, 0.74),
    ]

    best = select_best_configuration(configurations)

    assert best["strengthen_amount"] == 0.01
    assert best["decay_rate"] == 1.0
