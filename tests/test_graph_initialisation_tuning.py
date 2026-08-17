from evaluation.tune_graph_initialisation import (
    build_parameter_grid,
    select_best_configuration,
)


def make_configuration(
    neighbours: int,
    weight_scale: float,
    recall_at_10: float,
    mrr: float,
    recall_at_5: float,
) -> dict:
    return {
        "neighbours_per_document": neighbours,
        "initial_weight_scale": weight_scale,
        "adapted": {
            "recall_at_10": recall_at_10,
            "mrr": mrr,
            "recall_at_5": recall_at_5,
        },
    }


def test_parameter_grid_contains_every_combination() -> None:
    grid = build_parameter_grid([3, 5], [0.1, 0.2, 0.3])

    assert grid == [
        (3, 0.1),
        (3, 0.2),
        (3, 0.3),
        (5, 0.1),
        (5, 0.2),
        (5, 0.3),
    ]


def test_selection_uses_metrics_before_simpler_graph_tiebreaks() -> None:
    configurations = [
        make_configuration(3, 0.05, 0.80, 0.90, 0.75),
        make_configuration(15, 0.5, 0.81, 0.85, 0.70),
        make_configuration(5, 0.2, 0.81, 0.86, 0.74),
        make_configuration(3, 0.1, 0.81, 0.86, 0.74),
        make_configuration(3, 0.05, 0.81, 0.86, 0.74),
    ]

    best = select_best_configuration(configurations)

    assert best["neighbours_per_document"] == 3
    assert best["initial_weight_scale"] == 0.05
