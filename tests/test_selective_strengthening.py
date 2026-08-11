from evaluation.evaluate_selective_strengthening import select_best_condition


def make_condition(update_k: int, recall_at_10: float, mrr: float) -> dict:
    return {
        "update_k": update_k,
        "validation": {
            "adapted": {
                "recall_at_10": recall_at_10,
                "mrr": mrr,
            }
        },
    }


def test_selection_prefers_validation_recall_then_mrr_then_smaller_update_k() -> None:
    conditions = [
        make_condition(10, 0.80, 0.90),
        make_condition(5, 0.81, 0.85),
        make_condition(3, 0.81, 0.85),
    ]

    assert select_best_condition(conditions)["update_k"] == 3
