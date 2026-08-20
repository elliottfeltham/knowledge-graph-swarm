from evaluation import create_results_report
from evaluation.create_results_report import (
    build_pruning_rows,
    build_retrieval_rows,
)


def metric(mean: float, standard_deviation: float = 0.1) -> dict:
    return {
        "mean": mean,
        "standard_deviation": standard_deviation,
    }


def test_build_retrieval_rows_formats_aggregate_results() -> None:
    results = {
        "aggregate": {
            "faiss": {
                "recall_at_5": metric(0.7),
                "recall_at_10": metric(0.8),
                "mrr": metric(0.9),
                "retrieval_latency_ms": metric(0.2),
            },
            "optimised_adapted_agent_graph": {
                "recall_at_5": metric(0.75),
                "recall_at_10": metric(0.85),
                "mrr": metric(0.92),
                "retrieval_latency_ms": metric(0.3),
            },
        }
    }

    rows = build_retrieval_rows(results)

    assert rows[0]["system"] == "FAISS"
    assert rows[0]["recall_at_10_mean"] == 0.8
    assert rows[0]["mrr_std"] == 0.1
    assert rows[1]["system"] == "Optimised adapted graph"


def test_build_pruning_rows_averages_graph_statistics() -> None:
    seed_result = {
        "graph_before": {
            "undirected_edges": 100,
            "average_degree": 4.0,
            "average_edge_weight": 0.1,
        },
        "graph_after": {
            "undirected_edges": 80,
            "average_degree": 3.2,
            "average_edge_weight": 0.08,
        },
        "graph_deltas": {"undirected_edges": -20},
    }
    results = {
        "conditions": {
            "pruning_enabled": {
                "aggregate": {
                    "recall_at_5": metric(0.7),
                    "recall_at_10": metric(0.8),
                    "mrr": metric(0.9),
                },
                "seed_results": [seed_result, seed_result],
            }
        }
    }

    rows = build_pruning_rows(results)

    assert rows[0]["condition"] == "Pruning Enabled"
    assert rows[0]["edges_after"] == 80
    assert rows[0]["edge_delta"] == -20


def test_adaptation_report_creates_graph_evolution_figure(tmp_path, monkeypatch) -> None:
    csv_path = tmp_path / "adaptation.csv"
    retrieval_figure = tmp_path / "adaptation.png"
    graph_figure = tmp_path / "graph_evolution.png"
    monkeypatch.setattr(create_results_report, "ADAPTATION_OVER_TIME_CSV_PATH", csv_path)
    monkeypatch.setattr(
        create_results_report,
        "ADAPTATION_OVER_TIME_FIGURE_PATH",
        retrieval_figure,
    )
    monkeypatch.setattr(
        create_results_report,
        "GRAPH_EVOLUTION_FIGURE_PATH",
        graph_figure,
    )
    results = {
        "aggregate_checkpoints": [
            {
                "adaptation_queries": checkpoint,
                "metrics": {
                    "recall_at_5": metric(0.7),
                    "recall_at_10": metric(0.8),
                    "mrr": metric(0.9),
                },
                "graph": {
                    "undirected_edges": metric(100 + checkpoint),
                    "average_edge_weight": metric(0.1),
                    "max_edge_weight": metric(0.2),
                },
                "updates": {
                    "strengthened_edges": metric(checkpoint),
                    "created_edges": metric(checkpoint / 2),
                    "decayed_edges": metric(checkpoint * 2),
                    "pruned_edges": metric(0),
                },
            }
            for checkpoint in (0, 25)
        ]
    }

    create_results_report.save_adaptation_over_time_report(results)

    assert csv_path.exists()
    assert retrieval_figure.exists()
    assert graph_figure.exists()
