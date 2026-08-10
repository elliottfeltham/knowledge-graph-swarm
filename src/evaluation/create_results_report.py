import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import PROJECT_ROOT


MAIN_RESULTS_PATH = PROJECT_ROOT / "results" / "main_evaluation.json"
PRUNING_RESULTS_PATH = PROJECT_ROOT / "results" / "pruning_ablation.json"
REPORT_DIRECTORY = PROJECT_ROOT / "results" / "report"

SYSTEM_LABELS = {
    "faiss": "FAISS",
    "static_graph": "Static graph",
    "initial_agent_graph": "Initial agent graph",
    "adapted_agent_graph": "Adapted agent graph",
}

METRICS = [
    ("recall_at_5", "Recall@5"),
    ("recall_at_10", "Recall@10"),
    ("mrr", "MRR"),
]

COLOURS = ["#2878B5", "#E07A2D", "#2A9D8F", "#C44E52"]


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing evaluation results: {path}. Run src/run_experiments.py first."
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_retrieval_rows(results: dict) -> list[dict]:
    rows = []

    for system_name, metrics in results["aggregate"].items():
        rows.append(
            {
                "system": SYSTEM_LABELS[system_name],
                "recall_at_5_mean": metrics["recall_at_5"]["mean"],
                "recall_at_5_std": metrics["recall_at_5"]["standard_deviation"],
                "recall_at_10_mean": metrics["recall_at_10"]["mean"],
                "recall_at_10_std": metrics["recall_at_10"]["standard_deviation"],
                "mrr_mean": metrics["mrr"]["mean"],
                "mrr_std": metrics["mrr"]["standard_deviation"],
                "latency_ms_mean": metrics["retrieval_latency_ms"]["mean"],
                "latency_ms_std": metrics["retrieval_latency_ms"]["standard_deviation"],
            }
        )

    return rows


def average_graph_value(seed_results: list[dict], section: str, field: str) -> float:
    return mean(result[section][field] for result in seed_results)


def build_pruning_rows(results: dict) -> list[dict]:
    rows = []

    for condition_name, condition in results["conditions"].items():
        metrics = condition["aggregate"]
        seed_results = condition["seed_results"]

        rows.append(
            {
                "condition": condition_name.replace("_", " ").title(),
                "recall_at_5_mean": metrics["recall_at_5"]["mean"],
                "recall_at_5_std": metrics["recall_at_5"]["standard_deviation"],
                "recall_at_10_mean": metrics["recall_at_10"]["mean"],
                "recall_at_10_std": metrics["recall_at_10"]["standard_deviation"],
                "mrr_mean": metrics["mrr"]["mean"],
                "mrr_std": metrics["mrr"]["standard_deviation"],
                "edges_before": average_graph_value(
                    seed_results, "graph_before", "undirected_edges"
                ),
                "edges_after": average_graph_value(
                    seed_results, "graph_after", "undirected_edges"
                ),
                "edge_delta": average_graph_value(
                    seed_results, "graph_deltas", "undirected_edges"
                ),
                "average_degree_before": average_graph_value(
                    seed_results, "graph_before", "average_degree"
                ),
                "average_degree_after": average_graph_value(
                    seed_results, "graph_after", "average_degree"
                ),
                "average_edge_weight_before": average_graph_value(
                    seed_results, "graph_before", "average_edge_weight"
                ),
                "average_edge_weight_after": average_graph_value(
                    seed_results, "graph_after", "average_edge_weight"
                ),
            }
        )

    return rows


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {
                key: round(value, 6) if isinstance(value, float) else value
                for key, value in row.items()
            }
            for row in rows
        )


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_retrieval_comparison(results: dict, path: Path) -> None:
    systems = list(results["aggregate"])
    x_positions = list(range(len(systems)))
    bar_width = 0.22
    figure, axis = plt.subplots(figsize=(9, 5.2))

    for metric_index, (metric_name, metric_label) in enumerate(METRICS):
        values = [results["aggregate"][system][metric_name]["mean"] for system in systems]
        errors = [
            results["aggregate"][system][metric_name]["standard_deviation"]
            for system in systems
        ]
        positions = [
            position + (metric_index - 1) * bar_width
            for position in x_positions
        ]
        axis.bar(
            positions,
            values,
            width=bar_width,
            yerr=errors,
            capsize=3,
            label=metric_label,
            color=COLOURS[metric_index],
        )

    axis.set_title("Retrieval performance by system")
    axis.set_ylabel("Mean score")
    axis.set_ylim(0, 1)
    axis.set_xticks(x_positions, [SYSTEM_LABELS[system] for system in systems])
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=3, loc="upper center")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_adaptation_comparison(results: dict, path: Path) -> None:
    systems = ["initial_agent_graph", "adapted_agent_graph"]
    x_positions = list(range(len(METRICS)))
    bar_width = 0.34
    figure, axis = plt.subplots(figsize=(7.5, 5.2))

    for system_index, system_name in enumerate(systems):
        values = [
            results["aggregate"][system_name][metric_name]["mean"]
            for metric_name, _ in METRICS
        ]
        errors = [
            results["aggregate"][system_name][metric_name]["standard_deviation"]
            for metric_name, _ in METRICS
        ]
        positions = [
            position + (system_index - 0.5) * bar_width
            for position in x_positions
        ]
        axis.bar(
            positions,
            values,
            width=bar_width,
            yerr=errors,
            capsize=3,
            label=SYSTEM_LABELS[system_name],
            color=COLOURS[system_index + 1],
        )

    axis.set_title("Agent graph performance before and after adaptation")
    axis.set_ylabel("Mean score")
    axis.set_ylim(0, 1)
    axis.set_xticks(x_positions, [label for _, label in METRICS])
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_graph_structure_comparison(pruning_rows: list[dict], path: Path) -> None:
    labels = [row["condition"] for row in pruning_rows]
    x_positions = list(range(len(labels)))
    bar_width = 0.34
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.8))

    chart_data = [
        ("edges_before", "edges_after", "Edge count"),
        ("average_degree_before", "average_degree_after", "Average degree"),
    ]

    for axis, (before_key, after_key, title) in zip(axes, chart_data):
        before_values = [row[before_key] for row in pruning_rows]
        after_values = [row[after_key] for row in pruning_rows]
        before_positions = [position - bar_width / 2 for position in x_positions]
        after_positions = [position + bar_width / 2 for position in x_positions]

        axis.bar(
            before_positions,
            before_values,
            width=bar_width,
            label="Before adaptation",
            color=COLOURS[0],
        )
        axis.bar(
            after_positions,
            after_values,
            width=bar_width,
            label="After adaptation",
            color=COLOURS[2],
        )
        axis.set_title(title)
        axis.set_xticks(x_positions, labels)
        axis.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel("Mean across three seeds")
    axes[0].legend(frameon=False)
    figure.suptitle("Graph structure with and without pruning")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    main_results = load_json(MAIN_RESULTS_PATH)
    pruning_results = load_json(PRUNING_RESULTS_PATH)
    retrieval_rows = build_retrieval_rows(main_results)
    pruning_rows = build_pruning_rows(pruning_results)

    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    outputs = [
        REPORT_DIRECTORY / "retrieval_results.csv",
        REPORT_DIRECTORY / "pruning_ablation.csv",
        REPORT_DIRECTORY / "retrieval_comparison.png",
        REPORT_DIRECTORY / "adaptation_comparison.png",
        REPORT_DIRECTORY / "graph_structure.png",
    ]

    save_csv(outputs[0], retrieval_rows)
    save_csv(outputs[1], pruning_rows)
    save_retrieval_comparison(main_results, outputs[2])
    save_adaptation_comparison(main_results, outputs[3])
    save_graph_structure_comparison(pruning_rows, outputs[4])

    print("Created report outputs:")
    for output in outputs:
        print(f"- {output}")


if __name__ == "__main__":
    main()
