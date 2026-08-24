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
PARAMETER_TUNING_CSV_PATH = REPORT_DIRECTORY / "parameter_tuning.csv"
PARAMETER_TUNING_HEATMAP_PATH = REPORT_DIRECTORY / "parameter_tuning_heatmap.png"
SELECTIVE_STRENGTHENING_CSV_PATH = REPORT_DIRECTORY / "selective_strengthening.csv"
SELECTIVE_STRENGTHENING_FIGURE_PATH = REPORT_DIRECTORY / "selective_strengthening.png"
GRAPH_INITIALISATION_CSV_PATH = REPORT_DIRECTORY / "graph_initialisation_tuning.csv"
GRAPH_INITIALISATION_FIGURE_PATH = REPORT_DIRECTORY / "graph_initialisation_tuning.png"
GRAPH_UPDATE_CSV_PATH = REPORT_DIRECTORY / "update_dynamics_tuning.csv"
GRAPH_UPDATE_HEATMAP_PATH = REPORT_DIRECTORY / "update_dynamics_heatmap.png"
ADAPTATION_OVER_TIME_CSV_PATH = REPORT_DIRECTORY / "adaptation_over_time.csv"
ADAPTATION_OVER_TIME_FIGURE_PATH = REPORT_DIRECTORY / "adaptation_over_time.png"
GRAPH_EVOLUTION_FIGURE_PATH = REPORT_DIRECTORY / "graph_evolution.png"
REPEATED_ADAPTATION_CSV_PATH = REPORT_DIRECTORY / "repeated_adaptation.csv"
REPEATED_ADAPTATION_FIGURE_PATH = REPORT_DIRECTORY / "repeated_adaptation.png"

SYSTEM_LABELS = {
    "faiss": "FAISS",
    "static_graph": "Static graph",
    "original_initial_agent_graph": "Original initial graph",
    "original_adapted_agent_graph": "Original adapted graph",
    "optimised_initial_agent_graph": "Optimised initial graph",
    "optimised_adapted_agent_graph": "Optimised adapted graph",
}

METRICS = [
    ("recall_at_5", "Recall@5"),
    ("recall_at_10", "Recall@10"),
    ("mrr", "MRR"),
]

COLOURS = ["#2878B5", "#E07A2D", "#2A9D8F", "#C44E52", "#8172B3"]


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
                "recall_at_10_mean": metrics["recall_at_10"]["mean"],
                "mrr_mean": metrics["mrr"]["mean"],
                "edges_before": average_graph_value(
                    seed_results, "graph_before", "undirected_edges"
                ),
                "edges_after": average_graph_value(
                    seed_results, "graph_after", "undirected_edges"
                ),
                "edge_delta": average_graph_value(
                    seed_results, "graph_deltas", "undirected_edges"
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
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
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


def plot_heatmap(
    axis,
    rows: list[dict],
    x_field: str,
    y_field: str,
    value_field: str,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    x_values = sorted({row[x_field] for row in rows})
    y_values = sorted({row[y_field] for row in rows})
    values = [
        [
            next(
                row[value_field]
                for row in rows
                if row[x_field] == x_value and row[y_field] == y_value
            )
            for x_value in x_values
        ]
        for y_value in y_values
    ]
    minimum = min(map(min, values))
    maximum = max(map(max, values))

    if minimum == maximum:
        margin = max(abs(minimum) * 0.001, 0.0005)
        minimum -= margin
        maximum += margin

    image = axis.imshow(
        values,
        aspect="auto",
        cmap="YlGnBu",
        vmin=minimum,
        vmax=maximum,
    )
    colour_midpoint = (minimum + maximum) / 2
    decimal_places = 4 if "weight" in value_field else 3

    for y_position, row_values in enumerate(values):
        for x_position, value in enumerate(row_values):
            axis.text(
                x_position,
                y_position,
                f"{value:.{decimal_places}f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > colour_midpoint else "black",
            )

    axis.set(
        title=title,
        xlabel=x_label,
        ylabel=y_label,
        xticks=range(len(x_values)),
        yticks=range(len(y_values)),
    )
    axis.set_xticklabels(x_values)
    axis.set_yticklabels(y_values)
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def save_parameter_tuning_report(results: dict) -> None:
    configurations = results["configurations"]
    rows = []

    for configuration in configurations:
        for graph_state in ("initial", "adapted"):
            rows.append(
                {
                    "graph_weight": configuration["graph_weight"],
                    "seed_k": configuration["seed_k"],
                    "graph_state": graph_state,
                    "recall_at_5": configuration[graph_state]["recall_at_5"],
                    "recall_at_10": configuration[graph_state]["recall_at_10"],
                    "mrr": configuration[graph_state]["mrr"],
                }
            )

    save_csv(PARAMETER_TUNING_CSV_PATH, rows)
    configure_plot_style()
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    adapted_rows = [row for row in rows if row["graph_state"] == "adapted"]

    for axis, metric, label in (
        (axes[0], "recall_at_10", "Recall@10"),
        (axes[1], "mrr", "MRR"),
    ):
        plot_heatmap(
            axis,
            adapted_rows,
            "graph_weight",
            "seed_k",
            metric,
            f"Adapted graph {label}",
            "Graph weight",
            "Seed documents",
        )

    figure.tight_layout()
    figure.savefig(PARAMETER_TUNING_HEATMAP_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_graph_initialisation_report(results: dict) -> None:
    configurations = results["configurations"]
    rows = [
        {
            "neighbours_per_document": configuration["neighbours_per_document"],
            "initial_weight_scale": configuration["initial_weight_scale"],
            "undirected_edges": configuration["initial_graph_statistics"][
                "undirected_edges"
            ],
            "average_edge_weight": configuration["initial_graph_statistics"][
                "average_edge_weight"
            ],
            "adapted_recall_at_5": configuration["adapted"]["recall_at_5"],
            "adapted_recall_at_10": configuration["adapted"]["recall_at_10"],
            "adapted_mrr": configuration["adapted"]["mrr"],
        }
        for configuration in configurations
    ]

    save_csv(GRAPH_INITIALISATION_CSV_PATH, rows)
    configure_plot_style()
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.8))

    for neighbour_count in sorted(
        {row["neighbours_per_document"] for row in configurations}
    ):
        matching = sorted(
            (
                row
                for row in configurations
                if row["neighbours_per_document"] == neighbour_count
            ),
            key=lambda row: row["initial_weight_scale"],
        )
        scales = [row["initial_weight_scale"] for row in matching]

        for axis, metric, label in (
            (axes[0], "recall_at_10", "Recall@10"),
            (axes[1], "mrr", "MRR"),
        ):
            axis.plot(
                scales,
                [row["adapted"][metric] for row in matching],
                marker="o",
                label=f"{neighbour_count} neighbours",
            )
            axis.set(
                title=f"Adapted graph {label}",
                xlabel="Initial weight scale",
                ylabel=label,
            )
            axis.grid(alpha=0.2)

    axes[0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(
        GRAPH_INITIALISATION_FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_graph_update_report(results: dict) -> None:
    configurations = results["configurations"]
    rows = [
        {
            "strengthen_amount": configuration["strengthen_amount"],
            "decay_rate": configuration["decay_rate"],
            "recall_at_5": configuration["adapted"]["recall_at_5"],
            "recall_at_10": configuration["adapted"]["recall_at_10"],
            "mrr": configuration["adapted"]["mrr"],
            "final_undirected_edges": mean(
                result["graph_after"]["undirected_edges"]
                for result in configuration["seed_results"]
            ),
            "final_average_edge_weight": mean(
                result["graph_after"]["average_edge_weight"]
                for result in configuration["seed_results"]
            ),
        }
        for configuration in configurations
    ]

    save_csv(GRAPH_UPDATE_CSV_PATH, rows)
    configure_plot_style()
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.8))

    for axis, metric, label in (
        (axes[0], "recall_at_10", "Recall@10"),
        (axes[1], "final_average_edge_weight", "Mean edge weight"),
    ):
        plot_heatmap(
            axis,
            rows,
            "decay_rate",
            "strengthen_amount",
            metric,
            label,
            "Decay rate",
            "Strengthening amount",
        )

    figure.tight_layout()
    figure.savefig(GRAPH_UPDATE_HEATMAP_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_adaptation_over_time_report(results: dict) -> None:
    rows = []

    for checkpoint in results["aggregate_checkpoints"]:
        row = {"adaptation_queries": checkpoint["adaptation_queries"]}

        for field, values in checkpoint["metrics"].items():
            row[f"{field}_mean"] = values["mean"]
            row[f"{field}_std"] = values["standard_deviation"]

        for field, values in checkpoint["ranking_changes"].items():
            row[f"{field}_mean"] = values["mean"]
            row[f"{field}_std"] = values["standard_deviation"]

        for field, values in checkpoint["graph"].items():
            row[f"graph_{field}_mean"] = values["mean"]

        for field, values in checkpoint["updates"].items():
            row[f"{field}_mean"] = values["mean"]

        rows.append(row)

    csv_rows = [
        {
            "adaptation_queries": row["adaptation_queries"],
            "recall_at_5": row["recall_at_5_mean"],
            "recall_at_10": row["recall_at_10_mean"],
            "mrr": row["mrr_mean"],
            "top_10_set_change_percentage": row[
                "top_10_set_change_percentage_mean"
            ],
            "top_10_order_change_percentage": row[
                "top_10_order_change_percentage_mean"
            ],
            "undirected_edges": row["graph_undirected_edges_mean"],
            "average_edge_weight": row["graph_average_edge_weight_mean"],
            "created_edges": row["created_edges_mean"],
            "decayed_edges": row["decayed_edges_mean"],
        }
        for row in rows
    ]
    save_csv(ADAPTATION_OVER_TIME_CSV_PATH, csv_rows)
    configure_plot_style()
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    x_values = [row["adaptation_queries"] for row in rows]

    for metric, label in METRICS:
        axes[0].plot(
            x_values,
            [row[f"{metric}_mean"] for row in rows],
            marker="o",
            label=label,
        )

    axes[0].set(
        title="Retrieval during adaptation",
        xlabel="Adaptation queries",
        ylabel="Mean score",
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)
    axes[1].plot(
        x_values,
        [row["top_10_set_change_percentage_mean"] for row in rows],
        marker="o",
        label="Different documents",
        color=COLOURS[1],
    )
    axes[1].plot(
        x_values,
        [row["top_10_order_change_percentage_mean"] for row in rows],
        marker="o",
        label="Different order",
        color=COLOURS[2],
    )
    axes[1].set(
        title="Ranking changes from checkpoint 0",
        xlabel="Adaptation queries",
        ylabel="Evaluation queries changed (%)",
    )
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False)
    axes[2].plot(
        x_values,
        [row["graph_average_edge_weight_mean"] for row in rows],
        marker="o",
        color=COLOURS[2],
    )
    axes[2].set(
        title="Graph weight during adaptation",
        xlabel="Adaptation queries",
        ylabel="Mean edge weight",
    )
    axes[2].grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(ADAPTATION_OVER_TIME_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].plot(
        x_values,
        [row["graph_undirected_edges_mean"] for row in rows],
        marker="o",
        color=COLOURS[0],
    )
    axes[0].set(
        title="Graph size",
        xlabel="Adaptation queries",
        ylabel="Undirected edges",
    )

    for field, label, colour in (
        ("graph_average_edge_weight_mean", "Mean weight", COLOURS[1]),
        ("graph_max_edge_weight_mean", "Maximum weight", COLOURS[2]),
    ):
        axes[1].plot(
            x_values,
            [row[field] for row in rows],
            marker="o",
            label=label,
            color=colour,
        )

    axes[1].set(
        title="Relationship weights",
        xlabel="Adaptation queries",
        ylabel="Edge weight",
    )
    axes[1].legend(frameon=False)

    strengthened_values = [
        row["strengthened_edges_mean"] - row["created_edges_mean"]
        for row in rows
    ]
    axes[2].plot(
        x_values,
        strengthened_values,
        marker="o",
        label="Strengthened",
        color=COLOURS[1],
    )
    update_series = (
        ("created_edges_mean", "Created", COLOURS[2]),
        ("decayed_edges_mean", "Decayed", COLOURS[4]),
        ("pruned_edges_mean", "Pruned", COLOURS[3]),
    )

    for field, label, colour in update_series:
        if field in rows[0] and any(row[field] != 0 for row in rows):
            axes[2].plot(
                x_values,
                [row[field] for row in rows],
                marker="o",
                label=label,
                color=colour,
            )

    axes[2].set(
        title="Cumulative relationship updates",
        xlabel="Adaptation queries",
        ylabel="Edge updates",
    )
    axes[2].legend(frameon=False)

    for axis in axes:
        axis.grid(alpha=0.2)

    figure.tight_layout()
    figure.savefig(GRAPH_EVOLUTION_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_repeated_adaptation_report(results: dict) -> None:
    rows = []

    for condition_name, passes in results["aggregate"].items():
        for pass_result in passes:
            row = {
                "condition": condition_name,
                "interactions": pass_result["interactions"],
                "rankings_changed": pass_result["rankings_changed"]["mean"],
                "recall_at_5": pass_result["metrics"]["recall_at_5"]["mean"],
                "recall_at_10": pass_result["metrics"]["recall_at_10"]["mean"],
                "mrr": pass_result["metrics"]["mrr"]["mean"],
                "average_edge_weight": pass_result["graph"][
                    "average_edge_weight"
                ]["mean"],
                "max_edge_weight": pass_result["graph"]["max_edge_weight"][
                    "mean"
                ],
            }
            rows.append(row)

    save_csv(REPEATED_ADAPTATION_CSV_PATH, rows)
    configure_plot_style()
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.8))

    for condition_name, passes in results["aggregate"].items():
        x_values = [row["interactions"] for row in passes]
        label = condition_name.replace("_", " ").title()
        axes[0].plot(
            x_values,
            [row["metrics"]["recall_at_5"]["mean"] for row in passes],
            marker="o",
            label=label,
        )
        axes[1].plot(
            x_values,
            [row["rankings_changed"]["mean"] for row in passes],
            marker="o",
            label=label,
        )

    for axis, title, label in (
        (axes[0], "Recall during repeated adaptation", "Recall@5"),
        (axes[1], "Rankings changed from pass 0", "Evaluation queries"),
    ):
        axis.set(title=title, xlabel="Adaptation interactions", ylabel=label)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)

    figure.tight_layout()
    figure.savefig(REPEATED_ADAPTATION_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_selective_strengthening_report(results: dict) -> None:
    rows = []

    for condition in results["conditions"]:
        rows.append(
            {
                "update_k": condition["update_k"],
                "recall_at_5": condition["validation"]["adapted"]["recall_at_5"],
                "recall_at_10": condition["validation"]["adapted"]["recall_at_10"],
                "mrr": condition["validation"]["adapted"]["mrr"],
            }
        )

    save_csv(SELECTIVE_STRENGTHENING_CSV_PATH, rows)
    configure_plot_style()
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    conditions = sorted(results["conditions"], key=lambda row: row["update_k"])
    update_counts = [condition["update_k"] for condition in conditions]

    for axis, metric, label in (
        (axes[0], "recall_at_10", "Recall@10"),
        (axes[1], "mrr", "MRR"),
    ):
        axis.plot(
            update_counts,
            [condition["validation"]["adapted"][metric] for condition in conditions],
            marker="o",
            label="Validation",
        )
        axis.set(title=label, xlabel="Documents used for strengthening", ylabel=label)
        axis.set_xticks(update_counts)
        axis.grid(alpha=0.2)

    figure.tight_layout()
    figure.savefig(SELECTIVE_STRENGTHENING_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_retrieval_comparison(results: dict, path: Path) -> None:
    systems = list(results["aggregate"])
    x_positions = list(range(len(systems)))
    bar_width = 0.22
    figure, axis = plt.subplots(figsize=(12, 5.5))

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
    axis.set_xticks(
        x_positions,
        [SYSTEM_LABELS[system] for system in systems],
        rotation=15,
        ha="right",
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=3, loc="upper center")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_ranking_impact(results: dict, path: Path) -> None:
    impact = results["ranking_impact"]
    metric_names = ["recall_at_5", "recall_at_10", "mrr"]
    labels = ["Recall@5", "Recall@10", "Reciprocal rank"]
    x_positions = list(range(len(metric_names)))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    bar_width = 0.24

    for outcome_index, (outcome, label, colour) in enumerate(
        (
            ("improved", "Improved", COLOURS[2]),
            ("unchanged", "Unchanged", "#999999"),
            ("worsened", "Worsened", COLOURS[3]),
        )
    ):
        positions = [
            position + (outcome_index - 1) * bar_width
            for position in x_positions
        ]
        values = [
            impact["outcomes"][metric_name][outcome]
            for metric_name in metric_names
        ]
        axes[0].bar(positions, values, width=bar_width, label=label, color=colour)

    axes[0].set(
        title="Optimised graph compared with FAISS",
        ylabel="Query runs",
        xticks=x_positions,
    )
    axes[0].set_xticklabels(labels)
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)

    changed_deltas = sorted(
        delta for delta in impact["mrr_deltas"] if delta != 0
    )
    colours = [
        COLOURS[3] if delta < 0 else COLOURS[2]
        for delta in changed_deltas
    ]
    axes[1].bar(range(len(changed_deltas)), changed_deltas, color=colours, width=1)
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set(
        title="Queries with a changed relevant-document rank",
        xlabel="Changed query runs, ordered by effect",
        ylabel="Change in reciprocal rank",
    )
    axes[1].grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_graph_structure_comparison(pruning_rows: list[dict], path: Path) -> None:
    labels = [row["condition"] for row in pruning_rows]
    x_positions = list(range(len(labels)))
    bar_width = 0.34
    figure, axis = plt.subplots(figsize=(7, 4.8))
    before_positions = [position - bar_width / 2 for position in x_positions]
    after_positions = [position + bar_width / 2 for position in x_positions]

    axis.bar(
        before_positions,
        [row["edges_before"] for row in pruning_rows],
        width=bar_width,
        label="Before adaptation",
        color=COLOURS[0],
    )
    axis.bar(
        after_positions,
        [row["edges_after"] for row in pruning_rows],
        width=bar_width,
        label="After adaptation",
        color=COLOURS[2],
    )
    axis.set(
        title="Graph size with and without pruning",
        ylabel="Mean undirected edges",
        xticks=x_positions,
    )
    axis.set_xticklabels(labels)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
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
        REPORT_DIRECTORY / "graph_structure.png",
        REPORT_DIRECTORY / "ranking_impact.png",
    ]

    save_csv(outputs[0], retrieval_rows)
    save_csv(
        outputs[1],
        [
            {
                key: row[key]
                for key in (
                    "condition",
                    "recall_at_5_mean",
                    "recall_at_10_mean",
                    "mrr_mean",
                    "edges_after",
                    "edge_delta",
                    "average_edge_weight_after",
                )
            }
            for row in pruning_rows
        ],
    )
    save_retrieval_comparison(main_results, outputs[2])
    save_graph_structure_comparison(pruning_rows, outputs[3])
    save_ranking_impact(main_results, outputs[4])

    print("Created report outputs:")
    for output in outputs:
        print(f"- {output}")


if __name__ == "__main__":
    main()
