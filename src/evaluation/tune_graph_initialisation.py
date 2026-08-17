import argparse
import json
from pathlib import Path

from agents.graph_statistics import calculate_graph_statistics
from common import (
    PROJECT_ROOT,
    build_faiss_index,
    load_embeddings,
    load_qrels,
    load_queries,
    load_query_embeddings,
)
from evaluation.create_results_report import save_graph_initialisation_report
from evaluation.experiment_config import DEFAULT_SEEDS
from evaluation.experiment_helpers import (
    EvaluationContext,
    evaluate_configuration,
    split_query_groups,
)
from evaluation.graph_state import file_checksum, graph_file_is_unchanged
from graph.initialize_agent_graph import build_initial_agent_graph
from retrieval.agent_graph_retrieval import AGENT_GRAPH_PATH


DEFAULT_NEIGHBOUR_COUNTS = [3, 5, 10, 15]
DEFAULT_WEIGHT_SCALES = [0.05, 0.1, 0.2, 0.3, 0.5]
RESULTS_PATH = PROJECT_ROOT / "results" / "graph_initialisation_tuning.json"


def build_parameter_grid(
    neighbour_counts: list[int],
    weight_scales: list[float],
) -> list[tuple[int, float]]:
    return [
        (neighbour_count, weight_scale)
        for neighbour_count in neighbour_counts
        for weight_scale in weight_scales
    ]


def select_best_configuration(configurations: list[dict]) -> dict:
    return max(
        configurations,
        key=lambda configuration: (
            configuration["adapted"]["recall_at_10"],
            configuration["adapted"]["mrr"],
            configuration["adapted"]["recall_at_5"],
            -configuration["neighbours_per_document"],
            -configuration["initial_weight_scale"],
        ),
    )


def print_summary(best: dict) -> None:
    print("\nSelected graph initialisation")
    print("=" * 60)
    print(f"Neighbours per document: {best['neighbours_per_document']}")
    print(f"Initial weight scale:    {best['initial_weight_scale']}")
    print(f"Validation Recall@10:    {best['adapted']['recall_at_10']:.4f}")
    print(f"Validation MRR:          {best['adapted']['mrr']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate nearest-neighbour counts and initial edge weights."
    )
    parser.add_argument("--adaptation-queries", type=int, default=100)
    parser.add_argument("--validation-queries", type=int, default=100)
    parser.add_argument("--evaluation-queries", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--neighbour-counts",
        type=int,
        nargs="+",
        default=DEFAULT_NEIGHBOUR_COUNTS,
    )
    parser.add_argument(
        "--weight-scales",
        type=float,
        nargs="+",
        default=DEFAULT_WEIGHT_SCALES,
    )
    parser.add_argument("--output-path", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    print("Loading evaluation data...")
    queries = load_queries()
    qrels = load_qrels()
    embeddings, document_ids = load_embeddings()
    encoded_queries, query_ids = load_query_embeddings()
    query_embeddings = {
        query_id: encoded_queries[position:position + 1]
        for position, query_id in enumerate(query_ids)
    }
    query_splits = {
        seed: split_query_groups(
            queries,
            seed,
            args.adaptation_queries,
            args.validation_queries,
            args.evaluation_queries,
        )
        for seed in args.seeds
    }
    required_query_ids = {
        query["query_id"]
        for split in query_splits.values()
        for group in split
        for query in group
    }

    if required_query_ids - set(query_embeddings):
        raise ValueError("Missing saved query embeddings. Run src/prepare_system.py.")

    index = build_faiss_index(embeddings)
    application_graph_checksum = file_checksum(AGENT_GRAPH_PATH)
    configurations = []
    validation_faiss = None

    for neighbour_count, weight_scale in build_parameter_grid(
        args.neighbour_counts,
        args.weight_scales,
    ):
        print(
            f"Testing neighbours={neighbour_count}, "
            f"initial_weight_scale={weight_scale}..."
        )
        initial_agents = build_initial_agent_graph(
            neighbours_per_doc=neighbour_count,
            initial_weight_scale=weight_scale,
        )
        context = EvaluationContext(
            query_embeddings=query_embeddings,
            qrels=qrels,
            index=index,
            document_ids=document_ids,
            initial_agents=initial_agents,
            top_k=args.top_k,
        )
        evaluation = evaluate_configuration(
            context=context,
            query_splits=query_splits,
            query_group=1,
            seed_k=3,
            graph_weight=0.05,
            update_k=3,
        )
        validation_faiss = evaluation["faiss"]
        configurations.append(
            {
                "neighbours_per_document": neighbour_count,
                "initial_weight_scale": weight_scale,
                "initial_graph_statistics": calculate_graph_statistics(initial_agents),
                "initial": evaluation["initial"],
                "adapted": evaluation["adapted"],
                "seed_results": evaluation["seed_results"],
            }
        )

    best = select_best_configuration(configurations)
    application_graph_unchanged = graph_file_is_unchanged(
        AGENT_GRAPH_PATH,
        application_graph_checksum,
    )

    if not application_graph_unchanged:
        raise RuntimeError("Graph initialisation tuning modified the application graph")

    results = {
        "settings": {
            "seeds": args.seeds,
            "adaptation_queries": args.adaptation_queries,
            "validation_queries": args.validation_queries,
            "reserved_evaluation_queries": args.evaluation_queries,
            "top_k": args.top_k,
            "seed_k": 3,
            "graph_weight": 0.05,
            "update_k": 3,
            "neighbour_counts": args.neighbour_counts,
            "initial_weight_scales": args.weight_scales,
            "pruning_enabled": False,
        },
        "selection_rule": (
            "Highest validation Recall@10, then MRR, Recall@5, fewer "
            "neighbours, and lower initial weight scale."
        ),
        "application_graph_unchanged": application_graph_unchanged,
        "validation_faiss": validation_faiss,
        "best_configuration": {
            key: value for key, value in best.items() if key != "seed_results"
        },
        "configurations": configurations,
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    save_graph_initialisation_report(results)
    print_summary(best)
    print(f"Application graph unchanged: {application_graph_unchanged}")
    print(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()
