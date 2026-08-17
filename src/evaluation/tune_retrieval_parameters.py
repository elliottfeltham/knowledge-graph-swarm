import argparse
import json
from pathlib import Path

from common import (
    PROJECT_ROOT,
    build_faiss_index,
    load_embeddings,
    load_qrels,
    load_query_embeddings,
    load_queries,
)
from evaluation.create_results_report import save_parameter_tuning_report
from evaluation.experiment_config import DEFAULT_SEEDS
from evaluation.experiment_helpers import (
    EvaluationContext,
    evaluate_configuration,
    split_query_groups,
)
from evaluation.graph_state import file_checksum, graph_file_is_unchanged
from graph.config import INITIAL_WEIGHT_SCALE, NEIGHBOURS_PER_DOCUMENT
from graph.initialize_agent_graph import build_initial_agent_graph
from retrieval.agent_graph_retrieval import AGENT_GRAPH_PATH


DEFAULT_GRAPH_WEIGHTS = [0.05, 0.1, 0.2, 0.3, 0.5]
DEFAULT_SEED_COUNTS = [3, 5, 10]
RESULTS_PATH = PROJECT_ROOT / "results" / "parameter_tuning.json"


def select_best_configuration(configurations: list[dict]) -> dict:
    return max(
        configurations,
        key=lambda configuration: (
            configuration["adapted"]["recall_at_10"],
            configuration["adapted"]["mrr"],
            -configuration["graph_weight"],
        ),
    )


def print_summary(validation: dict, best: dict, held_out: dict) -> None:
    print("\nValidation results")
    print("=" * 60)
    print(
        f"FAISS: Recall@10 {validation['recall_at_10']:.4f}, "
        f"MRR {validation['mrr']:.4f}"
    )
    print(f"Best adapted graph: seed_k={best['seed_k']}, "
          f"graph_weight={best['graph_weight']}")
    print(f"Recall@10 {best['adapted']['recall_at_10']:.4f}, "
          f"MRR {best['adapted']['mrr']:.4f}")

    print("\nHeld-out evaluation")
    print("=" * 60)
    print(f"FAISS: Recall@10 {held_out['faiss']['recall_at_10']:.4f}, "
          f"MRR {held_out['faiss']['mrr']:.4f}")
    print(f"Selected adapted graph: Recall@10 "
          f"{held_out['adapted']['recall_at_10']:.4f}, "
          f"MRR {held_out['adapted']['mrr']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate how seed count and graph weight affect retrieval."
    )
    parser.add_argument("--adaptation-queries", type=int, default=100)
    parser.add_argument("--validation-queries", type=int, default=100)
    parser.add_argument("--evaluation-queries", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--graph-weights", type=float, nargs="+", default=DEFAULT_GRAPH_WEIGHTS
    )
    parser.add_argument(
        "--seed-counts", type=int, nargs="+", default=DEFAULT_SEED_COUNTS
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
    index = build_faiss_index(embeddings)
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

    print("Building a fresh initial agent graph...")
    initial_agents = build_initial_agent_graph(
        NEIGHBOURS_PER_DOCUMENT,
        INITIAL_WEIGHT_SCALE,
    )
    context = EvaluationContext(
        query_embeddings=query_embeddings,
        qrels=qrels,
        index=index,
        document_ids=document_ids,
        initial_agents=initial_agents,
        top_k=args.top_k,
    )
    application_graph_checksum = file_checksum(AGENT_GRAPH_PATH)
    configurations = []
    validation_faiss = None

    for seed_k in args.seed_counts:
        for graph_weight in args.graph_weights:
            print(f"Testing seed_k={seed_k}, graph_weight={graph_weight}...")
            evaluation = evaluate_configuration(
                context=context,
                query_splits=query_splits,
                query_group=1,
                seed_k=seed_k,
                graph_weight=graph_weight,
            )
            validation_faiss = evaluation["faiss"]
            configurations.append(
                {
                    "seed_k": seed_k,
                    "graph_weight": graph_weight,
                    "initial": evaluation["initial"],
                    "adapted": evaluation["adapted"],
                    "seed_results": evaluation["seed_results"],
                }
            )

    best = select_best_configuration(configurations)
    print("Evaluating the selected configuration on held-out queries...")
    held_out = evaluate_configuration(
        context=context,
        query_splits=query_splits,
        query_group=2,
        seed_k=best["seed_k"],
        graph_weight=best["graph_weight"],
    )

    application_graph_unchanged = graph_file_is_unchanged(
        AGENT_GRAPH_PATH,
        application_graph_checksum,
    )
    if not application_graph_unchanged:
        raise RuntimeError("Parameter evaluation modified the application graph")

    results = {
        "settings": {
            "seeds": args.seeds,
            "adaptation_queries": args.adaptation_queries,
            "validation_queries": args.validation_queries,
            "held_out_evaluation_queries": args.evaluation_queries,
            "top_k": args.top_k,
            "graph_weights": args.graph_weights,
            "seed_counts": args.seed_counts,
            "pruning_enabled": False,
        },
        "selection_rule": "Highest validation Recall@10, then MRR, then lower graph weight.",
        "application_graph_unchanged": application_graph_unchanged,
        "validation_faiss": validation_faiss,
        "best_configuration": {
            key: value for key, value in best.items() if key != "seed_results"
        },
        "held_out_evaluation": held_out,
        "configurations": configurations,
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    save_parameter_tuning_report(results)
    print_summary(validation_faiss, best, held_out)
    print(f"Application graph unchanged: {application_graph_unchanged}")
    print(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()
