import argparse
import json
from pathlib import Path

from agents import GraphUpdateConfig
from common import (
    PROJECT_ROOT,
    build_faiss_index,
    load_embeddings,
    load_qrels,
    load_queries,
    load_query_embeddings,
)
from evaluation.experiment_config import (
    DECAY_RATE,
    DEFAULT_SEEDS,
    OPTIMISED_CONFIG,
    STRENGTHEN_AMOUNT,
)
from evaluation.experiment_helpers import (
    EvaluationContext,
    evaluate_configuration,
    split_query_groups,
)
from evaluation.create_results_report import save_selective_strengthening_report
from evaluation.graph_state import file_checksum, graph_file_is_unchanged
from graph.config import INITIAL_WEIGHT_SCALE, NEIGHBOURS_PER_DOCUMENT
from graph.initialize_agent_graph import build_initial_agent_graph
from retrieval.agent_graph_retrieval import AGENT_GRAPH_PATH


DEFAULT_UPDATE_COUNTS = [3, 5, 10]
RESULTS_PATH = PROJECT_ROOT / "results" / "selective_strengthening.json"


def select_best_condition(conditions: list[dict]) -> dict:
    return max(
        conditions,
        key=lambda condition: (
            condition["validation"]["adapted"]["recall_at_10"],
            condition["validation"]["adapted"]["mrr"],
            -condition["update_k"],
        ),
    )


def average_graph_summary(seed_results: list[dict]) -> dict:
    seed_count = len(seed_results)
    update_keys = ("strengthened_edges", "created_edges", "pruned_edges")

    return {
        "updates": {
            key: sum(result["updates"][key] for result in seed_results)
            for key in update_keys
        },
        "final_undirected_edges": sum(
            result["graph_after"]["undirected_edges"] for result in seed_results
        ) / seed_count,
        "final_average_degree": sum(
            result["graph_after"]["average_degree"] for result in seed_results
        ) / seed_count,
    }


def print_summary(results: dict) -> None:
    print("\nSelective strengthening results")
    print("=" * 64)

    for condition in results["conditions"]:
        validation = condition["validation"]["adapted"]
        print(
            f"update_k={condition['update_k']}: "
            f"validation R@10 {validation['recall_at_10']:.4f}, "
            f"MRR {validation['mrr']:.4f}"
        )

    print(f"Selected update_k: {results['selected_update_k']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare how many retrieved documents should strengthen the graph."
    )
    parser.add_argument("--adaptation-queries", type=int, default=100)
    parser.add_argument("--validation-queries", type=int, default=100)
    parser.add_argument("--evaluation-queries", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--update-counts", type=int, nargs="+", default=DEFAULT_UPDATE_COUNTS
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

    print("Building a fresh initial agent graph...")
    context = EvaluationContext(
        query_embeddings=query_embeddings,
        qrels=qrels,
        index=build_faiss_index(embeddings),
        document_ids=document_ids,
        initial_agents=build_initial_agent_graph(
            NEIGHBOURS_PER_DOCUMENT,
            INITIAL_WEIGHT_SCALE,
        ),
        top_k=args.top_k,
    )
    application_graph_checksum = file_checksum(AGENT_GRAPH_PATH)
    conditions = []

    for update_k in args.update_counts:
        print(f"Testing update_k={update_k}...")
        validation = evaluate_configuration(
            context=context,
            query_splits=query_splits,
            query_group=1,
            seed_k=OPTIMISED_CONFIG["seed_k"],
            graph_weight=OPTIMISED_CONFIG["graph_weight"],
            update_config=GraphUpdateConfig(
                strengthen_amount=STRENGTHEN_AMOUNT,
                decay_rate=DECAY_RATE,
                enable_pruning=False,
                update_k=update_k,
            ),
        )
        conditions.append(
            {
                "update_k": update_k,
                "validation": validation,
                "adaptation": average_graph_summary(validation["seed_results"]),
            }
        )

    selected = select_best_condition(conditions)
    application_graph_unchanged = graph_file_is_unchanged(
        AGENT_GRAPH_PATH,
        application_graph_checksum,
    )
    if not application_graph_unchanged:
        raise RuntimeError("Selective strengthening evaluation modified the application graph")

    results = {
        "settings": {
            "seeds": args.seeds,
            "adaptation_queries": args.adaptation_queries,
            "validation_queries": args.validation_queries,
            "reserved_evaluation_queries": args.evaluation_queries,
            "top_k": args.top_k,
            "seed_k": OPTIMISED_CONFIG["seed_k"],
            "graph_weight": OPTIMISED_CONFIG["graph_weight"],
            "strengthen_amount": STRENGTHEN_AMOUNT,
            "decay_rate": DECAY_RATE,
            "pruning_enabled": False,
            "update_counts": args.update_counts,
        },
        "selection_rule": "Highest validation Recall@10, then MRR, then smaller update_k.",
        "selected_update_k": selected["update_k"],
        "application_graph_unchanged": application_graph_unchanged,
        "conditions": conditions,
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    save_selective_strengthening_report(results)
    print_summary(results)
    print(f"Application graph unchanged: {application_graph_unchanged}")
    print(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()
