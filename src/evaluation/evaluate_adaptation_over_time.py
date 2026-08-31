"""Measure graph structure and retrieval behaviour throughout adaptation."""

import argparse
import copy
import json
from pathlib import Path
from statistics import mean, stdev

from agents import GraphUpdateConfig
from agents.graph_statistics import calculate_graph_statistics
from common import (
    PROJECT_ROOT,
    build_faiss_index,
    load_embeddings,
    load_qrels,
    load_queries,
    load_query_embeddings,
)
from evaluation.create_results_report import save_adaptation_over_time_report
from evaluation.experiment_config import DEFAULT_SEEDS, OPTIMISED_CONFIG
from evaluation.experiment_helpers import (
    EvaluationContext,
    adapt_graph,
    calculate_faiss_metrics,
    evaluate_agent_graph,
    split_query_groups,
    summarise_agent_rows,
)
from evaluation.graph_state import file_checksum, graph_file_is_unchanged
from graph.config import INITIAL_WEIGHT_SCALE, NEIGHBOURS_PER_DOCUMENT
from graph.initialize_agent_graph import build_initial_agent_graph
from retrieval.agent_graph_retrieval import AGENT_GRAPH_PATH


DEFAULT_CHECKPOINTS = [0, 25, 50, 100, 150, 200, 250, 300]
RESULTS_PATH = PROJECT_ROOT / "results" / "adaptation_over_time.json"


def validate_checkpoints(checkpoints: list[int], adaptation_queries: int) -> list[int]:
    checkpoints = sorted(set(checkpoints))

    if not checkpoints or checkpoints[0] != 0:
        raise ValueError("Checkpoints must include 0")
    if checkpoints[-1] > adaptation_queries:
        raise ValueError("A checkpoint exceeds the number of adaptation queries")

    return checkpoints


def aggregate_checkpoints(seed_results: list[dict]) -> list[dict]:
    checkpoints = []

    for checkpoint_index, first_checkpoint in enumerate(seed_results[0]["checkpoints"]):
        matching = [result["checkpoints"][checkpoint_index] for result in seed_results]
        row = {"adaptation_queries": first_checkpoint["adaptation_queries"]}

        for section in ("metrics", "ranking_changes", "graph", "updates"):
            row[section] = {}

            for field in matching[0][section]:
                values = [checkpoint[section][field] for checkpoint in matching]
                row[section][field] = {
                    "mean": mean(values),
                    "standard_deviation": stdev(values) if len(values) > 1 else 0.0,
                }

        checkpoints.append(row)

    return checkpoints


def calculate_ranking_changes(
    initial_rows: list[dict],
    current_rows: list[dict],
) -> dict:
    initial_rankings = {
        row["query_id"]: row["retrieved_doc_ids"]
        for row in initial_rows
    }
    set_changes = 0
    order_changes = 0

    for row in current_rows:
        initial_ranking = initial_rankings[row["query_id"]]
        current_ranking = row["retrieved_doc_ids"]
        set_changes += set(initial_ranking) != set(current_ranking)
        order_changes += initial_ranking != current_ranking

    query_count = len(current_rows)
    return {
        "top_10_set_changes": set_changes,
        "top_10_set_change_percentage": 100 * set_changes / query_count,
        "top_10_order_changes": order_changes,
        "top_10_order_change_percentage": 100 * order_changes / query_count,
    }


def evaluate_seed(
    seed: int,
    context: EvaluationContext,
    adaptation_queries: list[dict],
    evaluation_queries: list[dict],
    checkpoints: list[int],
) -> dict:
    config = GraphUpdateConfig(
        strengthen_amount=OPTIMISED_CONFIG["strengthen_amount"],
        decay_rate=OPTIMISED_CONFIG["decay_rate"],
        enable_pruning=OPTIMISED_CONFIG["pruning_enabled"],
        update_k=OPTIMISED_CONFIG["update_k"],
    )
    agents = copy.deepcopy(context.initial_agents)
    faiss_metrics = calculate_faiss_metrics(evaluation_queries, context)
    update_totals = {
        "strengthened_edges": 0,
        "created_edges": 0,
        "decayed_edges": 0,
        "pruned_edges": 0,
    }
    checkpoint_results = []
    previous_checkpoint = 0
    initial_rows = None

    for checkpoint in checkpoints:
        query_batch = adaptation_queries[previous_checkpoint:checkpoint]
        batch_updates = adapt_graph(
            query_batch,
            context,
            agents,
            OPTIMISED_CONFIG["seed_k"],
            OPTIMISED_CONFIG["graph_weight"],
            update_config=config,
        )

        for field in update_totals:
            update_totals[field] += batch_updates[field]

        rows = evaluate_agent_graph(
            evaluation_queries,
            context,
            agents,
            OPTIMISED_CONFIG["seed_k"],
            OPTIMISED_CONFIG["graph_weight"],
            faiss_metrics,
        )
        if initial_rows is None:
            initial_rows = rows

        checkpoint_results.append(
            {
                "adaptation_queries": checkpoint,
                "metrics": summarise_agent_rows(rows),
                "ranking_changes": calculate_ranking_changes(initial_rows, rows),
                "graph": calculate_graph_statistics(agents),
                "updates": update_totals.copy(),
            }
        )
        previous_checkpoint = checkpoint

    return {
        "seed": seed,
        "faiss": {
            field: mean(metrics[field] for metrics in faiss_metrics.values())
            for field in ("recall_at_5", "recall_at_10", "mrr")
        },
        "checkpoints": checkpoint_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure retrieval performance as the graph adapts over time."
    )
    parser.add_argument("--adaptation-queries", type=int, default=300)
    parser.add_argument("--evaluation-queries", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=DEFAULT_CHECKPOINTS,
    )
    parser.add_argument("--output-path", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()
    checkpoints = validate_checkpoints(args.checkpoints, args.adaptation_queries)

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
            0,
            args.evaluation_queries,
        )
        for seed in args.seeds
    }
    required_query_ids = {
        query["query_id"]
        for groups in query_splits.values()
        for group in groups
        for query in group
    }

    if required_query_ids - set(query_embeddings):
        raise ValueError("Missing saved query embeddings. Run src/prepare_system.py.")

    initial_agents = build_initial_agent_graph(
        neighbours_per_doc=NEIGHBOURS_PER_DOCUMENT,
        initial_weight_scale=INITIAL_WEIGHT_SCALE,
    )
    context = EvaluationContext(
        query_embeddings=query_embeddings,
        qrels=qrels,
        index=build_faiss_index(embeddings),
        document_ids=document_ids,
        initial_agents=initial_agents,
        top_k=args.top_k,
    )
    application_graph_checksum = file_checksum(AGENT_GRAPH_PATH)
    seed_results = []

    for seed, groups in query_splits.items():
        print(f"Evaluating seed {seed}...")
        seed_results.append(
            evaluate_seed(
                seed,
                context,
                groups[0],
                groups[2],
                checkpoints,
            )
        )

    application_graph_unchanged = graph_file_is_unchanged(
        AGENT_GRAPH_PATH,
        application_graph_checksum,
    )
    if not application_graph_unchanged:
        raise RuntimeError("Adaptation experiment modified the application graph")

    results = {
        "settings": {
            "seeds": args.seeds,
            "adaptation_queries": args.adaptation_queries,
            "evaluation_queries": args.evaluation_queries,
            "checkpoints": checkpoints,
            "top_k": args.top_k,
            **OPTIMISED_CONFIG,
        },
        "application_graph_unchanged": application_graph_unchanged,
        "seed_results": seed_results,
        "aggregate_checkpoints": aggregate_checkpoints(seed_results),
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    save_adaptation_over_time_report(results)
    first = results["aggregate_checkpoints"][0]["metrics"]
    final = results["aggregate_checkpoints"][-1]["metrics"]
    print("\nAdaptation over time")
    print("=" * 60)
    print(f"Initial Recall@10: {first['recall_at_10']['mean']:.4f}")
    print(f"Final Recall@10:   {final['recall_at_10']['mean']:.4f}")
    print(f"Initial MRR:       {first['mrr']['mean']:.4f}")
    print(f"Final MRR:         {final['mrr']['mean']:.4f}")
    print(f"Results saved to:  {args.output_path}")


if __name__ == "__main__":
    main()
