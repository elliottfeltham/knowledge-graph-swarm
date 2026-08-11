import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from agents import GraphUpdateConfig
from agents.graph_statistics import calculate_graph_statistics
from agents.graph_update import update_graph_after_retrieval
from common import (
    PROJECT_ROOT,
    build_faiss_index,
    faiss_retrieve,
    load_embeddings,
    load_qrels,
    load_query_embeddings,
    load_queries,
    mean_reciprocal_rank,
    recall_at_k,
)
from evaluation.graph_state import file_checksum, graph_file_is_unchanged
from evaluation.create_results_report import save_parameter_tuning_report
from graph.initialize_agent_graph import build_initial_agent_graph
from retrieval.agent_graph_retrieval import AGENT_GRAPH_PATH, agent_graph_retrieve


DEFAULT_SEEDS = [10, 20, 30]
DEFAULT_GRAPH_WEIGHTS = [0.05, 0.1, 0.2, 0.3, 0.5]
DEFAULT_SEED_COUNTS = [3, 5, 10]
RESULTS_PATH = PROJECT_ROOT / "results" / "parameter_tuning.json"


@dataclass
class EvaluationContext:
    query_embeddings: dict
    qrels: dict[str, set[str]]
    index: object
    document_ids: list[str]
    initial_agents: dict
    top_k: int


def split_query_groups(
    queries: list[dict],
    seed: int,
    adaptation_count: int,
    validation_count: int,
    evaluation_count: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    required_queries = adaptation_count + validation_count + evaluation_count

    if len(queries) < required_queries:
        raise ValueError(
            f"Need {required_queries} queries but only found {len(queries)}"
        )

    shuffled_queries = queries.copy()
    random.Random(seed).shuffle(shuffled_queries)
    validation_start = adaptation_count
    evaluation_start = validation_start + validation_count
    evaluation_end = evaluation_start + evaluation_count

    return (
        shuffled_queries[:validation_start],
        shuffled_queries[validation_start:evaluation_start],
        shuffled_queries[evaluation_start:evaluation_end],
    )


def calculate_metrics(
    retrieved_doc_ids: list[str],
    relevant_doc_ids: set[str],
) -> dict[str, float]:
    return {
        "recall_at_5": recall_at_k(retrieved_doc_ids, relevant_doc_ids, 5),
        "recall_at_10": recall_at_k(retrieved_doc_ids, relevant_doc_ids, 10),
        "mrr": mean_reciprocal_rank(retrieved_doc_ids, relevant_doc_ids),
    }


def compare_query_metrics(agent_metrics: dict, faiss_metrics: dict) -> str:
    agent_score = (agent_metrics["recall_at_10"], agent_metrics["mrr"])
    faiss_score = (faiss_metrics["recall_at_10"], faiss_metrics["mrr"])

    if agent_score > faiss_score:
        return "improved"
    if agent_score < faiss_score:
        return "worsened"
    return "unchanged"


def summarise_metrics(metric_rows: list[dict]) -> dict:
    return {
        "queries_evaluated": len(metric_rows),
        "recall_at_5": mean(row["recall_at_5"] for row in metric_rows),
        "recall_at_10": mean(row["recall_at_10"] for row in metric_rows),
        "mrr": mean(row["mrr"] for row in metric_rows),
    }


def summarise_agent_rows(rows: list[dict]) -> dict:
    return {
        **summarise_metrics([row["metrics"] for row in rows]),
        "queries_improved": sum(row["outcome"] == "improved" for row in rows),
        "queries_unchanged": sum(row["outcome"] == "unchanged" for row in rows),
        "queries_worsened": sum(row["outcome"] == "worsened" for row in rows),
    }


def calculate_faiss_metrics(
    queries: list[dict],
    context: EvaluationContext,
) -> dict[str, dict]:
    results = {}

    for query in queries:
        query_id = query["query_id"]
        retrieved_doc_ids = [
            doc_id
            for doc_id, _ in faiss_retrieve(
                index=context.index,
                query_embedding=context.query_embeddings[query_id],
                document_ids=context.document_ids,
                top_k=context.top_k,
            )
        ]
        results[query_id] = calculate_metrics(
            retrieved_doc_ids,
            context.qrels.get(query_id, set()),
        )

    return results


def evaluate_agent_graph(
    queries: list[dict],
    context: EvaluationContext,
    agents,
    seed_k: int,
    graph_weight: float,
    faiss_metrics: dict[str, dict],
) -> list[dict]:
    rows = []

    for query in queries:
        query_id = query["query_id"]
        retrieved_doc_ids = [
            doc_id
            for doc_id, _ in agent_graph_retrieve(
                index=context.index,
                query_embedding=context.query_embeddings[query_id],
                document_ids=context.document_ids,
                agents=agents,
                top_k=context.top_k,
                seed_k=seed_k,
                graph_weight=graph_weight,
            )
        ]
        metrics = calculate_metrics(
            retrieved_doc_ids,
            context.qrels.get(query_id, set()),
        )

        rows.append(
            {
                "query_id": query_id,
                "metrics": metrics,
                "outcome": compare_query_metrics(metrics, faiss_metrics[query_id]),
            }
        )

    return rows


def adapt_graph(
    queries: list[dict],
    context: EvaluationContext,
    agents,
    seed_k: int,
    graph_weight: float,
    update_k: int | None = None,
) -> dict:
    totals = {
        "strengthened_edges": 0,
        "created_edges": 0,
        "pruned_edges": 0,
    }
    update_config = GraphUpdateConfig(
        enable_pruning=False,
        update_k=update_k,
    )

    for query in queries:
        retrieved_doc_ids = [
            doc_id
            for doc_id, _ in agent_graph_retrieve(
                index=context.index,
                query_embedding=context.query_embeddings[query["query_id"]],
                document_ids=context.document_ids,
                agents=agents,
                top_k=context.top_k,
                seed_k=seed_k,
                graph_weight=graph_weight,
            )
        ]
        update_summary = update_graph_after_retrieval(
            agents=agents,
            retrieved_doc_ids=retrieved_doc_ids,
            config=update_config,
        )

        for key in totals:
            totals[key] += update_summary[key]

    return totals


def aggregate_seed_summaries(seed_results: list[dict], graph_state: str) -> dict:
    summaries = [result[graph_state] for result in seed_results]

    return {
        "queries_evaluated": sum(row["queries_evaluated"] for row in summaries),
        "recall_at_5": mean(row["recall_at_5"] for row in summaries),
        "recall_at_10": mean(row["recall_at_10"] for row in summaries),
        "mrr": mean(row["mrr"] for row in summaries),
        "queries_improved": sum(row["queries_improved"] for row in summaries),
        "queries_unchanged": sum(row["queries_unchanged"] for row in summaries),
        "queries_worsened": sum(row["queries_worsened"] for row in summaries),
    }


def evaluate_configuration(
    context: EvaluationContext,
    query_splits: dict,
    query_group: int,
    seed_k: int,
    graph_weight: float,
    update_k: int | None = None,
) -> dict:
    seed_results = []
    all_faiss_metrics = []

    for seed, query_groups in query_splits.items():
        adaptation_queries = query_groups[0]
        evaluation_queries = query_groups[query_group]
        faiss_metrics = calculate_faiss_metrics(
            queries=evaluation_queries,
            context=context,
        )
        initial_rows = evaluate_agent_graph(
            queries=evaluation_queries,
            context=context,
            agents=context.initial_agents,
            seed_k=seed_k,
            graph_weight=graph_weight,
            faiss_metrics=faiss_metrics,
        )
        adapted_agents = copy.deepcopy(context.initial_agents)
        graph_before = calculate_graph_statistics(adapted_agents)
        updates = adapt_graph(
            queries=adaptation_queries,
            context=context,
            agents=adapted_agents,
            seed_k=seed_k,
            graph_weight=graph_weight,
            update_k=update_k,
        )
        graph_after = calculate_graph_statistics(adapted_agents)
        adapted_rows = evaluate_agent_graph(
            queries=evaluation_queries,
            context=context,
            agents=adapted_agents,
            seed_k=seed_k,
            graph_weight=graph_weight,
            faiss_metrics=faiss_metrics,
        )

        all_faiss_metrics.extend(faiss_metrics.values())
        seed_results.append(
            {
                "seed": seed,
                "updates": updates,
                "graph_before": graph_before,
                "graph_after": graph_after,
                "initial": summarise_agent_rows(initial_rows),
                "adapted": summarise_agent_rows(adapted_rows),
            }
        )

    return {
        "faiss": summarise_metrics(all_faiss_metrics),
        "initial": aggregate_seed_summaries(seed_results, "initial"),
        "adapted": aggregate_seed_summaries(seed_results, "adapted"),
        "seed_results": seed_results,
    }


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
    initial_agents = build_initial_agent_graph(5, 0.2)
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
