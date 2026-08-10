import argparse
import json
import random
import time
from pathlib import Path
from statistics import mean, stdev

from agents import GraphUpdateConfig, load_agents, save_agents
from agents.graph_update import update_graph_after_retrieval
from common import (
    PROJECT_ROOT,
    build_faiss_index,
    faiss_retrieve,
    load_embeddings,
    load_qrels,
    load_query_embeddings,
    load_queries,
    load_static_graph,
    mean_reciprocal_rank,
    recall_at_k,
    static_graph_retrieve,
)
from evaluation.graph_state import (
    copy_agent_graph,
    create_initial_evaluation_graph,
    file_checksum,
    graph_file_is_unchanged,
)
from retrieval.agent_graph_retrieval import AGENT_GRAPH_PATH, agent_graph_retrieve


DEFAULT_SEEDS = [10, 20, 30]
RESULTS_PATH = PROJECT_ROOT / "results" / "main_evaluation.json"
PRUNING_ABLATION_PATH = PROJECT_ROOT / "results" / "pruning_ablation.json"
METRIC_NAMES = ["recall_at_5", "recall_at_10", "mrr", "retrieval_latency_ms"]


def split_queries(
    queries: list[dict],
    seed: int,
    adaptation_count: int,
    evaluation_count: int,
) -> tuple[list[dict], list[dict]]:
    required_queries = adaptation_count + evaluation_count

    if len(queries) < required_queries:
        raise ValueError(
            f"Need {required_queries} queries but only found {len(queries)}"
        )

    shuffled_queries = queries.copy()
    random.Random(seed).shuffle(shuffled_queries)

    adaptation_queries = shuffled_queries[:adaptation_count]
    evaluation_queries = shuffled_queries[
        adaptation_count:adaptation_count + evaluation_count
    ]

    return adaptation_queries, evaluation_queries


def calculate_query_metrics(
    retrieved_doc_ids: list[str],
    relevant_doc_ids: set[str],
    retrieval_latency_ms: float,
) -> dict:
    return {
        "recall_at_5": recall_at_k(retrieved_doc_ids, relevant_doc_ids, 5),
        "recall_at_10": recall_at_k(retrieved_doc_ids, relevant_doc_ids, 10),
        "mrr": mean_reciprocal_rank(retrieved_doc_ids, relevant_doc_ids),
        "retrieval_latency_ms": retrieval_latency_ms,
    }


def summarise_query_metrics(rows: list[dict]) -> dict:
    return {
        "queries_evaluated": len(rows),
        **{
            f"mean_{metric_name}": mean(row[metric_name] for row in rows)
            for metric_name in METRIC_NAMES
        },
    }


def aggregate_seed_results(seed_results: list[dict]) -> dict:
    systems = seed_results[0]["systems"]
    aggregate = {}

    for system_name in systems:
        aggregate[system_name] = {}

        for metric_name in METRIC_NAMES:
            values = [
                result["systems"][system_name][f"mean_{metric_name}"]
                for result in seed_results
            ]
            aggregate[system_name][metric_name] = {
                "mean": mean(values),
                "standard_deviation": stdev(values) if len(values) > 1 else 0.0,
            }

    return aggregate


def retrieve_with_latency(retrieval_function) -> tuple[list[str], float]:
    start_time = time.perf_counter()
    retrieved_doc_ids = [doc_id for doc_id, _ in retrieval_function()]
    latency_ms = (time.perf_counter() - start_time) * 1000

    return retrieved_doc_ids, latency_ms


def evaluate_seed(
    seed: int,
    adaptation_queries: list[dict],
    evaluation_queries: list[dict],
    query_embeddings: dict[str, object],
    qrels: dict[str, set[str]],
    index,
    document_ids: list[str],
    static_graph: dict,
    top_k: int,
    seed_k: int,
    graph_weight: float,
    pruning_enabled: bool,
    seed_directory: Path,
) -> dict:
    initial_graph_path = seed_directory / "initial_graph.jsonl"
    adapted_graph_path = seed_directory / "adapted_graph.jsonl"

    initial_agents = create_initial_evaluation_graph(initial_graph_path)
    copy_agent_graph(initial_graph_path, adapted_graph_path)
    adapted_agents = load_agents(adapted_graph_path)

    update_totals = {
        "strengthened_edges": 0,
        "created_edges": 0,
        "pruned_edges": 0,
    }

    for query_row in adaptation_queries:
        query_embedding = query_embeddings[query_row["query_id"]]
        retrieved_doc_ids = [
            doc_id
            for doc_id, _ in agent_graph_retrieve(
                index=index,
                query_embedding=query_embedding,
                document_ids=document_ids,
                agents=adapted_agents,
                top_k=top_k,
                seed_k=seed_k,
                graph_weight=graph_weight,
            )
        ]
        update_summary = update_graph_after_retrieval(
            agents=adapted_agents,
            retrieved_doc_ids=retrieved_doc_ids,
            config=GraphUpdateConfig(enable_pruning=pruning_enabled),
        )

        for key in update_totals:
            update_totals[key] += update_summary[key]

    save_agents(adapted_graph_path, adapted_agents)

    system_rows = {
        "faiss": [],
        "static_graph": [],
        "initial_agent_graph": [],
        "adapted_agent_graph": [],
    }

    for query_row in evaluation_queries:
        query_id = query_row["query_id"]
        query_embedding = query_embeddings[query_id]
        relevant_doc_ids = qrels.get(query_id, set())

        retrievals = {
            "faiss": lambda: faiss_retrieve(
                index=index,
                query_embedding=query_embedding,
                document_ids=document_ids,
                top_k=top_k,
            ),
            "static_graph": lambda: static_graph_retrieve(
                index=index,
                query_embedding=query_embedding,
                document_ids=document_ids,
                graph=static_graph,
                top_k=top_k,
                seed_k=seed_k,
                graph_activation_weight=graph_weight,
            ),
            "initial_agent_graph": lambda: agent_graph_retrieve(
                index=index,
                query_embedding=query_embedding,
                document_ids=document_ids,
                agents=initial_agents,
                top_k=top_k,
                seed_k=seed_k,
                graph_weight=graph_weight,
            ),
            "adapted_agent_graph": lambda: agent_graph_retrieve(
                index=index,
                query_embedding=query_embedding,
                document_ids=document_ids,
                agents=adapted_agents,
                top_k=top_k,
                seed_k=seed_k,
                graph_weight=graph_weight,
            ),
        }

        for system_name, retrieval_function in retrievals.items():
            retrieved_doc_ids, latency_ms = retrieve_with_latency(
                retrieval_function
            )
            system_rows[system_name].append(
                calculate_query_metrics(
                    retrieved_doc_ids=retrieved_doc_ids,
                    relevant_doc_ids=relevant_doc_ids,
                    retrieval_latency_ms=latency_ms,
                )
            )

    return {
        "seed": seed,
        "pruning_enabled": pruning_enabled,
        "adaptation_query_ids": [row["query_id"] for row in adaptation_queries],
        "evaluation_query_ids": [row["query_id"] for row in evaluation_queries],
        "paths": {
            "initial_graph": str(initial_graph_path),
            "adapted_graph": str(adapted_graph_path),
        },
        "updates": update_totals,
        "systems": {
            system_name: summarise_query_metrics(rows)
            for system_name, rows in system_rows.items()
        },
    }


def save_results(path: Path, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)


def print_results(aggregate: dict) -> None:
    print("\nMain evaluation results")
    print("=" * 80)

    for system_name, metrics in aggregate.items():
        print(f"\n{system_name}")
        print(f"Recall@5:  {metrics['recall_at_5']['mean']:.4f} "
              f"(+/- {metrics['recall_at_5']['standard_deviation']:.4f})")
        print(f"Recall@10: {metrics['recall_at_10']['mean']:.4f} "
              f"(+/- {metrics['recall_at_10']['standard_deviation']:.4f})")
        print(f"MRR:       {metrics['mrr']['mean']:.4f} "
              f"(+/- {metrics['mrr']['standard_deviation']:.4f})")
        print(f"Latency:   {metrics['retrieval_latency_ms']['mean']:.2f} ms")


def print_pruning_ablation(conditions: dict) -> None:
    print("\nPruning ablation results")
    print("=" * 80)

    for condition_name, condition in conditions.items():
        metrics = condition["aggregate"]
        print(f"\n{condition_name.replace('_', ' ')}")
        print(f"Recall@5:  {metrics['recall_at_5']['mean']:.4f} "
              f"(+/- {metrics['recall_at_5']['standard_deviation']:.4f})")
        print(f"Recall@10: {metrics['recall_at_10']['mean']:.4f} "
              f"(+/- {metrics['recall_at_10']['standard_deviation']:.4f})")
        print(f"MRR:       {metrics['mrr']['mean']:.4f} "
              f"(+/- {metrics['mrr']['standard_deviation']:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the main dissertation retrieval evaluation."
    )
    parser.add_argument("--adaptation-queries", type=int, default=100)
    parser.add_argument("--evaluation-queries", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed-k", type=int, default=5)
    parser.add_argument("--graph-weight", type=float, default=0.3)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--pruning-ablation",
        action="store_true",
        help="Compare adaptive retrieval with pruning enabled and disabled.",
    )
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args()

    print("Loading evaluation data...")
    queries = load_queries()
    qrels = load_qrels()

    query_splits = {}
    required_queries = {}

    for seed in args.seeds:
        query_splits[seed] = split_queries(
            queries=queries,
            seed=seed,
            adaptation_count=args.adaptation_queries,
            evaluation_count=args.evaluation_queries,
        )

        for query_group in query_splits[seed]:
            for query_row in query_group:
                required_queries[query_row["query_id"]] = query_row

    print("Loading embeddings and static graph...")
    embeddings, document_ids = load_embeddings()
    encoded_queries, query_ids = load_query_embeddings()
    static_graph = load_static_graph()
    all_query_embeddings = {
        query_id: encoded_queries[position:position + 1]
        for position, query_id in enumerate(query_ids)
    }
    missing_query_ids = set(required_queries) - set(all_query_embeddings)

    if missing_query_ids:
        raise ValueError(
            "Missing saved embeddings for experiment queries. "
            "Run src/prepare_system.py before evaluation."
        )

    query_embeddings = {
        query_id: all_query_embeddings[query_id]
        for query_id in required_queries
    }

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    application_graph_checksum = file_checksum(AGENT_GRAPH_PATH)
    if args.pruning_ablation:
        pruning_conditions = {
            "pruning_enabled": True,
            "pruning_disabled": False,
        }
        conditions = {}

        for condition_name, pruning_enabled in pruning_conditions.items():
            seed_results = []

            for seed in args.seeds:
                print(f"\nRunning {condition_name}, seed {seed}...")
                adaptation_queries, evaluation_queries = query_splits[seed]
                seed_results.append(
                    evaluate_seed(
                        seed=seed,
                        adaptation_queries=adaptation_queries,
                        evaluation_queries=evaluation_queries,
                        query_embeddings=query_embeddings,
                        qrels=qrels,
                        index=index,
                        document_ids=document_ids,
                        static_graph=static_graph,
                        top_k=args.top_k,
                        seed_k=args.seed_k,
                        graph_weight=args.graph_weight,
                        pruning_enabled=pruning_enabled,
                        seed_directory=(
                            PROJECT_ROOT
                            / "results"
                            / "pruning_ablation"
                            / condition_name
                            / f"seed_{seed}"
                        ),
                    )
                )

            conditions[condition_name] = {
                "seed_results": seed_results,
                "aggregate": aggregate_seed_results(seed_results)[
                    "adapted_agent_graph"
                ],
            }
    else:
        seed_results = []

        for seed in args.seeds:
            print(f"\nRunning seed {seed}...")
            adaptation_queries, evaluation_queries = query_splits[seed]
            seed_results.append(
                evaluate_seed(
                    seed=seed,
                    adaptation_queries=adaptation_queries,
                    evaluation_queries=evaluation_queries,
                    query_embeddings=query_embeddings,
                    qrels=qrels,
                    index=index,
                    document_ids=document_ids,
                    static_graph=static_graph,
                    top_k=args.top_k,
                    seed_k=args.seed_k,
                    graph_weight=args.graph_weight,
                    pruning_enabled=True,
                    seed_directory=(
                        PROJECT_ROOT
                        / "results"
                        / "main_evaluation"
                        / f"seed_{seed}"
                    ),
                )
            )

    application_graph_unchanged = graph_file_is_unchanged(
        AGENT_GRAPH_PATH,
        application_graph_checksum,
    )

    if not application_graph_unchanged:
        raise RuntimeError(
            "Evaluation unexpectedly modified the application agent graph"
        )

    settings = {
        "seeds": args.seeds,
        "adaptation_queries": args.adaptation_queries,
        "evaluation_queries": args.evaluation_queries,
        "top_k": args.top_k,
        "seed_k": args.seed_k,
        "graph_weight": args.graph_weight,
    }

    if args.pruning_ablation:
        output_path = args.output_path or PRUNING_ABLATION_PATH
        results = {
            "settings": settings,
            "application_agent_graph": str(AGENT_GRAPH_PATH),
            "application_graph_unchanged": application_graph_unchanged,
            "conditions": conditions,
        }
        print_pruning_ablation(conditions)
    else:
        output_path = args.output_path or RESULTS_PATH
        aggregate = aggregate_seed_results(seed_results)
        results = {
            "settings": {
                **settings,
                "pruning_enabled": True,
            },
            "application_agent_graph": str(AGENT_GRAPH_PATH),
            "application_graph_unchanged": application_graph_unchanged,
            "seed_results": seed_results,
            "aggregate": aggregate,
        }
        print_results(aggregate)

    save_results(output_path, results)
    print(f"\nApplication graph unchanged: {application_graph_unchanged}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
