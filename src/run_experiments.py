import argparse
import json
import random
import time
from pathlib import Path
from statistics import mean, stdev

from agents import GraphUpdateConfig, load_agents, save_agents
from agents.graph_statistics import calculate_graph_deltas, calculate_graph_statistics
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
ORIGINAL_CONFIG = {
    "seed_k": 5,
    "graph_weight": 0.3,
    "pruning_enabled": True,
}
OPTIMISED_CONFIG = {
    "seed_k": 3,
    "graph_weight": 0.05,
    "pruning_enabled": False,
    "update_k": 10,
}
SELECTIVE_CONFIG = {
    **OPTIMISED_CONFIG,
    "update_k": 3,
}
RESULTS_PATH = PROJECT_ROOT / "results" / "main_evaluation.json"
PRUNING_ABLATION_PATH = PROJECT_ROOT / "results" / "pruning_ablation.json"
METRIC_NAMES = ["recall_at_5", "recall_at_10", "mrr", "retrieval_latency_ms"]


def project_relative_path(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def split_queries(
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
    aggregate = {}

    for system_name in seed_results[0]["systems"]:
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


def adapt_graph(
    initial_graph_path: Path,
    adapted_graph_path: Path,
    adaptation_queries: list[dict],
    query_embeddings: dict[str, object],
    index,
    document_ids: list[str],
    top_k: int,
    config: dict,
) -> tuple[dict, dict]:
    copy_agent_graph(initial_graph_path, adapted_graph_path)
    agents = load_agents(adapted_graph_path)
    graph_before = calculate_graph_statistics(agents)
    update_totals = {
        "strengthened_edges": 0,
        "created_edges": 0,
        "pruned_edges": 0,
    }

    for query in adaptation_queries:
        retrieved_doc_ids = [
            doc_id
            for doc_id, _ in agent_graph_retrieve(
                index=index,
                query_embedding=query_embeddings[query["query_id"]],
                document_ids=document_ids,
                agents=agents,
                top_k=top_k,
                seed_k=config["seed_k"],
                graph_weight=config["graph_weight"],
            )
        ]
        update_summary = update_graph_after_retrieval(
            agents=agents,
            retrieved_doc_ids=retrieved_doc_ids,
            config=GraphUpdateConfig(
                enable_pruning=config["pruning_enabled"],
                update_k=config.get("update_k"),
            ),
        )

        for key in update_totals:
            update_totals[key] += update_summary[key]

    save_agents(adapted_graph_path, agents)
    graph_after = calculate_graph_statistics(agents)
    adaptation_summary = {
        "settings": config,
        "updates": update_totals,
        "graph_before": graph_before,
        "graph_after": graph_after,
        "graph_deltas": calculate_graph_deltas(graph_before, graph_after),
    }
    return agents, adaptation_summary


def evaluate_retrievals(
    evaluation_queries: list[dict],
    query_embeddings: dict[str, object],
    qrels: dict[str, set[str]],
    retrievals_for_query,
) -> dict:
    system_rows = {}

    for query in evaluation_queries:
        query_id = query["query_id"]
        relevant_doc_ids = qrels.get(query_id, set())

        for system_name, retrieval_function in retrievals_for_query(query_id).items():
            retrieved_doc_ids, latency_ms = retrieve_with_latency(retrieval_function)
            system_rows.setdefault(system_name, []).append(
                calculate_query_metrics(
                    retrieved_doc_ids,
                    relevant_doc_ids,
                    latency_ms,
                )
            )

    return {
        system_name: summarise_query_metrics(rows)
        for system_name, rows in system_rows.items()
    }


def evaluate_final_seed(
    seed: int,
    adaptation_queries: list[dict],
    validation_queries: list[dict],
    evaluation_queries: list[dict],
    query_embeddings: dict[str, object],
    qrels: dict[str, set[str]],
    index,
    document_ids: list[str],
    static_graph: dict,
    top_k: int,
    seed_directory: Path,
) -> dict:
    initial_graph_path = seed_directory / "initial_graph.jsonl"
    original_graph_path = seed_directory / "original_adapted_graph.jsonl"
    optimised_graph_path = seed_directory / "optimised_adapted_graph.jsonl"
    selective_graph_path = seed_directory / "selective_adapted_graph.jsonl"
    initial_agents = create_initial_evaluation_graph(initial_graph_path)
    original_agents, original_adaptation = adapt_graph(
        initial_graph_path,
        original_graph_path,
        adaptation_queries,
        query_embeddings,
        index,
        document_ids,
        top_k,
        ORIGINAL_CONFIG,
    )
    optimised_agents, optimised_adaptation = adapt_graph(
        initial_graph_path,
        optimised_graph_path,
        adaptation_queries,
        query_embeddings,
        index,
        document_ids,
        top_k,
        OPTIMISED_CONFIG,
    )
    selective_agents, selective_adaptation = adapt_graph(
        initial_graph_path,
        selective_graph_path,
        adaptation_queries,
        query_embeddings,
        index,
        document_ids,
        top_k,
        SELECTIVE_CONFIG,
    )

    def retrievals_for_query(query_id: str) -> dict:
        query_embedding = query_embeddings[query_id]
        return {
            "faiss": lambda: faiss_retrieve(
                index, query_embedding, document_ids, top_k
            ),
            "static_graph": lambda: static_graph_retrieve(
                index=index,
                query_embedding=query_embedding,
                document_ids=document_ids,
                graph=static_graph,
                top_k=top_k,
                seed_k=ORIGINAL_CONFIG["seed_k"],
                graph_activation_weight=ORIGINAL_CONFIG["graph_weight"],
            ),
            "original_initial_agent_graph": lambda: agent_graph_retrieve(
                index,
                query_embedding,
                document_ids,
                initial_agents,
                top_k,
                ORIGINAL_CONFIG["seed_k"],
                ORIGINAL_CONFIG["graph_weight"],
            ),
            "original_adapted_agent_graph": lambda: agent_graph_retrieve(
                index,
                query_embedding,
                document_ids,
                original_agents,
                top_k,
                ORIGINAL_CONFIG["seed_k"],
                ORIGINAL_CONFIG["graph_weight"],
            ),
            "optimised_initial_agent_graph": lambda: agent_graph_retrieve(
                index,
                query_embedding,
                document_ids,
                initial_agents,
                top_k,
                OPTIMISED_CONFIG["seed_k"],
                OPTIMISED_CONFIG["graph_weight"],
            ),
            "optimised_adapted_agent_graph": lambda: agent_graph_retrieve(
                index,
                query_embedding,
                document_ids,
                optimised_agents,
                top_k,
                OPTIMISED_CONFIG["seed_k"],
                OPTIMISED_CONFIG["graph_weight"],
            ),
            "selective_adapted_agent_graph": lambda: agent_graph_retrieve(
                index,
                query_embedding,
                document_ids,
                selective_agents,
                top_k,
                SELECTIVE_CONFIG["seed_k"],
                SELECTIVE_CONFIG["graph_weight"],
            ),
        }

    return {
        "seed": seed,
        "adaptation_query_ids": [row["query_id"] for row in adaptation_queries],
        "validation_query_ids": [row["query_id"] for row in validation_queries],
        "evaluation_query_ids": [row["query_id"] for row in evaluation_queries],
        "paths": {
            "initial_graph": project_relative_path(initial_graph_path),
            "original_adapted_graph": project_relative_path(original_graph_path),
            "optimised_adapted_graph": project_relative_path(optimised_graph_path),
            "selective_adapted_graph": project_relative_path(selective_graph_path),
        },
        "adaptations": {
            "original": original_adaptation,
            "optimised": optimised_adaptation,
            "selective": selective_adaptation,
        },
        "systems": evaluate_retrievals(
            evaluation_queries,
            query_embeddings,
            qrels,
            retrievals_for_query,
        ),
    }


def evaluate_ablation_seed(
    seed: int,
    adaptation_queries: list[dict],
    evaluation_queries: list[dict],
    query_embeddings: dict[str, object],
    qrels: dict[str, set[str]],
    index,
    document_ids: list[str],
    top_k: int,
    pruning_enabled: bool,
    seed_directory: Path,
) -> dict:
    initial_graph_path = seed_directory / "initial_graph.jsonl"
    adapted_graph_path = seed_directory / "adapted_graph.jsonl"
    create_initial_evaluation_graph(initial_graph_path)
    config = {**OPTIMISED_CONFIG, "pruning_enabled": pruning_enabled}
    agents, adaptation = adapt_graph(
        initial_graph_path,
        adapted_graph_path,
        adaptation_queries,
        query_embeddings,
        index,
        document_ids,
        top_k,
        config,
    )

    def retrievals_for_query(query_id: str) -> dict:
        return {
            "adapted_agent_graph": lambda: agent_graph_retrieve(
                index=index,
                query_embedding=query_embeddings[query_id],
                document_ids=document_ids,
                agents=agents,
                top_k=top_k,
                seed_k=config["seed_k"],
                graph_weight=config["graph_weight"],
            )
        }

    return {
        "seed": seed,
        "pruning_enabled": pruning_enabled,
        "updates": adaptation["updates"],
        "graph_before": adaptation["graph_before"],
        "graph_after": adaptation["graph_after"],
        "graph_deltas": adaptation["graph_deltas"],
        "systems": evaluate_retrievals(
            evaluation_queries,
            query_embeddings,
            qrels,
            retrievals_for_query,
        ),
    }


def save_results(path: Path, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)


def print_results(title: str, aggregate: dict) -> None:
    print(f"\n{title}")
    print("=" * 80)

    for system_name, metrics in aggregate.items():
        print(f"\n{system_name}")
        print(f"Recall@5:  {metrics['recall_at_5']['mean']:.4f} "
              f"(+/- {metrics['recall_at_5']['standard_deviation']:.4f})")
        print(f"Recall@10: {metrics['recall_at_10']['mean']:.4f} "
              f"(+/- {metrics['recall_at_10']['standard_deviation']:.4f})")
        print(f"MRR:       {metrics['mrr']['mean']:.4f} "
              f"(+/- {metrics['mrr']['standard_deviation']:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the final dissertation retrieval evaluation."
    )
    parser.add_argument("--adaptation-queries", type=int, default=100)
    parser.add_argument("--validation-queries", type=int, default=100)
    parser.add_argument("--evaluation-queries", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--pruning-ablation", action="store_true")
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args()

    print("Loading evaluation data...")
    queries = load_queries()
    qrels = load_qrels()
    query_splits = {
        seed: split_queries(
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

    embeddings, document_ids = load_embeddings()
    encoded_queries, query_ids = load_query_embeddings()
    all_query_embeddings = {
        query_id: encoded_queries[position:position + 1]
        for position, query_id in enumerate(query_ids)
    }
    missing_query_ids = required_query_ids - set(all_query_embeddings)
    if missing_query_ids:
        raise ValueError("Missing query embeddings. Run src/prepare_system.py first.")

    query_embeddings = {
        query_id: all_query_embeddings[query_id]
        for query_id in required_query_ids
    }
    static_graph = load_static_graph()
    index = build_faiss_index(embeddings)
    application_graph_checksum = file_checksum(AGENT_GRAPH_PATH)

    if args.pruning_ablation:
        conditions = {}

        for condition_name, pruning_enabled in {
            "pruning_enabled": True,
            "pruning_disabled": False,
        }.items():
            seed_results = []

            for seed, (adaptation, _, evaluation) in query_splits.items():
                print(f"Running {condition_name}, seed {seed}...")
                seed_results.append(
                    evaluate_ablation_seed(
                        seed,
                        adaptation,
                        evaluation,
                        query_embeddings,
                        qrels,
                        index,
                        document_ids,
                        args.top_k,
                        pruning_enabled,
                        PROJECT_ROOT / "results" / "pruning_ablation"
                        / condition_name / f"seed_{seed}",
                    )
                )

            conditions[condition_name] = {
                "seed_results": seed_results,
                "aggregate": aggregate_seed_results(seed_results)[
                    "adapted_agent_graph"
                ],
            }

        results = {
            "settings": {
                "seeds": args.seeds,
                "adaptation_queries": args.adaptation_queries,
                "validation_queries": args.validation_queries,
                "evaluation_queries": args.evaluation_queries,
                "top_k": args.top_k,
                "retrieval": OPTIMISED_CONFIG,
            },
            "conditions": conditions,
        }
        output_path = args.output_path or PRUNING_ABLATION_PATH
        print_results(
            "Pruning ablation results",
            {
                name: condition["aggregate"]
                for name, condition in conditions.items()
            },
        )
    else:
        seed_results = []

        for seed, (adaptation, validation, evaluation) in query_splits.items():
            print(f"Running final comparison, seed {seed}...")
            seed_results.append(
                evaluate_final_seed(
                    seed,
                    adaptation,
                    validation,
                    evaluation,
                    query_embeddings,
                    qrels,
                    index,
                    document_ids,
                    static_graph,
                    args.top_k,
                    PROJECT_ROOT / "results" / "main_evaluation" / f"seed_{seed}",
                )
            )

        aggregate = aggregate_seed_results(seed_results)
        results = {
            "settings": {
                "seeds": args.seeds,
                "adaptation_queries": args.adaptation_queries,
                "validation_queries": args.validation_queries,
                "evaluation_queries": args.evaluation_queries,
                "top_k": args.top_k,
                "static_graph": ORIGINAL_CONFIG,
                "original_agent_graph": ORIGINAL_CONFIG,
                "optimised_agent_graph": OPTIMISED_CONFIG,
                "selective_agent_graph": SELECTIVE_CONFIG,
            },
            "seed_results": seed_results,
            "aggregate": aggregate,
        }
        output_path = args.output_path or RESULTS_PATH
        print_results("Final evaluation results", aggregate)

    application_graph_unchanged = graph_file_is_unchanged(
        AGENT_GRAPH_PATH,
        application_graph_checksum,
    )
    if not application_graph_unchanged:
        raise RuntimeError("Evaluation modified the application agent graph")

    results["application_agent_graph"] = project_relative_path(AGENT_GRAPH_PATH)
    results["application_graph_unchanged"] = application_graph_unchanged
    save_results(output_path, results)
    print(f"\nApplication graph unchanged: {application_graph_unchanged}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
