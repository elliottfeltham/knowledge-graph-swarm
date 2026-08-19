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
from evaluation.create_results_report import save_repeated_adaptation_report
from evaluation.experiment_config import DEFAULT_SEEDS
from evaluation.experiment_helpers import (
    EvaluationContext,
    adapt_graph,
    calculate_faiss_metrics,
    calculate_metrics,
    compare_query_metrics,
    split_query_groups,
    summarise_agent_rows,
)
from evaluation.graph_state import file_checksum, graph_file_is_unchanged
from graph.config import INITIAL_WEIGHT_SCALE, NEIGHBOURS_PER_DOCUMENT
from graph.initialize_agent_graph import build_initial_agent_graph
from retrieval.agent_graph_retrieval import AGENT_GRAPH_PATH, agent_graph_retrieve


RESULTS_PATH = PROJECT_ROOT / "results" / "repeated_adaptation.json"
CONDITIONS = {
    "current_strength": {
        "seed_k": 3,
        "graph_weight": 0.05,
        "update_k": 3,
        "strengthen_amount": 0.01,
        "decay_rate": 0.995,
    },
    "stronger_adaptation": {
        "seed_k": 3,
        "graph_weight": 0.1,
        "update_k": 3,
        "strengthen_amount": 0.05,
        "decay_rate": 0.995,
    },
}


def evaluate_graph(
    queries: list[dict],
    context: EvaluationContext,
    agents,
    config: dict,
    faiss_metrics: dict[str, dict],
) -> tuple[dict, dict[str, list[str]]]:
    rows = []
    rankings = {}

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
                seed_k=config["seed_k"],
                graph_weight=config["graph_weight"],
            )
        ]
        metrics = calculate_metrics(
            retrieved_doc_ids,
            context.qrels.get(query_id, set()),
        )
        rankings[query_id] = retrieved_doc_ids
        rows.append(
            {
                "metrics": metrics,
                "outcome": compare_query_metrics(metrics, faiss_metrics[query_id]),
            }
        )

    return summarise_agent_rows(rows), rankings


def aggregate_condition_results(seed_results: list[dict]) -> dict[str, list[dict]]:
    aggregate = {}

    for condition_name in seed_results[0]["conditions"]:
        passes = []

        for pass_index, first_pass in enumerate(
            seed_results[0]["conditions"][condition_name]
        ):
            matching = [
                result["conditions"][condition_name][pass_index]
                for result in seed_results
            ]
            row = {
                "pass": first_pass["pass"],
                "interactions": first_pass["interactions"],
                "rankings_changed": {
                    "mean": mean(item["rankings_changed"] for item in matching),
                    "standard_deviation": (
                        stdev(item["rankings_changed"] for item in matching)
                        if len(matching) > 1
                        else 0.0
                    ),
                },
            }

            for section in ("metrics", "graph", "updates"):
                row[section] = {}

                for field in matching[0][section]:
                    values = [item[section][field] for item in matching]
                    row[section][field] = {
                        "mean": mean(values),
                        "standard_deviation": (
                            stdev(values) if len(values) > 1 else 0.0
                        ),
                    }

            passes.append(row)

        aggregate[condition_name] = passes

    return aggregate


def evaluate_seed(
    seed: int,
    context: EvaluationContext,
    adaptation_queries: list[dict],
    evaluation_queries: list[dict],
    passes: int,
) -> dict:
    faiss_metrics = calculate_faiss_metrics(evaluation_queries, context)
    condition_results = {}

    for condition_name, condition in CONDITIONS.items():
        agents = copy.deepcopy(context.initial_agents)
        update_config = GraphUpdateConfig(
            strengthen_amount=condition["strengthen_amount"],
            decay_rate=condition["decay_rate"],
            enable_pruning=False,
            update_k=condition["update_k"],
        )
        update_totals = {
            "strengthened_edges": 0,
            "created_edges": 0,
            "pruned_edges": 0,
        }
        initial_metrics, initial_rankings = evaluate_graph(
            evaluation_queries,
            context,
            agents,
            condition,
            faiss_metrics,
        )
        pass_results = [
            {
                "pass": 0,
                "interactions": 0,
                "rankings_changed": 0,
                "metrics": initial_metrics,
                "graph": calculate_graph_statistics(agents),
                "updates": update_totals.copy(),
            }
        ]

        for pass_number in range(1, passes + 1):
            updates = adapt_graph(
                adaptation_queries,
                context,
                agents,
                condition["seed_k"],
                condition["graph_weight"],
                update_config=update_config,
            )

            for field in update_totals:
                update_totals[field] += updates[field]

            metrics, rankings = evaluate_graph(
                evaluation_queries,
                context,
                agents,
                condition,
                faiss_metrics,
            )
            pass_results.append(
                {
                    "pass": pass_number,
                    "interactions": pass_number * len(adaptation_queries),
                    "rankings_changed": sum(
                        ranking != initial_rankings[query_id]
                        for query_id, ranking in rankings.items()
                    ),
                    "metrics": metrics,
                    "graph": calculate_graph_statistics(agents),
                    "updates": update_totals.copy(),
                }
            )

        condition_results[condition_name] = pass_results

    return {"seed": seed, "conditions": condition_results}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate repeated exposure to the same adaptation queries."
    )
    parser.add_argument("--adaptation-queries", type=int, default=100)
    parser.add_argument("--evaluation-queries", type=int, default=100)
    parser.add_argument("--reserved-evaluation-queries", type=int, default=300)
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--output-path", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    if args.passes < 1:
        raise ValueError("Passes must be at least 1")

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
            args.evaluation_queries,
            args.reserved_evaluation_queries,
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

    context = EvaluationContext(
        query_embeddings=query_embeddings,
        qrels=qrels,
        index=build_faiss_index(embeddings),
        document_ids=document_ids,
        initial_agents=build_initial_agent_graph(
            neighbours_per_doc=NEIGHBOURS_PER_DOCUMENT,
            initial_weight_scale=INITIAL_WEIGHT_SCALE,
        ),
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
                groups[1],
                args.passes,
            )
        )

    application_graph_unchanged = graph_file_is_unchanged(
        AGENT_GRAPH_PATH,
        application_graph_checksum,
    )
    if not application_graph_unchanged:
        raise RuntimeError("Repeated adaptation modified the application graph")

    results = {
        "settings": {
            "seeds": args.seeds,
            "adaptation_queries": args.adaptation_queries,
            "evaluation_queries": args.evaluation_queries,
            "reserved_evaluation_queries": args.reserved_evaluation_queries,
            "passes": args.passes,
            "top_k": args.top_k,
            "conditions": CONDITIONS,
        },
        "application_graph_unchanged": application_graph_unchanged,
        "seed_results": seed_results,
        "aggregate": aggregate_condition_results(seed_results),
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    save_repeated_adaptation_report(results)
    print("\nRepeated adaptation")
    print("=" * 64)

    for condition_name, pass_results in results["aggregate"].items():
        initial = pass_results[0]["metrics"]
        final = pass_results[-1]["metrics"]
        changed = pass_results[-1]["rankings_changed"]["mean"]
        print(f"{condition_name}:")
        print(
            f"  Recall@10 {initial['recall_at_10']['mean']:.4f} -> "
            f"{final['recall_at_10']['mean']:.4f}"
        )
        print(
            f"  MRR       {initial['mrr']['mean']:.4f} -> "
            f"{final['mrr']['mean']:.4f}"
        )
        print(f"  Mean rankings changed: {changed:.1f}")

    print(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()
