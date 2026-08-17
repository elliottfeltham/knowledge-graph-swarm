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
from evaluation.create_results_report import save_graph_update_report
from evaluation.experiment_config import DEFAULT_SEEDS, OPTIMISED_CONFIG
from evaluation.experiment_helpers import (
    EvaluationContext,
    evaluate_configuration,
    split_query_groups,
)
from evaluation.graph_state import file_checksum, graph_file_is_unchanged
from graph.config import INITIAL_WEIGHT_SCALE, NEIGHBOURS_PER_DOCUMENT
from graph.initialize_agent_graph import build_initial_agent_graph
from retrieval.agent_graph_retrieval import AGENT_GRAPH_PATH


DEFAULT_STRENGTHEN_AMOUNTS = [0.01, 0.025, 0.05, 0.075, 0.1]
DEFAULT_DECAY_RATES = [0.98, 0.99, 0.995, 0.999, 1.0]
RESULTS_PATH = PROJECT_ROOT / "results" / "update_dynamics_tuning.json"


def build_parameter_grid(
    strengthen_amounts: list[float],
    decay_rates: list[float],
) -> list[tuple[float, float]]:
    return [
        (strengthen_amount, decay_rate)
        for strengthen_amount in strengthen_amounts
        for decay_rate in decay_rates
    ]


def select_best_configuration(configurations: list[dict]) -> dict:
    return max(
        configurations,
        key=lambda configuration: (
            configuration["adapted"]["recall_at_10"],
            configuration["adapted"]["mrr"],
            configuration["adapted"]["recall_at_5"],
            -configuration["strengthen_amount"],
            -abs(1.0 - configuration["decay_rate"]),
        ),
    )


def print_summary(best: dict) -> None:
    print("\nSelected graph update settings")
    print("=" * 60)
    print(f"Strengthening amount: {best['strengthen_amount']}")
    print(f"Decay rate:           {best['decay_rate']}")
    print(f"Validation Recall@10: {best['adapted']['recall_at_10']:.4f}")
    print(f"Validation MRR:       {best['adapted']['mrr']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate graph strengthening amounts and decay rates."
    )
    parser.add_argument("--adaptation-queries", type=int, default=100)
    parser.add_argument("--validation-queries", type=int, default=100)
    parser.add_argument("--evaluation-queries", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--strengthen-amounts",
        type=float,
        nargs="+",
        default=DEFAULT_STRENGTHEN_AMOUNTS,
    )
    parser.add_argument(
        "--decay-rates",
        type=float,
        nargs="+",
        default=DEFAULT_DECAY_RATES,
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
    configurations = []
    validation_faiss = None

    for strengthen_amount, decay_rate in build_parameter_grid(
        args.strengthen_amounts,
        args.decay_rates,
    ):
        print(
            f"Testing strengthen_amount={strengthen_amount}, "
            f"decay_rate={decay_rate}..."
        )
        evaluation = evaluate_configuration(
            context=context,
            query_splits=query_splits,
            query_group=1,
            seed_k=OPTIMISED_CONFIG["seed_k"],
            graph_weight=OPTIMISED_CONFIG["graph_weight"],
            update_config=GraphUpdateConfig(
                strengthen_amount=strengthen_amount,
                decay_rate=decay_rate,
                enable_pruning=False,
                update_k=OPTIMISED_CONFIG["update_k"],
            ),
        )
        validation_faiss = evaluation["faiss"]
        configurations.append(
            {
                "strengthen_amount": strengthen_amount,
                "decay_rate": decay_rate,
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
        raise RuntimeError("Graph update tuning modified the application graph")

    results = {
        "settings": {
            "seeds": args.seeds,
            "adaptation_queries": args.adaptation_queries,
            "validation_queries": args.validation_queries,
            "reserved_evaluation_queries": args.evaluation_queries,
            "top_k": args.top_k,
            "seed_k": OPTIMISED_CONFIG["seed_k"],
            "graph_weight": OPTIMISED_CONFIG["graph_weight"],
            "update_k": OPTIMISED_CONFIG["update_k"],
            "neighbours_per_document": NEIGHBOURS_PER_DOCUMENT,
            "initial_weight_scale": INITIAL_WEIGHT_SCALE,
            "strengthen_amounts": args.strengthen_amounts,
            "decay_rates": args.decay_rates,
            "pruning_enabled": False,
        },
        "selection_rule": (
            "Highest validation Recall@10, then MRR, Recall@5, lower "
            "strengthening amount, and decay rate closest to 1.0."
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

    save_graph_update_report(results)
    print_summary(best)
    print(f"Application graph unchanged: {application_graph_unchanged}")
    print(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()
