import argparse
import json
import shutil
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer

from agents import DocumentAgent, GraphUpdateConfig, load_agents, save_agents
from agents.graph_statistics import (
    build_edge_snapshot,
    calculate_graph_deltas,
    calculate_graph_statistics,
    compare_edge_snapshots,
)
from agents.graph_update import update_graph_after_retrieval
from common import (
    MODEL_NAME,
    PROJECT_ROOT,
    average,
    build_faiss_index,
    encode_query,
    load_embeddings,
    load_qrels,
    load_queries,
    mean_reciprocal_rank,
    recall_at_k,
)
from retrieval.agent_graph_retrieval import AGENT_GRAPH_PATH, agent_graph_retrieve


SIMULATED_AGENT_GRAPH_PATH = PROJECT_ROOT / "results" / "simulated_document_agents.jsonl"
SIMULATION_SUMMARY_PATH = PROJECT_ROOT / "results" / "adaptive_simulation_summary.json"


def evaluate_agent_graph(
    queries: list[dict],
    qrels: dict[str, set[str]],
    index,
    model: SentenceTransformer,
    document_ids: list[str],
    agents: dict[str, DocumentAgent],
    top_k: int,
    seed_k: int,
    graph_weight: float,
) -> dict:
    rows = []

    for query_row in queries:
        query_id = query_row["query_id"]
        query_text = query_row["query"]
        relevant_doc_ids = qrels.get(query_id, set())

        query_embedding = encode_query(model, query_text)
        retrieved_doc_ids = [
            doc_id
            for doc_id, _ in agent_graph_retrieve(
                index=index,
                query_embedding=query_embedding,
                document_ids=document_ids,
                agents=agents,
                top_k=top_k,
                seed_k=seed_k,
                graph_weight=graph_weight,
            )
        ]

        rows.append(
            {
                "recall_at_5": recall_at_k(retrieved_doc_ids, relevant_doc_ids, 5),
                "recall_at_10": recall_at_k(retrieved_doc_ids, relevant_doc_ids, 10),
                "mrr": mean_reciprocal_rank(retrieved_doc_ids, relevant_doc_ids),
            }
        )

    return summarise_metrics(rows)


def summarise_metrics(rows: list[dict]) -> dict:
    return {
        "queries_evaluated": len(rows),
        "mean_recall_at_5": average([row["recall_at_5"] for row in rows]),
        "mean_recall_at_10": average([row["recall_at_10"] for row in rows]),
        "mean_mrr": average([row["mrr"] for row in rows]),
    }


def copy_agent_graph(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target_path)


def split_queries(
    queries: list[dict],
    adaptation_query_count: int,
    evaluation_query_count: int,
) -> tuple[list[dict], list[dict]]:
    required_queries = adaptation_query_count + evaluation_query_count

    if len(queries) < required_queries:
        raise ValueError(
            f"Need {required_queries} queries but only found {len(queries)}. "
            "Use smaller --adaptation-queries or --evaluation-queries values."
        )

    adaptation_queries = queries[:adaptation_query_count]
    evaluation_queries = queries[adaptation_query_count:required_queries]

    return adaptation_queries, evaluation_queries


def simulate_user_queries(
    queries: list[dict],
    index,
    model: SentenceTransformer,
    document_ids: list[str],
    agents: dict[str, DocumentAgent],
    top_k: int,
    seed_k: int,
    graph_weight: float,
    update_config: GraphUpdateConfig,
) -> dict:
    total_strengthened_edges = 0
    total_pruned_edges = 0

    for query_number, query_row in enumerate(queries, start=1):
        query_embedding = encode_query(model, query_row["query"])
        retrieved_doc_ids = [
            doc_id
            for doc_id, _ in agent_graph_retrieve(
                index=index,
                query_embedding=query_embedding,
                document_ids=document_ids,
                agents=agents,
                top_k=top_k,
                seed_k=seed_k,
                graph_weight=graph_weight,
            )
        ]

        update_summary = update_graph_after_retrieval(
            agents=agents,
            retrieved_doc_ids=retrieved_doc_ids,
            config=update_config,
        )
        total_strengthened_edges += update_summary["strengthened_edges"]
        total_pruned_edges += update_summary["pruned_edges"]

        if query_number % 50 == 0 or query_number == len(queries):
            print(f"Simulated {query_number}/{len(queries)} user queries")

    return {
        "adaptation_queries": len(queries),
        "total_strengthened_edges": total_strengthened_edges,
        "total_pruned_edges": total_pruned_edges,
    }


def calculate_metric_deltas(before_metrics: dict, after_metrics: dict) -> dict:
    return {
        "mean_recall_at_5": after_metrics["mean_recall_at_5"] - before_metrics["mean_recall_at_5"],
        "mean_recall_at_10": after_metrics["mean_recall_at_10"] - before_metrics["mean_recall_at_10"],
        "mean_mrr": after_metrics["mean_mrr"] - before_metrics["mean_mrr"],
    }


def save_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def print_metric_comparison(before_metrics: dict, after_metrics: dict, deltas: dict) -> None:
    print("\nAdaptive simulation summary")
    print("=" * 80)
    print(f"Evaluation queries: {before_metrics['queries_evaluated']}")
    print(f"Before Recall@5: {before_metrics['mean_recall_at_5']:.4f}")
    print(f"After Recall@5:  {after_metrics['mean_recall_at_5']:.4f}")
    print(f"Delta Recall@5:  {deltas['mean_recall_at_5']:+.4f}")
    print(f"Before Recall@10: {before_metrics['mean_recall_at_10']:.4f}")
    print(f"After Recall@10:  {after_metrics['mean_recall_at_10']:.4f}")
    print(f"Delta Recall@10:  {deltas['mean_recall_at_10']:+.4f}")
    print(f"Before MRR: {before_metrics['mean_mrr']:.4f}")
    print(f"After MRR:  {after_metrics['mean_mrr']:.4f}")
    print(f"Delta MRR:  {deltas['mean_mrr']:+.4f}")


def print_graph_comparison(graph_before: dict, graph_after: dict, graph_deltas: dict, edge_changes: dict) -> None:
    print("\nGraph adaptation")
    print("=" * 80)
    print(f"Edges before: {graph_before['undirected_edges']}")
    print(f"Edges after:  {graph_after['undirected_edges']}")
    print(f"Edge delta:   {graph_deltas['undirected_edges']:+}")
    print(f"Average degree before: {graph_before['average_degree']:.4f}")
    print(f"Average degree after:  {graph_after['average_degree']:.4f}")
    print(f"Average degree delta:  {graph_deltas['average_degree']:+.4f}")
    print(f"Average edge weight before: {graph_before['average_edge_weight']:.4f}")
    print(f"Average edge weight after:  {graph_after['average_edge_weight']:.4f}")
    print(f"Average edge weight delta:  {graph_deltas['average_edge_weight']:+.4f}")
    print(f"\nNew edges: {edge_changes['new_edges']}")
    print(f"Removed edges: {edge_changes['removed_edges']}")
    print(f"Strengthened edges: {edge_changes['strengthened_edges']}")
    print(f"Weakened edges: {edge_changes['weakened_edges']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate user queries and measure agent graph adaptation.")
    parser.add_argument("--adaptation-queries", type=int, default=100)
    parser.add_argument("--evaluation-queries", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed-k", type=int, default=5)
    parser.add_argument("--graph-weight", type=float, default=0.3)
    parser.add_argument(
        "--changed-edges",
        type=int,
        default=10,
        help="Number of largest edge changes to save in the simulation summary.",
    )
    parser.add_argument(
        "--disable-pruning",
        action="store_true",
        help="Skip pruning during adaptive graph updates.",
    )

    args = parser.parse_args()

    print("Loading queries and qrels...")
    queries = load_queries()
    qrels = load_qrels()
    adaptation_queries, evaluation_queries = split_queries(
        queries=queries,
        adaptation_query_count=args.adaptation_queries,
        evaluation_query_count=args.evaluation_queries,
    )

    print("Loading embeddings...")
    embeddings, document_ids = load_embeddings()

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Loading original agent graph...")
    original_agents = load_agents(AGENT_GRAPH_PATH)

    print("Copying graph for simulation...")
    copy_agent_graph(AGENT_GRAPH_PATH, SIMULATED_AGENT_GRAPH_PATH)
    simulated_agents = load_agents(SIMULATED_AGENT_GRAPH_PATH)
    edge_snapshot_before = build_edge_snapshot(simulated_agents)
    graph_before = calculate_graph_statistics(simulated_agents)

    print("Evaluating original graph before simulated use...")
    start_time = time.perf_counter()
    before_metrics = evaluate_agent_graph(
        queries=evaluation_queries,
        qrels=qrels,
        index=index,
        model=model,
        document_ids=document_ids,
        agents=original_agents,
        top_k=args.top_k,
        seed_k=args.seed_k,
        graph_weight=args.graph_weight,
    )
    before_latency_ms = (time.perf_counter() - start_time) * 1000

    update_config = GraphUpdateConfig(enable_pruning=not args.disable_pruning)

    print("Simulating user queries...")
    update_totals = simulate_user_queries(
        queries=adaptation_queries,
        index=index,
        model=model,
        document_ids=document_ids,
        agents=simulated_agents,
        top_k=args.top_k,
        seed_k=args.seed_k,
        graph_weight=args.graph_weight,
        update_config=update_config,
    )
    save_agents(SIMULATED_AGENT_GRAPH_PATH, simulated_agents)

    print("Evaluating copied graph after simulated use...")
    edge_snapshot_after = build_edge_snapshot(simulated_agents)
    graph_after = calculate_graph_statistics(simulated_agents)
    start_time = time.perf_counter()
    after_metrics = evaluate_agent_graph(
        queries=evaluation_queries,
        qrels=qrels,
        index=index,
        model=model,
        document_ids=document_ids,
        agents=simulated_agents,
        top_k=args.top_k,
        seed_k=args.seed_k,
        graph_weight=args.graph_weight,
    )
    after_latency_ms = (time.perf_counter() - start_time) * 1000

    deltas = calculate_metric_deltas(
        before_metrics=before_metrics,
        after_metrics=after_metrics,
    )
    graph_deltas = calculate_graph_deltas(
        before_stats=graph_before,
        after_stats=graph_after,
    )
    edge_changes = compare_edge_snapshots(
        before_snapshot=edge_snapshot_before,
        after_snapshot=edge_snapshot_after,
        changed_edge_limit=args.changed_edges,
    )
    summary = {
        "settings": {
            "adaptation_queries": args.adaptation_queries,
            "evaluation_queries": args.evaluation_queries,
            "top_k": args.top_k,
            "seed_k": args.seed_k,
            "graph_weight": args.graph_weight,
            "changed_edges": args.changed_edges,
            "pruning_enabled": not args.disable_pruning,
        },
        "paths": {
            "source_agent_graph": str(AGENT_GRAPH_PATH),
            "simulated_agent_graph": str(SIMULATED_AGENT_GRAPH_PATH),
        },
        "updates": update_totals,
        "before": before_metrics,
        "after": after_metrics,
        "deltas": deltas,
        "graph_before": graph_before,
        "graph_after": graph_after,
        "graph_deltas": graph_deltas,
        "edge_changes": {
            "new_edges": edge_changes["new_edges"],
            "removed_edges": edge_changes["removed_edges"],
            "strengthened_edges": edge_changes["strengthened_edges"],
            "weakened_edges": edge_changes["weakened_edges"],
        },
        "top_changed_edges": edge_changes["top_changed_edges"],
        "latency_ms": {
            "before_evaluation": before_latency_ms,
            "after_evaluation": after_latency_ms,
        },
    }
    save_summary(SIMULATION_SUMMARY_PATH, summary)

    print_metric_comparison(
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        deltas=deltas,
    )
    print_graph_comparison(
        graph_before=graph_before,
        graph_after=graph_after,
        graph_deltas=graph_deltas,
        edge_changes=edge_changes,
    )
    print(f"\nAdaptation queries: {update_totals['adaptation_queries']}")
    print(f"Total strengthened edges: {update_totals['total_strengthened_edges']}")
    print(f"Total pruned edges: {update_totals['total_pruned_edges']}")
    print(f"Copied graph saved to: {SIMULATED_AGENT_GRAPH_PATH}")
    print(f"Summary saved to: {SIMULATION_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
