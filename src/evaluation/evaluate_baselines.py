import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

from sentence_transformers import SentenceTransformer

from common import (
    MODEL_NAME,
    PROJECT_ROOT,
    average,
    build_faiss_index,
    encode_query,
    faiss_retrieve,
    load_embeddings,
    load_qrels,
    load_queries,
    load_static_graph,
    mean_reciprocal_rank,
    recall_at_k,
    static_graph_retrieve,
)
from retrieval.agent_graph_retrieval import agent_graph_retrieve, load_agent_graph


RESULTS_CSV_PATH = PROJECT_ROOT / "results" / "baseline_results.csv"
SUMMARY_JSON_PATH = PROJECT_ROOT / "results" / "baseline_summary.json"


def evaluate_system_result(
    system_name: str,
    query_id: str,
    query_text: str,
    retrieved_doc_ids: list[str],
    relevant_doc_ids: set[str],
    embedding_latency_ms: float,
    retrieval_latency_ms: float,
) -> dict:
    recall_5 = recall_at_k(retrieved_doc_ids, relevant_doc_ids, 5)
    recall_10 = recall_at_k(retrieved_doc_ids, relevant_doc_ids, 10)
    mrr = mean_reciprocal_rank(retrieved_doc_ids, relevant_doc_ids)

    total_latency_ms = embedding_latency_ms + retrieval_latency_ms

    return {
        "system": system_name,
        "query_id": query_id,
        "query": query_text,
        "recall_at_5": recall_5,
        "recall_at_10": recall_10,
        "mrr": mrr,
        "embedding_latency_ms": embedding_latency_ms,
        "retrieval_latency_ms": retrieval_latency_ms,
        "total_latency_ms": total_latency_ms,
        "retrieved_doc_ids": json.dumps(retrieved_doc_ids),
        "relevant_doc_ids": json.dumps(sorted(relevant_doc_ids)),
    }


def average(values: list[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def summarise_results(rows: list[dict]) -> dict:
    summary = {}

    systems = sorted(set(row["system"] for row in rows))

    for system in systems:
        system_rows = [row for row in rows if row["system"] == system]

        summary[system] = {
            "queries_evaluated": len(system_rows),
            "mean_recall_at_5": average([row["recall_at_5"] for row in system_rows]),
            "mean_recall_at_10": average([row["recall_at_10"] for row in system_rows]),
            "mean_mrr": average([row["mrr"] for row in system_rows]),
            "mean_embedding_latency_ms": average([row["embedding_latency_ms"] for row in system_rows]),
            "mean_retrieval_latency_ms": average([row["retrieval_latency_ms"] for row in system_rows]),
            "mean_total_latency_ms": average([row["total_latency_ms"] for row in system_rows]),
        }

    return summary


def save_results_csv(rows: list[dict]) -> None:
    RESULTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "system",
        "query_id",
        "query",
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "embedding_latency_ms",
        "retrieval_latency_ms",
        "total_latency_ms",
        "retrieved_doc_ids",
        "relevant_doc_ids",
    ]

    with open(RESULTS_CSV_PATH, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_json(summary: dict) -> None:
    SUMMARY_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def print_summary(summary: dict) -> None:
    print("\nEvaluation summary")
    print("=" * 80)

    for system, metrics in summary.items():
        print(f"\n{system}")
        print("-" * 80)
        print(f"Queries evaluated: {metrics['queries_evaluated']}")
        print(f"Mean Recall@5: {metrics['mean_recall_at_5']:.4f}")
        print(f"Mean Recall@10: {metrics['mean_recall_at_10']:.4f}")
        print(f"Mean MRR: {metrics['mean_mrr']:.4f}")
        print(f"Mean embedding latency: {metrics['mean_embedding_latency_ms']:.2f} ms")
        print(f"Mean retrieval latency: {metrics['mean_retrieval_latency_ms']:.2f} ms")
        print(f"Mean total latency: {metrics['mean_total_latency_ms']:.2f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FAISS, static graph, and agent graph retrieval.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed-k", type=int, default=5)
    parser.add_argument("--graph-weight", type=float, default=0.3)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of queries to evaluate for quick testing.",
    )

    args = parser.parse_args()

    print("Loading queries and qrels...")
    queries = load_queries()
    qrels = load_qrels()

    if args.limit is not None:
        queries = queries[: args.limit]

    print("Loading embeddings...")
    embeddings, document_ids = load_embeddings()

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    print("Loading static graph...")
    graph = load_static_graph()

    print("Loading agent graph...")
    agents = load_agent_graph()
    print(f"Loaded agent graph agents: {len(agents)}")

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Evaluating {len(queries)} queries...")
    print(f"top_k={args.top_k}, seed_k={args.seed_k}, graph_weight={args.graph_weight}")

    rows = []

    for query_number, query_row in enumerate(queries, start=1):
        query_id = query_row["query_id"]
        query_text = query_row["query"]
        relevant_doc_ids = qrels.get(query_id, set())

        embedding_start = time.perf_counter()

        query_embedding = encode_query(model, query_text)

        embedding_latency_ms = (time.perf_counter() - embedding_start) * 1000

        faiss_start = time.perf_counter()

        faiss_doc_ids = [
            doc_id
            for doc_id, _ in faiss_retrieve(
                index=index,
                query_embedding=query_embedding,
                document_ids=document_ids,
                top_k=args.top_k,
            )
        ]

        faiss_retrieval_latency_ms = (time.perf_counter() - faiss_start) * 1000

        rows.append(
            evaluate_system_result(
                system_name="faiss",
                query_id=query_id,
                query_text=query_text,
                retrieved_doc_ids=faiss_doc_ids,
                relevant_doc_ids=relevant_doc_ids,
                embedding_latency_ms=embedding_latency_ms,
                retrieval_latency_ms=faiss_retrieval_latency_ms,
            )
        )

        static_graph_start = time.perf_counter()

        static_graph_doc_ids = [
            doc_id
            for doc_id, _ in static_graph_retrieve(
                index=index,
                query_embedding=query_embedding,
                document_ids=document_ids,
                graph=graph,
                top_k=args.top_k,
                seed_k=args.seed_k,
                graph_activation_weight=args.graph_weight,
            )
        ]

        static_graph_retrieval_latency_ms = (time.perf_counter() - static_graph_start) * 1000

        rows.append(
            evaluate_system_result(
                system_name="static_graph",
                query_id=query_id,
                query_text=query_text,
                retrieved_doc_ids=static_graph_doc_ids,
                relevant_doc_ids=relevant_doc_ids,
                embedding_latency_ms=embedding_latency_ms,
                retrieval_latency_ms=static_graph_retrieval_latency_ms,
            )
        )

        agent_graph_start = time.perf_counter()

        agent_graph_doc_ids = [
            doc_id
            for doc_id, _ in agent_graph_retrieve(
                index=index,
                query_embedding=query_embedding,
                document_ids=document_ids,
                agents=agents,
                top_k=args.top_k,
                seed_k=args.seed_k,
                graph_weight=args.graph_weight,
            )
        ]

        agent_graph_retrieval_latency_ms = (time.perf_counter() - agent_graph_start) * 1000

        rows.append(
            evaluate_system_result(
                system_name="agent_graph",
                query_id=query_id,
                query_text=query_text,
                retrieved_doc_ids=agent_graph_doc_ids,
                relevant_doc_ids=relevant_doc_ids,
                embedding_latency_ms=embedding_latency_ms,
                retrieval_latency_ms=agent_graph_retrieval_latency_ms,
            )
        )

        if query_number % 50 == 0 or query_number == len(queries):
            print(f"Evaluated {query_number}/{len(queries)} queries")

    summary = summarise_results(rows)

    save_results_csv(rows)
    save_summary_json(summary)

    print_summary(summary)

    print(f"\nSaved detailed results to: {RESULTS_CSV_PATH}")
    print(f"Saved summary results to: {SUMMARY_JSON_PATH}")


if __name__ == "__main__":
    main()
