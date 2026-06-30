import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

QUERIES_PATH = PROJECT_ROOT / "data" / "processed" / "queries.jsonl"
QRELS_PATH = PROJECT_ROOT / "data" / "processed" / "qrels.jsonl"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "document_embeddings.npy"
IDS_PATH = PROJECT_ROOT / "data" / "embeddings" / "document_ids.json"
STATIC_GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "static_graph_edges.jsonl"

RESULTS_CSV_PATH = PROJECT_ROOT / "results" / "baseline_results.csv"
SUMMARY_JSON_PATH = PROJECT_ROOT / "results" / "baseline_summary.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def read_jsonl(path: Path) -> list[dict]:
    rows = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def load_queries() -> list[dict]:
    return read_jsonl(QUERIES_PATH)


def load_qrels() -> dict[str, set[str]]:
    qrels = defaultdict(set)

    rows = read_jsonl(QRELS_PATH)

    for row in rows:
        query_id = row["query_id"]
        doc_id = row["doc_id"]
        qrels[query_id].add(doc_id)

    return qrels


def load_embeddings() -> tuple[np.ndarray, list[str]]:
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")

    with open(IDS_PATH, "r", encoding="utf-8") as file:
        document_ids = json.load(file)

    if len(document_ids) != embeddings.shape[0]:
        raise ValueError(
            f"Document ID count ({len(document_ids)}) does not match "
            f"embedding count ({embeddings.shape[0]})"
        )

    return embeddings, document_ids


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    embedding_dimensions = embeddings.shape[1]

    index = faiss.IndexFlatIP(embedding_dimensions)
    index.add(embeddings)

    return index


def load_static_graph() -> dict[str, dict[str, float]]:
    graph = defaultdict(dict)

    rows = read_jsonl(STATIC_GRAPH_PATH)

    for row in rows:
        source = row["source"]
        target = row["target"]
        weight = float(row["weight"])

        graph[source][target] = weight
        graph[target][source] = weight

    return graph


def recall_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    if not relevant_doc_ids:
        return 0.0

    retrieved_at_k = retrieved_doc_ids[:k]
    hits = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_doc_ids)

    return hits / len(relevant_doc_ids)


def mean_reciprocal_rank(retrieved_doc_ids: list[str], relevant_doc_ids: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank

    return 0.0


def faiss_retrieve(
    index: faiss.Index,
    query_embedding: np.ndarray,
    document_ids: list[str],
    top_k: int,
) -> list[str]:
    scores, indices = index.search(query_embedding, top_k)

    retrieved_doc_ids = []

    for index_position in indices[0]:
        doc_id = document_ids[int(index_position)]
        retrieved_doc_ids.append(doc_id)

    return retrieved_doc_ids


def static_graph_retrieve(
    index: faiss.Index,
    query_embedding: np.ndarray,
    document_ids: list[str],
    graph: dict[str, dict[str, float]],
    top_k: int,
    seed_k: int,
    graph_activation_weight: float,
) -> list[str]:
    seed_scores, seed_indices = index.search(query_embedding, seed_k)

    document_scores = defaultdict(float)

    for seed_score, seed_index in zip(seed_scores[0], seed_indices[0]):
        seed_doc_id = document_ids[int(seed_index)]
        seed_score = float(seed_score)

        # Seed documents keep their FAISS similarity score.
        document_scores[seed_doc_id] += seed_score

        # Connected graph neighbours receive partial activation.
        for neighbour_id, edge_weight in graph[seed_doc_id].items():
            neighbour_activation = seed_score * edge_weight * graph_activation_weight
            document_scores[neighbour_id] += neighbour_activation

    ranked_doc_ids = [
        doc_id
        for doc_id, score in sorted(
            document_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]

    return ranked_doc_ids[:top_k]


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
    parser = argparse.ArgumentParser(description="Evaluate FAISS and static graph retrieval baselines.")
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

        query_embedding = model.encode(
            [query_text],
            normalize_embeddings=True,
        ).astype("float32")

        embedding_latency_ms = (time.perf_counter() - embedding_start) * 1000

        faiss_start = time.perf_counter()

        faiss_doc_ids = faiss_retrieve(
            index=index,
            query_embedding=query_embedding,
            document_ids=document_ids,
            top_k=args.top_k,
        )

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

        static_graph_doc_ids = static_graph_retrieve(
            index=index,
            query_embedding=query_embedding,
            document_ids=document_ids,
            graph=graph,
            top_k=args.top_k,
            seed_k=args.seed_k,
            graph_activation_weight=args.graph_weight,
        )

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
