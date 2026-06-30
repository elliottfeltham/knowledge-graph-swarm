import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOCUMENTS_PATH = PROJECT_ROOT / "data" / "processed" / "documents.jsonl"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "document_embeddings.npy"
IDS_PATH = PROJECT_ROOT / "data" / "embeddings" / "document_ids.json"
STATIC_GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "static_graph_edges.jsonl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def read_jsonl(path: Path) -> list[dict]:
    rows = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def load_documents() -> dict[str, dict]:
    documents = read_jsonl(DOCUMENTS_PATH)
    return {doc["doc_id"]: doc for doc in documents}


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

    edges = read_jsonl(STATIC_GRAPH_PATH)

    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        weight = float(edge["weight"])

        graph[source][target] = weight
        graph[target][source] = weight

    return graph


def search(
    query: str,
    top_k: int = 10,
    seed_k: int = 5,
    graph_activation_weight: float = 0.3,
) -> tuple[list[dict], float]:
    documents = load_documents()
    embeddings, document_ids = load_embeddings()
    graph = load_static_graph()

    index = build_faiss_index(embeddings)
    model = SentenceTransformer(MODEL_NAME)

    start_time = time.perf_counter()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    ).astype("float32")

    # Use the FAISS index to find the first relevant documents as a seed.
    seed_scores, seed_indices = index.search(query_embedding, seed_k)

    document_scores = defaultdict(float)

    for seed_score, seed_index in zip(seed_scores[0], seed_indices[0]):
        seed_doc_id = document_ids[int(seed_index)]
        seed_score = float(seed_score)

        document_scores[seed_doc_id] += seed_score

        for neighbour_id, edge_weight in graph[seed_doc_id].items():
            neighbour_activation = seed_score * edge_weight * graph_activation_weight
            document_scores[neighbour_id] += neighbour_activation

    latency_ms = (time.perf_counter() - start_time) * 1000

    ranked_results = sorted(
        document_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:top_k]

    results = []

    for doc_id, score in ranked_results:
        document = documents[doc_id]

        results.append(
            {
                "doc_id": doc_id,
                "title": document["title"],
                "score": float(score),
                "text_preview": document["text"][:300],
            }
        )

    return results, latency_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Run static graph retrieval baseline.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--seed-k",
        type=int,
        default=5,
        help="Number of FAISS seed documents to activate.",
    )
    parser.add_argument(
        "--graph-weight",
        type=float,
        default=0.3,
        help="How strongly graph neighbours affect final ranking.",
    )

    args = parser.parse_args()

    results, latency_ms = search(
        query=args.query,
        top_k=args.top_k,
        seed_k=args.seed_k,
        graph_activation_weight=args.graph_weight,
    )

    print(f"\nQuery: {args.query}")
    print(f"Static graph search latency: {latency_ms:.2f} ms")
    print(f"Top {args.top_k} results:")

    for rank, result in enumerate(results, start=1):
        print("\n" + "-" * 80)
        print(f"Rank {rank}")
        print(f"Title: {result['title']}")
        print(f"Doc ID: {result['doc_id']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Preview: {result['text_preview']}")


if __name__ == "__main__":
    main()
