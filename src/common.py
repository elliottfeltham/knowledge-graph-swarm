import json
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCUMENTS_PATH = PROJECT_ROOT / "data" / "processed" / "documents.jsonl"
QUERIES_PATH = PROJECT_ROOT / "data" / "processed" / "queries.jsonl"
QRELS_PATH = PROJECT_ROOT / "data" / "processed" / "qrels.jsonl"
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
    return {document["doc_id"]: document for document in documents}


def load_queries() -> list[dict]:
    return read_jsonl(QUERIES_PATH)


def load_qrels() -> dict[str, set[str]]:
    qrels = defaultdict(set)

    for row in read_jsonl(QRELS_PATH):
        qrels[row["query_id"]].add(row["doc_id"])

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

    for row in read_jsonl(STATIC_GRAPH_PATH):
        source = row["source"]
        target = row["target"]
        weight = float(row["weight"])

        graph[source][target] = weight
        graph[target][source] = weight

    return graph


def encode_query(model, query: str) -> np.ndarray:
    return model.encode([query], normalize_embeddings=True).astype("float32")


def faiss_retrieve(
    index: faiss.Index,
    query_embedding: np.ndarray,
    document_ids: list[str],
    top_k: int,
) -> list[tuple[str, float]]:
    scores, indices = index.search(query_embedding, top_k)

    retrieved = []

    for score, index_position in zip(scores[0], indices[0]):
        retrieved.append((document_ids[int(index_position)], float(score)))

    return retrieved


def static_graph_retrieve(
    index: faiss.Index,
    query_embedding: np.ndarray,
    document_ids: list[str],
    graph: dict[str, dict[str, float]],
    top_k: int,
    seed_k: int,
    graph_activation_weight: float,
) -> list[tuple[str, float]]:
    seed_scores, seed_indices = index.search(query_embedding, seed_k)

    document_scores = defaultdict(float)

    for seed_score, seed_index in zip(seed_scores[0], seed_indices[0]):
        seed_doc_id = document_ids[int(seed_index)]
        seed_score = float(seed_score)

        document_scores[seed_doc_id] += seed_score

        for neighbour_id, edge_weight in graph[seed_doc_id].items():
            neighbour_activation = seed_score * edge_weight * graph_activation_weight
            document_scores[neighbour_id] += neighbour_activation

    ranked_doc_ids = sorted(
        document_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return [(doc_id, float(score)) for doc_id, score in ranked_doc_ids[:top_k]]


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


def average(values: list[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)
