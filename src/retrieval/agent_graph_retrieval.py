from collections import defaultdict
from pathlib import Path
import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from agents import DocumentAgent, load_agents
from common import (
    MODEL_NAME,
    PROJECT_ROOT,
    build_faiss_index,
    encode_query,
    load_documents,
    load_embeddings,
)


AGENT_GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "document_agents.jsonl"


def load_agent_graph(path: Path = AGENT_GRAPH_PATH) -> dict[str, DocumentAgent]:
    return load_agents(path)


def calculate_neighbour_activation(
    seed_score: float,
    edge_weight: float,
    graph_weight: float,
) -> float:
    return seed_score * edge_weight * graph_weight


def agent_graph_retrieve(
    index: faiss.Index,
    query_embedding: np.ndarray,
    document_ids: list[str],
    agents: dict[str, DocumentAgent],
    top_k: int,
    seed_k: int,
    graph_weight: float,
) -> list[tuple[str, float]]:
    seed_scores, seed_indices = index.search(query_embedding, seed_k)

    document_scores = defaultdict(float)

    for seed_score, seed_index in zip(seed_scores[0], seed_indices[0]):
        seed_doc_id = document_ids[int(seed_index)]
        seed_score = float(seed_score)

        document_scores[seed_doc_id] += seed_score

        for neighbour_id, edge in agents[seed_doc_id].edges.items():
            neighbour_activation = calculate_neighbour_activation(
                seed_score=seed_score,
                edge_weight=edge.weight,
                graph_weight=graph_weight,
            )
            document_scores[neighbour_id] += neighbour_activation

    ranked_doc_ids = sorted(
        document_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return [(doc_id, float(score)) for doc_id, score in ranked_doc_ids[:top_k]]


def search(
    query: str,
    top_k: int = 10,
    seed_k: int = 5,
    graph_weight: float = 0.3,
) -> tuple[list[dict], float]:
    # TODO: Load this state once at API startup instead of inside each search call.
    documents = load_documents()
    embeddings, document_ids = load_embeddings()
    agents = load_agent_graph()

    index = build_faiss_index(embeddings)
    model = SentenceTransformer(MODEL_NAME)

    start_time = time.perf_counter()

    query_embedding = encode_query(model, query)

    ranked_results = agent_graph_retrieve(
        index=index,
        query_embedding=query_embedding,
        document_ids=document_ids,
        agents=agents,
        top_k=top_k,
        seed_k=seed_k,
        graph_weight=graph_weight,
    )

    latency_ms = (time.perf_counter() - start_time) * 1000

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
