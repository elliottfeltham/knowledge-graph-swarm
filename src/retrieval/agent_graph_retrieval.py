from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np

from agents import DocumentAgent, load_agents
from common import PROJECT_ROOT


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
