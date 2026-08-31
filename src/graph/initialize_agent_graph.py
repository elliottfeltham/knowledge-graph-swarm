"""Initialise document agents with weak embedding-neighbour relationships."""

import argparse
from pathlib import Path

import numpy as np

from agents import DocumentAgent, save_agents
from common import build_faiss_index, load_documents, load_embeddings
from graph.config import INITIAL_WEIGHT_SCALE, NEIGHBOURS_PER_DOCUMENT


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "document_agents.jsonl"


def create_agents(documents: dict[str, dict], document_ids: list[str]) -> dict[str, DocumentAgent]:
    """Create one document agent for every embedded corpus document."""

    agents = {}

    for doc_id in document_ids:
        document = documents[doc_id]

        agents[doc_id] = DocumentAgent(
            doc_id=doc_id,
            title=document["title"],
            text=document["text"],
        )

    return agents


def add_weak_neighbour_edges(
    agents: dict[str, DocumentAgent],
    embeddings: np.ndarray,
    document_ids: list[str],
    neighbours_per_doc: int,
    initial_weight_scale: float,
) -> None:
    """Connect embedding neighbours using scaled cosine-similarity weights."""

    index = build_faiss_index(embeddings)

    scores, indices = index.search(embeddings, neighbours_per_doc + 1)

    for source_index, (row_scores, row_indices) in enumerate(zip(scores, indices)):
        source_id = document_ids[source_index]

        for score, target_index in zip(row_scores, row_indices):
            target_index = int(target_index)

            if target_index == source_index:
                continue

            target_id = document_ids[target_index]
            weak_weight = float(score) * initial_weight_scale

            add_undirected_edge(
                agents=agents,
                source_id=source_id,
                target_id=target_id,
                weight=weak_weight,
            )


def add_undirected_edge(
    agents: dict[str, DocumentAgent],
    source_id: str,
    target_id: str,
    weight: float,
) -> None:
    current_edge = agents[source_id].edges.get(target_id)

    if current_edge is not None and current_edge.weight >= weight:
        return

    agents[source_id].add_edge(target_id, weight)
    agents[target_id].add_edge(source_id, weight)


def count_undirected_edges(agents: dict[str, DocumentAgent]) -> int:
    return sum(len(agent.edges) for agent in agents.values()) // 2


def build_initial_agent_graph(
    neighbours_per_doc: int,
    initial_weight_scale: float,
) -> dict[str, DocumentAgent]:
    """Build a fresh weak graph from the prepared documents and embeddings."""

    documents = load_documents()
    embeddings, document_ids = load_embeddings()

    agents = create_agents(documents=documents, document_ids=document_ids)

    add_weak_neighbour_edges(
        agents=agents,
        embeddings=embeddings,
        document_ids=document_ids,
        neighbours_per_doc=neighbours_per_doc,
        initial_weight_scale=initial_weight_scale,
    )

    return agents


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialise document agents with weak nearest-neighbour edges.")
    parser.add_argument(
        "--k",
        type=int,
        default=NEIGHBOURS_PER_DOCUMENT,
        help="Number of nearest neighbours to connect each document to.",
    )
    parser.add_argument(
        "--initial-weight-scale",
        type=float,
        default=INITIAL_WEIGHT_SCALE,
        help="Multiplier applied to embedding similarity when creating weak initial edges.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=OUTPUT_PATH,
        help="Where to save the generated document agents JSONL file.",
    )

    args = parser.parse_args()

    agents = build_initial_agent_graph(
        neighbours_per_doc=args.k,
        initial_weight_scale=args.initial_weight_scale,
    )

    save_agents(args.output_path, agents)

    agent_count = len(agents)
    edge_count = count_undirected_edges(agents)
    average_degree = (2 * edge_count) / agent_count if agent_count else 0.0

    print(f"Agents: {agent_count}")
    print(f"Undirected edges: {edge_count}")
    print(f"Average degree: {average_degree:.2f}")
    print(f"Saved agents to: {args.output_path}")


if __name__ == "__main__":
    main()
