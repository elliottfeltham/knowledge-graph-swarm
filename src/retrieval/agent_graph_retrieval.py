"""Low-level activation scoring for document-agent graph retrieval."""

import argparse
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np

from agents import DocumentAgent, GraphUpdateConfig
from common import PROJECT_ROOT


AGENT_GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "document_agents.jsonl"


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
    """Retrieve FAISS seeds and rank their graph neighbours by activation."""

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
    seed_k: int = 3,
    graph_weight: float = 0.05,
) -> tuple[list[dict], float]:
    from retrieval.agent_graph_runtime import AgentGraphRuntime

    runtime = AgentGraphRuntime.load()

    return runtime.search(
        query=query,
        top_k=top_k,
        seed_k=seed_k,
        graph_weight=graph_weight,
    )


def adaptive_search(
    query: str,
    top_k: int = 10,
    seed_k: int = 3,
    graph_weight: float = 0.05,
    update_config: GraphUpdateConfig | None = None,
    agent_graph_path: Path = AGENT_GRAPH_PATH,
) -> tuple[list[dict], float, dict]:
    from retrieval.agent_graph_runtime import AgentGraphRuntime

    runtime = AgentGraphRuntime.load(agent_graph_path=agent_graph_path)

    return runtime.adaptive_search(
        query=query,
        top_k=top_k,
        seed_k=seed_k,
        graph_weight=graph_weight,
        update_config=update_config,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent graph retrieval.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--seed-k",
        type=int,
        default=3,
        help="Number of FAISS seed documents to activate.",
    )
    parser.add_argument(
        "--graph-weight",
        type=float,
        default=0.05,
        help="How strongly graph neighbours affect final ranking.",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Update and save the agent graph after retrieval.",
    )
    parser.add_argument(
        "--enable-pruning",
        action="store_true",
        help="Enable pruning during adaptive graph updates.",
    )

    args = parser.parse_args()

    update_summary = None

    if args.adaptive:
        update_config = GraphUpdateConfig(
            enable_pruning=args.enable_pruning,
            update_k=3,
        )
        results, latency_ms, update_summary = adaptive_search(
            query=args.query,
            top_k=args.top_k,
            seed_k=args.seed_k,
            graph_weight=args.graph_weight,
            update_config=update_config,
        )
    else:
        results, latency_ms = search(
            query=args.query,
            top_k=args.top_k,
            seed_k=args.seed_k,
            graph_weight=args.graph_weight,
        )

    print(f"\nQuery: {args.query}")
    print(f"Agent graph search latency: {latency_ms:.2f} ms")
    print(f"Adaptive update: {'yes' if args.adaptive else 'no'}")

    if update_summary is not None:
        print(f"Retrieved documents updated: {update_summary['retrieved_documents']}")
        print(f"Strengthened edges: {update_summary['strengthened_edges']}")
        print(f"Pruned edges: {update_summary['pruned_edges']}")

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
