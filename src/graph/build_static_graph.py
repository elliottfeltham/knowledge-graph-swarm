import argparse
import json
from pathlib import Path

import faiss
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "document_embeddings.npy"
IDS_PATH = PROJECT_ROOT / "data" / "embeddings" / "document_ids.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "static_graph_edges.jsonl"


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


def build_static_graph(embeddings: np.ndarray, document_ids: list[str], neighbours_per_doc: int) -> list[dict]:
    embedding_dimensions = embeddings.shape[1]

    index = faiss.IndexFlatIP(embedding_dimensions)
    index.add(embeddings)

    # Searches for closest match other than itself.
    scores, indices = index.search(embeddings, neighbours_per_doc + 1)

    edges = []
    seen_edges = set()

    for source_index, (row_scores, row_indices) in enumerate(zip(scores, indices)):
        source_id = document_ids[source_index]

        for score, target_index in zip(row_scores, row_indices):
            target_index = int(target_index)

            if target_index == source_index:
                continue

            target_id = document_ids[target_index]

            edge_key = tuple(sorted((source_id, target_id)))

            if edge_key in seen_edges:
                continue

            seen_edges.add(edge_key)

            edges.append(
                {
                    "source": source_id,
                    "target": target_id,
                    "weight": float(score),
                }
            )

    return edges


def save_edges(edges: list[dict]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        for edge in edges:
            file.write(json.dumps(edge) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a static nearest-neighbour document graph.")
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of nearest neighbours to connect each document to.",
    )

    args = parser.parse_args()

    embeddings, document_ids = load_embeddings()

    print(f"Loaded embeddings: {embeddings.shape}")
    print(f"Loaded document IDs: {len(document_ids)}")
    print(f"Building static graph with k={args.k}")

    edges = build_static_graph(
        embeddings=embeddings,
        document_ids=document_ids,
        neighbours_per_doc=args.k,
    )

    save_edges(edges)

    print(f"Saved {len(edges)} edges to {OUTPUT_PATH}")

    average_degree = (2 * len(edges)) / len(document_ids)

    print(f"Nodes: {len(document_ids)}")
    print(f"Edges: {len(edges)}")
    print(f"Approx average degree: {average_degree:.2f}")


if __name__ == "__main__":
    main()
