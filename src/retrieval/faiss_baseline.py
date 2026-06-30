import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOCUMENTS_PATH = PROJECT_ROOT / "data" / "processed" / "documents.jsonl"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "document_embeddings.npy"
IDS_PATH = PROJECT_ROOT / "data" / "embeddings" / "document_ids.json"

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

    # IndexFlatIP = exact inner product search.
    # Because our embeddings were normalised, inner product is equivalent to cosine similarity.
    index = faiss.IndexFlatIP(embedding_dimensions)
    index.add(embeddings)

    return index


def search(query: str, top_k: int = 10) -> tuple[list[dict], float]:
    documents = load_documents()
    embeddings, document_ids = load_embeddings()

    index = build_faiss_index(embeddings)

    model = SentenceTransformer(MODEL_NAME)

    start_time = time.perf_counter()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    latency_ms = (time.perf_counter() - start_time) * 1000

    results = []

    for score, index_position in zip(scores[0], indices[0]):
        doc_id = document_ids[int(index_position)]
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
    parser = argparse.ArgumentParser(description="Run FAISS baseline retrieval.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=10)

    args = parser.parse_args()

    results, latency_ms = search(args.query, args.top_k)

    print(f"\nQuery: {args.query}")
    print(f"FAISS search latency: {latency_ms:.2f} ms")
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
