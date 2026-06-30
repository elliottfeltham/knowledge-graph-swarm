import argparse
import time
from sentence_transformers import SentenceTransformer

from src.common import (
    MODEL_NAME,
    build_faiss_index,
    encode_query,
    faiss_retrieve,
    load_documents,
    load_embeddings,
)


def search(query: str, top_k: int = 10) -> tuple[list[dict], float]:
    documents = load_documents()
    embeddings, document_ids = load_embeddings()

    index = build_faiss_index(embeddings)

    model = SentenceTransformer(MODEL_NAME)

    start_time = time.perf_counter()

    query_embedding = encode_query(model, query)

    ranked_results = faiss_retrieve(
        index=index,
        query_embedding=query_embedding,
        document_ids=document_ids,
        top_k=top_k,
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
