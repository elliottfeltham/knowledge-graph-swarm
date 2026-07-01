import argparse
import time
from sentence_transformers import SentenceTransformer

from common import (
    MODEL_NAME,
    build_faiss_index,
    encode_query,
    load_documents,
    load_embeddings,
    load_static_graph,
    static_graph_retrieve,
)


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

    query_embedding = encode_query(model, query)

    latency_ms = (time.perf_counter() - start_time) * 1000

    ranked_results = static_graph_retrieve(
        index=index,
        query_embedding=query_embedding,
        document_ids=document_ids,
        graph=graph,
        top_k=top_k,
        seed_k=seed_k,
        graph_activation_weight=graph_activation_weight,
    )

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
