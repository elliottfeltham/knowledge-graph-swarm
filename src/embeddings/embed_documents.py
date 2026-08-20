import json

import numpy as np

from common import PROJECT_ROOT, read_jsonl
from embeddings.model import load_embedding_model


DOCUMENTS_PATH = PROJECT_ROOT / "data" / "processed" / "documents.jsonl"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "document_embeddings.npy"
IDS_PATH = EMBEDDINGS_DIR / "document_ids.json"
QUERIES_PATH = PROJECT_ROOT / "data" / "processed" / "queries.jsonl"
QUERY_EMBEDDINGS_PATH = EMBEDDINGS_DIR / "query_embeddings.npy"
QUERY_IDS_PATH = EMBEDDINGS_DIR / "query_ids.json"


def create_embeddings() -> None:
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    documents = read_jsonl(DOCUMENTS_PATH)
    queries = read_jsonl(QUERIES_PATH)

    doc_ids = []
    texts = []

    for doc in documents:
        doc_ids.append(doc["doc_id"])
        texts.append(f"{doc['title']}\n{doc['text']}")

    print(f"Loaded {len(texts)} documents")

    model = load_embedding_model()

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    np.save(EMBEDDINGS_PATH, embeddings.astype("float32"))

    with open(IDS_PATH, "w", encoding="utf-8") as file:
        json.dump(doc_ids, file, indent=2)

    print(f"Saved embeddings to {EMBEDDINGS_PATH}")
    print(f"Saved document IDs to {IDS_PATH}")
    print(f"Embedding shape: {embeddings.shape}")

    query_ids = [query["query_id"] for query in queries]
    query_texts = [query["query"] for query in queries]
    query_embeddings = model.encode(
        query_texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    np.save(QUERY_EMBEDDINGS_PATH, query_embeddings.astype("float32"))

    with open(QUERY_IDS_PATH, "w", encoding="utf-8") as file:
        json.dump(query_ids, file, indent=2)

    print(f"Saved query embeddings to {QUERY_EMBEDDINGS_PATH}")
    print(f"Saved query IDs to {QUERY_IDS_PATH}")
    print(f"Query embedding shape: {query_embeddings.shape}")


def main() -> None:
    create_embeddings()


if __name__ == "__main__":
    main()
