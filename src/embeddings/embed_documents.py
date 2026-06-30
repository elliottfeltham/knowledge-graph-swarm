import json

import numpy as np
from sentence_transformers import SentenceTransformer

from src.common import MODEL_NAME, PROJECT_ROOT, read_jsonl


DOCUMENTS_PATH = PROJECT_ROOT / "data" / "processed" / "documents.jsonl"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "document_embeddings.npy"
IDS_PATH = EMBEDDINGS_DIR / "document_ids.json"


def main() -> None:
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    documents = read_jsonl(DOCUMENTS_PATH)

    doc_ids = []
    texts = []

    for doc in documents:
        doc_ids.append(doc["doc_id"])
        texts.append(f"{doc['title']}\n{doc['text']}")

    print(f"Loaded {len(texts)} documents")

    model = SentenceTransformer(MODEL_NAME)

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


if __name__ == "__main__":
    main()
