from sentence_transformers import SentenceTransformer

from common import MODEL_NAME, MODEL_REVISION


def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION)
