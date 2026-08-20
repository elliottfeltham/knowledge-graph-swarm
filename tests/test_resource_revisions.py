from data import load_dataset as dataset_module
from embeddings import model as model_module


def test_dataset_loader_uses_pinned_revision(monkeypatch, tmp_path) -> None:
    loader_arguments = {}

    class EmptyDataset:
        def select(self, indexes):
            return []

    def fake_load_dataset(name, config, **kwargs):
        loader_arguments.update(name=name, config=config, **kwargs)
        return EmptyDataset()

    monkeypatch.setattr(dataset_module, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(dataset_module, "PROCESSED_DIR", tmp_path)

    dataset_module.prepare_dataset(limit=0)

    assert loader_arguments == {
        "name": "hotpotqa/hotpot_qa",
        "config": "distractor",
        "split": "train",
        "revision": "1908d6afbbead072334abe2965f91bd2709910ab",
    }


def test_embedding_model_loader_uses_pinned_revision(monkeypatch) -> None:
    loader_arguments = {}

    def fake_sentence_transformer(name, **kwargs):
        loader_arguments.update(name=name, **kwargs)
        return object()

    monkeypatch.setattr(
        model_module,
        "SentenceTransformer",
        fake_sentence_transformer,
    )

    model_module.load_embedding_model()

    assert loader_arguments == {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    }
