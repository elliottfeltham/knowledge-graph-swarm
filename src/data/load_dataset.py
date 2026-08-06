import json

from datasets import load_dataset

from common import PROJECT_ROOT


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def normalise_doc_id(title: str) -> str:
    return "doc_" + "".join(
        char.lower() if char.isalnum() else "_"
        for char in title
    ).strip("_")

def write_jsonl(path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_dataset(limit: int = 500) -> None:
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    subset = dataset.select(range(limit))

    documents = {}
    queries = []
    qrels = []

    for example in subset:
        query_id = example["id"]

        queries.append(
            {
                "query_id": query_id,
                "query": example["question"],
                "answer": example["answer"],
            }
        )

        titles = example["context"]["title"]
        sentence_groups = example["context"]["sentences"]

        for title, sentences in zip(titles, sentence_groups):
            doc_id = normalise_doc_id(title)
            text = " ".join(sentences)

            documents[doc_id] = {
                "doc_id": doc_id,
                "title": title,
                "text": text,
            }

        supporting_titles = set(example["supporting_facts"]["title"])

        for title in supporting_titles:
            qrels.append(
                {
                    "query_id": query_id,
                    "doc_id": normalise_doc_id(title),
                    "score": 1,
                }
            )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(PROCESSED_DIR / "documents.jsonl", documents.values())
    write_jsonl(PROCESSED_DIR / "queries.jsonl", queries)
    write_jsonl(PROCESSED_DIR / "qrels.jsonl", qrels)

    print(f"Documents: {len(documents)}")
    print(f"Queries: {len(queries)}")
    print(f"Qrels: {len(qrels)}")


def main() -> None:
    prepare_dataset()


if __name__ == "__main__":
    main()
