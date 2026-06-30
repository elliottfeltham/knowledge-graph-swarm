import json
from datasets import load_dataset

from src.common import PROJECT_ROOT

dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
subset = dataset.select(range(500))

processed_dir = PROJECT_ROOT / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

documents = {}
queries = []
qrels = []

def normalise_doc_id(title):
    return "doc_" + "".join(
        char.lower() if char.isalnum() else "_"
        for char in title
    ).strip("_")

# Build documents, queries, and qrels from the HotpotQA dataset
for example in subset:
    query_id = example["id"]

    queries.append({
        "query_id": query_id,
        "query": example["question"],
        "answer": example["answer"]
    })

    titles = example["context"]["title"]
    sentence_groups = example["context"]["sentences"]

    for title, sentences in zip(titles, sentence_groups):
        doc_id = normalise_doc_id(title)
        text = " ".join(sentences)

        documents[doc_id] = {
            "doc_id": doc_id,
            "title": title,
            "text": text
        }

    supporting_titles = set(example["supporting_facts"]["title"])

    for title in supporting_titles:
        qrels.append({
            "query_id": query_id,
            "doc_id": normalise_doc_id(title),
            "score": 1
        })

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

write_jsonl(processed_dir / "documents.jsonl", documents.values())
write_jsonl(processed_dir / "queries.jsonl", queries)
write_jsonl(processed_dir / "qrels.jsonl", qrels)

print(f"Documents: {len(documents)}")
print(f"Queries: {len(queries)}")
print(f"Qrels: {len(qrels)}")
