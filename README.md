# An Emergent Knowledge Graph Using Swarm Intelligence

This is an emergent knowledge graph that uses swarm intelligence to organise itself. Currently, the system works in a terminal as a CLI but will include a 3D visualisation and an API.

This is a work-in-progress MSc dissertation project.

## Setup

Install the project and dependencies with `pip install -r requirements.txt` from the repository root.

## Quick Demo

These commands rebuild the small HotpotQA dataset, create the graphs, and run the current agent graph retrieval system.

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/data/load_dataset.py
python src/embeddings/embed_documents.py
python src/graph/build_static_graph.py
PYTHONPATH=src python src/graph/initialize_agent_graph.py
PYTHONPATH=src python src/retrieval/agent_graph_retrieval.py --query "Which magazine was started first Arthur's Magazine or First for Women?"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/data/load_dataset.py
python src/embeddings/embed_documents.py
python src/graph/build_static_graph.py
$env:PYTHONPATH = "src"
python src/graph/initialize_agent_graph.py
python src/retrieval/agent_graph_retrieval.py --query "Which magazine was started first Arthur's Magazine or First for Women?"
```

The final command should return `Arthur's Magazine` and `First for Women` near the top of the results.

To run a quick frozen evaluation across FAISS, the static graph, and the agent graph:

```bash
PYTHONPATH=src python src/evaluation/evaluate_baselines.py --limit 10
```

To simulate user queries and compare the agent graph before and after adaptation:

```bash
PYTHONPATH=src python src/evaluation/simulate_user_workflow.py --adaptation-queries 100 --evaluation-queries 100 --disable-pruning
```

This copies the current agent graph into `results/`, adapts the copy, and leaves the main saved graph unchanged.
The summary also reports graph statistics such as edge count, average degree, average edge weight, and the largest changed edges.

To see the adaptive update mode, run:

```bash
PYTHONPATH=src python src/retrieval/agent_graph_retrieval.py --query "Which magazine was started first Arthur's Magazine or First for Women?" --adaptive --disable-pruning
```

This runs the same retrieval, then strengthens edges between the returned documents and saves the updated agent graph.

## Evaluation Pipeline

This is the pipeline so far for evaluating FAISS, the static graph baseline, and the agent graph retrieval system.

1. Build the processed dataset with [src/data/load_dataset.py](src/data/load_dataset.py).
2. Generate document embeddings with [src/embeddings/embed_documents.py](src/embeddings/embed_documents.py).
3. Build the static nearest-neighbour graph with [src/graph/build_static_graph.py](src/graph/build_static_graph.py).
4. Run a single-query FAISS baseline with [src/retrieval/faiss_baseline.py](src/retrieval/faiss_baseline.py).
5. Run the graph-augmented baseline with [src/retrieval/static_graph_baseline.py](src/retrieval/static_graph_baseline.py).
6. Evaluate FAISS, the static graph, and the agent graph over the query set with [src/evaluation/evaluate_baselines.py](src/evaluation/evaluate_baselines.py).

## Agent Graph Pipeline

The agent graph is the first version of the adaptive system. Each document is loaded into a `DocumentAgent` object, and the agent's connections are saved to disk as JSONL so the graph can be inspected and loaded again later.

At runtime, the Python object is the document agent. The JSONL file is only the saved representation of those agents.

For now, the agent graph can be built directly from the current processed documents and embeddings:

```bash
PYTHONPATH=src .venv/bin/python src/graph/initialize_agent_graph.py
```

This creates:

```text
data/processed/document_agents.jsonl
```

The generated file is ignored by Git because it can be recreated from the processed documents and embeddings. Later, this step should be wrapped into a single build command for the whole system.

## Agent Graph Retrieval

The agent graph retrieval script uses FAISS to find the first set of related document agents, then lets activation spread once through their graph connections.

By default, this is a read-only retrieval step. It uses the saved agent graph but does not strengthen, decay or prune any connections.

```bash
PYTHONPATH=src .venv/bin/python src/retrieval/agent_graph_retrieval.py --query "Which magazine was started first Arthur's Magazine or First for Women?"
```

Arguments:

`--query`: the search query to run. This is required.

`--top-k`: controls how many ranked documents are returned. The default is `10`. A smaller value, such as `5`, is useful for quick inspection. A larger value, such as `20`, shows more of the graph-influenced ranking.

`--seed-k`: controls how many FAISS results are used as the starting agents. The default is `5`. A smaller value, such as `3`, keeps the search focused on the closest semantic matches. A larger value, such as `10`, lets more documents spread activation into the graph.

`--graph-weight`: controls how much influence neighbouring agents have on the final ranking. The default is `0.3`. `0.0` ignores the graph, `0.3` uses the graph gently, and `1.0` gives graph connections a much stronger effect.

`--adaptive`: updates and saves the agent graph after retrieval. This strengthens edges between documents retrieved together, decays unused edges, and prunes weak edges.

`--disable-pruning`: skips pruning during adaptive updates. This is useful while testing the adaptive system without immediately removing weak initial edges.

To run an adaptive query without pruning:

```bash
PYTHONPATH=src .venv/bin/python src/retrieval/agent_graph_retrieval.py --query "Which magazine was started first Arthur's Magazine or First for Women?" --adaptive --disable-pruning
```

## Data And Outputs

Generated documents, embeddings, graph edges, and results are written under `data/` and `results/` and are ignored by Git.
