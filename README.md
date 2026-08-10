# An Emergent Knowledge Graph Using Swarm Intelligence

This project investigates whether document retrieval can be improved by allowing a knowledge graph to self-organise through use. Each document is represented by an agent with mutable weighted relationships to other documents. When documents are retrieved together, their relationship can strengthen or form, while relationships that are no longer useful decay and may eventually be removed through pruning. Over time, these local update rules allow the graph’s structure to adapt to changing retrieval patterns.

The system includes a command-line interface, a FastAPI backend and a React frontend with an interactive 3D graph.

## Requirements

- Python 3 and `pip`
- Node.js and `npm` for the frontend

## Setup

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

## Running The Application

Prepare the dataset and graphs before starting the application.

On macOS or Linux:

```bash
PYTHONPATH=src python src/prepare_system.py
```

On Windows PowerShell:

```powershell
python src/prepare_system.py
```

This downloads the small HotpotQA dataset, creates document and query embeddings, and builds both the static and adaptive graphs. An internet connection is required the first time it runs.

Start the backend from the repository root.

On macOS or Linux:

```bash
PYTHONPATH=src python -m uvicorn api.main:app --port 8000
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m uvicorn api.main:app --port 8000
```

In a second terminal, start the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5174` in a browser. The interface includes example HotpotQA questions that can be used to try the current corpus.

Standard search retrieves documents without changing the graph.
Adaptive search saves changes to relationships between the retrieved document agents.
The Graph Statistics panel shows the current graph structure and can reset it to the original nearest-neighbour graph after testing.

The focused graph is intended for exploring local document relationships. The full graph shows the complete corpus in an observation-only view.

## Command-Line Demo

A search can also be run directly from the repository root:

```bash
PYTHONPATH=src python src/retrieval/agent_graph_retrieval.py --query "Which magazine was started first Arthur's Magazine or First for Women?"
```

The command should return `Arthur's Magazine` and `First for Women` near the top of the results.

Add `--adaptive` to update the graph after retrieval:

```bash
PYTHONPATH=src python src/retrieval/agent_graph_retrieval.py --query "Which magazine was started first Arthur's Magazine or First for Women?" --adaptive --disable-pruning
```

## Running The Evaluation

The complete dissertation experiment can be reproduced with one command:

```bash
PYTHONPATH=src python src/run_full_evaluation.py
```

This prepares the system, runs the main evaluation, compares pruning enabled and disabled, and creates the final CSV tables and figures. The experiment uses three fixed query splits with seeds `10`, `20` and `30`. Outputs are saved under `results/`.

If you prepared the system while running the application, reuse those files with:

```bash
PYTHONPATH=src python src/run_full_evaluation.py --skip-preparation
```

The stages can also be run separately when only one part of the evaluation is needed:

```text
PYTHONPATH=src python src/run_experiments.py
PYTHONPATH=src python src/run_experiments.py --pruning-ablation
PYTHONPATH=src python src/evaluation/create_results_report.py
```

Evaluation uses held-out queries and copied graph files. It does not modify the graph used by the interactive application.

## Running The Tests

Run the Python tests from the repository root:

```bash
PYTHONPATH=src python -m pytest
```

Run the frontend tests and production build from the `frontend/` directory:

```bash
cd frontend
npm run test
npm run build
```

## How The System Works

Each document is loaded as a `DocumentAgent`. Its weighted edges represent relationships with other document agents, while the JSONL graph file is the saved version of that state.

Retrieval starts with FAISS semantic search. The highest-ranked documents become seed agents and spread activation once through their neighbours. The semantic score and neighbour activation are then combined to produce the final ranking.

Read-only retrieval leaves the graph unchanged. Adaptive retrieval strengthens or creates relationships between documents retrieved together, decays unused relationships and optionally prunes weak ones. Evaluation queries are always frozen so that measuring the system does not change the graph being measured.

The `AgentGraphRuntime` loads the documents, embeddings, FAISS index, embedding model and document agents once. The command-line interface and FastAPI backend both use this shared runtime.

## Backend API

FastAPI's interactive documentation is available at `http://127.0.0.1:8000/docs` while the backend is running. It lists the search, adaptive search, graph, document, statistics and reset endpoints together with their request and response formats.

## Current Scope

The current implementation uses a prepared subset of HotpotQA. It supports frozen and adaptive retrieval, repeatable comparison with FAISS and a static graph, and interactive graph visualisation. A document ingestion pipeline and evaluation on additional datasets are possible future extensions.

Generated documents, embeddings, graph files and experiment results are written under `data/` and `results/`. These files are ignored by Git because they can be recreated from the preparation and evaluation scripts.
