"""Prepare the dataset, embeddings and graph files required by the system."""

from agents import save_agents
from common import (
    DOCUMENTS_PATH,
    EMBEDDINGS_PATH,
    IDS_PATH,
    QRELS_PATH,
    QUERIES_PATH,
    QUERY_EMBEDDINGS_PATH,
    QUERY_IDS_PATH,
    load_embeddings,
)
from data.load_dataset import prepare_dataset
from embeddings.embed_documents import create_embeddings
from graph.config import (
    INITIAL_WEIGHT_SCALE,
    NEIGHBOURS_PER_DOCUMENT,
)
from graph.build_static_graph import OUTPUT_PATH as STATIC_GRAPH_PATH
from graph.build_static_graph import build_static_graph, save_edges
from graph.initialize_agent_graph import OUTPUT_PATH as AGENT_GRAPH_PATH
from graph.initialize_agent_graph import build_initial_agent_graph

def print_stage(number: int, description: str) -> None:
    print(f"\n[{number}/4] {description}")


def prepare_system() -> None:
    """Run the complete data preparation workflow in dependency order."""
    print_stage(1, "Preparing the HotpotQA dataset")
    prepare_dataset()

    print_stage(2, "Creating document and query embeddings")
    create_embeddings()

    embeddings, document_ids = load_embeddings()

    print_stage(3, "Building the static graph")
    static_edges = build_static_graph(
        embeddings=embeddings,
        document_ids=document_ids,
        neighbours_per_doc=NEIGHBOURS_PER_DOCUMENT,
    )
    save_edges(static_edges)
    print(f"Saved {len(static_edges)} edges to {STATIC_GRAPH_PATH}")

    print_stage(4, "Initialising the agent graph")
    agents = build_initial_agent_graph(
        neighbours_per_doc=NEIGHBOURS_PER_DOCUMENT,
        initial_weight_scale=INITIAL_WEIGHT_SCALE,
    )
    save_agents(AGENT_GRAPH_PATH, agents)
    print(f"Saved {len(agents)} document agents to {AGENT_GRAPH_PATH}")

    print("\nSystem preparation complete.")
    print(f"Documents: {DOCUMENTS_PATH}")
    print(f"Queries: {QUERIES_PATH}")
    print(f"Relevance judgements: {QRELS_PATH}")
    print(f"Embeddings: {EMBEDDINGS_PATH}")
    print(f"Document IDs: {IDS_PATH}")
    print(f"Query embeddings: {QUERY_EMBEDDINGS_PATH}")
    print(f"Query IDs: {QUERY_IDS_PATH}")
    print(f"Static graph: {STATIC_GRAPH_PATH}")
    print(f"Agent graph: {AGENT_GRAPH_PATH}")


if __name__ == "__main__":
    prepare_system()
