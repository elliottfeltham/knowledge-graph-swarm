from agents import DocumentAgent
from retrieval.agent_graph_runtime import AgentGraphRuntime


def create_runtime(agents: dict[str, DocumentAgent]) -> AgentGraphRuntime:
    runtime = object.__new__(AgentGraphRuntime)
    runtime.agents = agents

    return runtime


def connect_agents(
    agents: dict[str, DocumentAgent],
    source_id: str,
    target_id: str,
    weight: float,
) -> None:
    agents[source_id].add_edge(target_id, weight)
    agents[target_id].add_edge(source_id, weight)


def test_subgraph_keeps_roots_and_selects_strongest_neighbours() -> None:
    agents = {
        doc_id: DocumentAgent(doc_id=doc_id, title=doc_id.upper(), text="")
        for doc_id in ["a", "b", "c", "d"]
    }
    connect_agents(agents, "a", "b", 0.9)
    connect_agents(agents, "a", "c", 0.7)
    connect_agents(agents, "a", "d", 0.5)
    runtime = create_runtime(agents)

    graph_data = runtime.subgraph_data(
        document_ids=["a"],
        depth=1,
        max_nodes=3,
    )

    assert [node["id"] for node in graph_data["nodes"]] == ["a", "b", "c"]
    assert graph_data["total_edges"] == 2
    assert graph_data["truncated"] is True


def test_empty_subgraph_request_starts_from_highest_degree_agents() -> None:
    agents = {
        doc_id: DocumentAgent(doc_id=doc_id, title=doc_id.upper(), text="")
        for doc_id in ["a", "b", "c"]
    }
    connect_agents(agents, "a", "b", 0.9)
    connect_agents(agents, "a", "c", 0.7)
    runtime = create_runtime(agents)

    graph_data = runtime.subgraph_data(
        document_ids=[],
        depth=0,
        max_nodes=1,
    )

    assert [node["id"] for node in graph_data["nodes"]] == ["a"]


def test_document_data_returns_a_bounded_preview() -> None:
    agents = {
        "a": DocumentAgent(doc_id="a", title="Document A", text="Long text")
    }
    runtime = create_runtime(agents)

    document = runtime.document_data("a", preview_length=4)

    assert document == {
        "doc_id": "a",
        "title": "Document A",
        "text_preview": "Long",
        "degree": 0,
    }
