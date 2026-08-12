from unittest.mock import Mock

import numpy as np

from agents import DocumentAgent
from agents.graph_update import GraphUpdateConfig
import retrieval.agent_graph_runtime as runtime_module
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


def test_search_does_not_change_or_save_the_graph(monkeypatch) -> None:
    agents = {
        "a": DocumentAgent(doc_id="a", title="Document A", text="Text A"),
        "b": DocumentAgent(doc_id="b", title="Document B", text="Text B"),
    }
    connect_agents(agents, "a", "b", 0.2)
    runtime = create_runtime(agents)
    runtime.documents = {
        "a": {"title": "Document A", "text": "Text A"},
        "b": {"title": "Document B", "text": "Text B"},
    }
    runtime.document_ids = ["a", "b"]
    runtime.index = object()
    runtime.model = object()
    runtime.save_graph = Mock()

    monkeypatch.setattr(
        runtime_module,
        "encode_query",
        lambda model, query: np.array([[0.1, 0.2]], dtype="float32"),
    )
    monkeypatch.setattr(
        runtime_module,
        "agent_graph_retrieve",
        lambda **kwargs: [("a", 0.8), ("b", 0.6)],
    )

    results, _ = runtime.search("example query")

    assert [result["doc_id"] for result in results] == ["a", "b"]
    assert agents["a"].edges["b"].weight == 0.2
    runtime.save_graph.assert_not_called()


def test_adaptive_search_updates_and_saves_the_graph() -> None:
    agents = {
        "a": DocumentAgent(doc_id="a", title="Document A", text=""),
        "b": DocumentAgent(doc_id="b", title="Document B", text=""),
    }
    runtime = create_runtime(agents)
    runtime.search = Mock(
        return_value=(
            [
                {"doc_id": "a"},
                {"doc_id": "b"},
            ],
            5.0,
        )
    )
    runtime.save_graph = Mock()

    _, _, summary = runtime.adaptive_search(
        "example query",
        update_config=GraphUpdateConfig(
            strengthen_amount=0.1,
            enable_pruning=False,
        ),
    )

    assert agents["a"].edges["b"].weight == 0.1
    assert summary["created_edges"] == 1
    runtime.save_graph.assert_called_once_with()


def test_adaptive_search_uses_selective_updates_by_default() -> None:
    agents = {
        doc_id: DocumentAgent(doc_id=doc_id, title=doc_id, text="")
        for doc_id in ("a", "b", "c", "d")
    }
    runtime = create_runtime(agents)
    runtime.search = Mock(
        return_value=(
            [{"doc_id": doc_id} for doc_id in agents],
            5.0,
        )
    )
    runtime.save_graph = Mock()

    _, _, summary = runtime.adaptive_search("example query")

    assert summary["retrieved_documents"] == 4
    assert summary["strengthened_edges"] == 3
    assert agents["d"].edges == {}


def test_reset_replaces_and_saves_the_initial_graph(monkeypatch) -> None:
    current_agents = {
        "a": DocumentAgent(doc_id="a", title="Document A", text=""),
    }
    initial_agents = {
        "a": DocumentAgent(doc_id="a", title="Document A", text=""),
        "b": DocumentAgent(doc_id="b", title="Document B", text=""),
    }
    connect_agents(initial_agents, "a", "b", 0.2)
    runtime = create_runtime(current_agents)
    runtime.save_graph = Mock()

    monkeypatch.setattr(
        runtime_module,
        "build_initial_agent_graph",
        lambda neighbours_per_doc, initial_weight_scale: initial_agents,
    )

    statistics = runtime.reset_graph()

    assert runtime.agents is initial_agents
    assert statistics["agents"] == 2
    assert statistics["undirected_edges"] == 1
    runtime.save_graph.assert_called_once_with()
