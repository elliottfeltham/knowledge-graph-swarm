from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import app


SEARCH_RESULT = {
    "doc_id": "document-1",
    "title": "Example Document",
    "score": 0.85,
    "text_preview": "An example document used by the API tests.",
}

GRAPH_STATISTICS = {
    "agent_count": 3,
    "undirected_edges": 2,
    "average_degree": 1.33,
    "average_edge_weight": 0.12,
}

GRAPH_DATA = {
    "nodes": [
        {
            "id": "document-1",
            "title": "Example Document",
            "degree": 2,
        }
    ],
    "edges": [],
    "total_nodes": 1,
    "total_edges": 0,
    "truncated": False,
}


class FakeAgentGraphRuntime:
    def __init__(self) -> None:
        self.search_arguments = None
        self.adaptive_search_arguments = None
        self.subgraph_arguments = None

    def search(self, query: str, top_k: int, seed_k: int, graph_weight: float):
        self.search_arguments = {
            "query": query,
            "top_k": top_k,
            "seed_k": seed_k,
            "graph_weight": graph_weight,
        }
        return [SEARCH_RESULT], 12.5

    def adaptive_search(self, query: str, top_k: int, seed_k: int, graph_weight: float):
        self.adaptive_search_arguments = {
            "query": query,
            "top_k": top_k,
            "seed_k": seed_k,
            "graph_weight": graph_weight,
        }
        update_summary = {
            "strengthened_edges": 1,
            "decayed_edges": 2,
            "pruned_edges": 0,
        }
        return [SEARCH_RESULT], 14.0, update_summary

    def graph_statistics(self) -> dict:
        return GRAPH_STATISTICS

    def graph_data(self) -> dict:
        return GRAPH_DATA

    def subgraph_data(self, document_ids: list[str], depth: int, max_nodes: int) -> dict:
        self.subgraph_arguments = {
            "document_ids": document_ids,
            "depth": depth,
            "max_nodes": max_nodes,
        }

        if "missing-document" in document_ids:
            raise ValueError("Unknown document IDs: missing-document")

        return GRAPH_DATA

    def reset_graph(self) -> dict:
        return GRAPH_STATISTICS


@pytest.fixture
def api_client():
    fake_runtime = FakeAgentGraphRuntime()

    with patch("api.main.AgentGraphRuntime.load", return_value=fake_runtime):
        with TestClient(app) as client:
            yield client, fake_runtime


def test_health_reports_loaded_runtime(api_client) -> None:
    client, _ = api_client

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "runtime_loaded": True,
    }


def test_search_returns_results_and_forwards_arguments(api_client) -> None:
    client, fake_runtime = api_client
    request_body = {
        "query": "example query",
        "top_k": 4,
        "seed_k": 2,
        "graph_weight": 0.5,
    }

    response = client.post("/search", json=request_body)

    assert response.status_code == 200
    assert response.json() == {
        "query": "example query",
        "adaptive": False,
        "latency_ms": 12.5,
        "results": [SEARCH_RESULT],
    }
    assert fake_runtime.search_arguments == request_body


def test_adaptive_search_returns_update_summary(api_client) -> None:
    client, fake_runtime = api_client
    request_body = {
        "query": "adaptive example",
        "top_k": 6,
        "seed_k": 3,
        "graph_weight": 0.4,
    }

    response = client.post("/adaptive-search", json=request_body)

    assert response.status_code == 200
    assert response.json() == {
        "query": "adaptive example",
        "adaptive": True,
        "latency_ms": 14.0,
        "results": [SEARCH_RESULT],
        "update_summary": {
            "strengthened_edges": 1,
            "decayed_edges": 2,
            "pruned_edges": 0,
        },
    }
    assert fake_runtime.adaptive_search_arguments == request_body


def test_graph_stats_returns_current_statistics(api_client) -> None:
    client, _ = api_client

    response = client.get("/graph/stats")

    assert response.status_code == 200
    assert response.json() == GRAPH_STATISTICS


def test_full_graph_returns_graph_data(api_client) -> None:
    client, _ = api_client

    response = client.get("/graph")

    assert response.status_code == 200
    assert response.json() == GRAPH_DATA


def test_subgraph_returns_graph_data_and_forwards_arguments(api_client) -> None:
    client, fake_runtime = api_client
    request_body = {
        "document_ids": ["document-1"],
        "depth": 2,
        "max_nodes": 100,
    }

    response = client.post("/graph/subgraph", json=request_body)

    assert response.status_code == 200
    assert response.json() == GRAPH_DATA
    assert fake_runtime.subgraph_arguments == request_body


def test_subgraph_uses_overview_defaults(api_client) -> None:
    client, fake_runtime = api_client

    response = client.post("/graph/subgraph", json={})

    assert response.status_code == 200
    assert fake_runtime.subgraph_arguments == {
        "document_ids": [],
        "depth": 1,
        "max_nodes": 250,
    }


@pytest.mark.parametrize(
    "request_body",
    [
        {"depth": 4},
        {"max_nodes": 0},
    ],
)
def test_subgraph_rejects_invalid_limits(api_client, request_body) -> None:
    client, _ = api_client

    response = client.post("/graph/subgraph", json=request_body)

    assert response.status_code == 422


def test_subgraph_rejects_unknown_document_ids(api_client) -> None:
    client, _ = api_client

    response = client.post(
        "/graph/subgraph",
        json={"document_ids": ["missing-document"]},
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "Unknown document IDs: missing-document",
    }


def test_reset_graph_returns_rebuilt_statistics(api_client) -> None:
    client, _ = api_client

    response = client.post("/graph/reset")

    assert response.status_code == 200
    assert response.json() == {
        "status": "reset",
        "graph_statistics": GRAPH_STATISTICS,
    }


@pytest.mark.parametrize(
    "request_body",
    [
        {"query": "   "},
        {"query": "example query", "top_k": 0},
    ],
)
def test_search_rejects_invalid_requests(api_client, request_body) -> None:
    client, _ = api_client

    response = client.post("/search", json=request_body)

    assert response.status_code == 422
