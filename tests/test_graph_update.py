import pytest

from agents.document_agent import DocumentAgent
from agents.graph_update import GraphUpdateConfig, update_graph_after_retrieval


def create_agents() -> dict[str, DocumentAgent]:
    return {
        doc_id: DocumentAgent(doc_id=doc_id, title=doc_id, text="")
        for doc_id in ("doc-1", "doc-2", "doc-3")
    }


def add_undirected_edge(
    agents: dict[str, DocumentAgent],
    source_id: str,
    target_id: str,
    weight: float,
) -> None:
    agents[source_id].add_edge(target_id, weight)
    agents[target_id].add_edge(source_id, weight)


def test_update_reports_strengthened_and_created_edges() -> None:
    agents = create_agents()
    add_undirected_edge(agents, "doc-1", "doc-2", 0.2)

    summary = update_graph_after_retrieval(
        agents,
        ["doc-1", "doc-2", "doc-3"],
        GraphUpdateConfig(strengthen_amount=0.1, enable_pruning=False),
    )

    assert summary["strengthened_edges"] == 3
    assert summary["created_edges"] == 2
    assert summary["pruned_edges"] == 0
    assert summary["edge_updates"] == [
        {
            "source": "doc-1",
            "target": "doc-2",
            "previous_weight": 0.2,
            "new_weight": pytest.approx(0.3),
            "change_type": "strengthened",
        },
        {
            "source": "doc-1",
            "target": "doc-3",
            "previous_weight": 0.0,
            "new_weight": 0.1,
            "change_type": "created",
        },
        {
            "source": "doc-2",
            "target": "doc-3",
            "previous_weight": 0.0,
            "new_weight": 0.1,
            "change_type": "created",
        },
    ]


def test_update_reports_pruned_edge_with_its_decayed_weight() -> None:
    agents = create_agents()
    add_undirected_edge(agents, "doc-1", "doc-2", 0.04)

    summary = update_graph_after_retrieval(
        agents,
        ["doc-3"],
        GraphUpdateConfig(
            decay_rate=0.5,
            prune_threshold=0.03,
            enable_pruning=True,
        ),
    )

    assert summary["strengthened_edges"] == 0
    assert summary["created_edges"] == 0
    assert summary["pruned_edges"] == 1
    assert summary["edge_updates"] == [
        {
            "source": "doc-1",
            "target": "doc-2",
            "previous_weight": 0.02,
            "new_weight": None,
            "change_type": "pruned",
        }
    ]
    assert "doc-2" not in agents["doc-1"].edges


def test_disabling_pruning_returns_no_pruned_updates() -> None:
    agents = create_agents()
    add_undirected_edge(agents, "doc-1", "doc-2", 0.01)

    summary = update_graph_after_retrieval(
        agents,
        ["doc-3"],
        GraphUpdateConfig(prune_threshold=0.05, enable_pruning=False),
    )

    assert summary["pruned_edges"] == 0
    assert summary["edge_updates"] == []
    assert "doc-2" in agents["doc-1"].edges


@pytest.mark.parametrize(
    ("update_k", "expected_edges"),
    [
        (3, 3),
        (5, 10),
        (10, 45),
        (None, 45),
    ],
)
def test_update_k_limits_strengthening(
    update_k: int | None,
    expected_edges: int,
) -> None:
    agents = {
        f"doc-{position}": DocumentAgent(
            doc_id=f"doc-{position}",
            title=f"doc-{position}",
            text="",
        )
        for position in range(1, 11)
    }
    retrieved_doc_ids = list(agents)

    summary = update_graph_after_retrieval(
        agents,
        retrieved_doc_ids,
        GraphUpdateConfig(update_k=update_k, enable_pruning=False),
    )

    assert summary["retrieved_documents"] == 10
    assert summary["updated_documents"] == (update_k or 10)
    assert summary["strengthened_edges"] == expected_edges
    assert summary["created_edges"] == expected_edges


def test_update_k_must_include_at_least_two_documents() -> None:
    with pytest.raises(ValueError, match="update_k must be at least 2"):
        GraphUpdateConfig(update_k=1)
