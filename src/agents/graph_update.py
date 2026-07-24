from dataclasses import dataclass

from agents.document_agent import DocumentAgent


@dataclass
class GraphUpdateConfig:
    strengthen_amount: float = 0.05
    decay_rate: float = 0.995
    prune_threshold: float = 0.05
    max_edge_weight: float = 1.0


def make_edge_key(source_id: str, target_id: str) -> tuple[str, str]:
    return tuple(sorted((source_id, target_id)))


def strengthen_edge(
    agents: dict[str, DocumentAgent],
    source_id: str,
    target_id: str,
    config: GraphUpdateConfig,
) -> None:
    current_edge = agents[source_id].edges.get(target_id)
    current_weight = current_edge.weight if current_edge is not None else 0.0
    strengthened_weight = min(
        current_weight + config.strengthen_amount,
        config.max_edge_weight,
    )

    agents[source_id].add_edge(target_id, strengthened_weight)
    agents[target_id].add_edge(source_id, strengthened_weight)


def strengthen_retrieved_edges(
    agents: dict[str, DocumentAgent],
    retrieved_doc_ids: list[str],
    config: GraphUpdateConfig,
) -> set[tuple[str, str]]:
    unique_doc_ids = list(dict.fromkeys(retrieved_doc_ids))
    strengthened_edges = set()

    for source_position, source_id in enumerate(unique_doc_ids):
        for target_id in unique_doc_ids[source_position + 1:]:
            strengthen_edge(
                agents=agents,
                source_id=source_id,
                target_id=target_id,
                config=config,
            )
            strengthened_edges.add(make_edge_key(source_id, target_id))

    return strengthened_edges


def decay_unused_edges(
    agents: dict[str, DocumentAgent],
    strengthened_edges: set[tuple[str, str]],
    config: GraphUpdateConfig,
) -> None:
    decayed_edges = set()

    for source_id, agent in agents.items():
        for target_id, edge in agent.edges.items():
            edge_key = make_edge_key(source_id, target_id)

            if edge_key in strengthened_edges or edge_key in decayed_edges:
                continue

            decayed_weight = edge.weight * config.decay_rate

            agents[source_id].add_edge(target_id, decayed_weight)
            agents[target_id].add_edge(source_id, decayed_weight)

            decayed_edges.add(edge_key)


def prune_weak_edges(
    agents: dict[str, DocumentAgent],
    config: GraphUpdateConfig,
) -> int:
    checked_edges = set()
    edges_to_remove = []

    for source_id, agent in agents.items():
        for target_id, edge in agent.edges.items():
            edge_key = make_edge_key(source_id, target_id)

            if edge_key in checked_edges:
                continue

            checked_edges.add(edge_key)

            if edge.weight < config.prune_threshold:
                edges_to_remove.append((source_id, target_id))

    for source_id, target_id in edges_to_remove:
        agents[source_id].edges.pop(target_id, None)
        agents[target_id].edges.pop(source_id, None)

    return len(edges_to_remove)


def update_graph_after_retrieval(
    agents: dict[str, DocumentAgent],
    retrieved_doc_ids: list[str],
    config: GraphUpdateConfig | None = None,
) -> dict:
    if config is None:
        config = GraphUpdateConfig()

    strengthened_edges = strengthen_retrieved_edges(
        agents=agents,
        retrieved_doc_ids=retrieved_doc_ids,
        config=config,
    )
    decay_unused_edges(
        agents=agents,
        strengthened_edges=strengthened_edges,
        config=config,
    )
    pruned_edges = prune_weak_edges(
        agents=agents,
        config=config,
    )

    return {
        "retrieved_documents": len(retrieved_doc_ids),
        "strengthened_edges": len(strengthened_edges),
        "pruned_edges": pruned_edges,
    }
