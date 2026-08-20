from dataclasses import dataclass

from agents.document_agent import DocumentAgent


@dataclass
class GraphUpdateConfig:
    strengthen_amount: float = 0.01
    decay_rate: float = 0.995
    prune_threshold: float = 0.05
    max_edge_weight: float = 1.0
    enable_pruning: bool = False
    update_k: int | None = 3

    def __post_init__(self) -> None:
        if self.update_k is not None and self.update_k < 2:
            raise ValueError("update_k must be at least 2")


def make_edge_key(source_id: str, target_id: str) -> tuple[str, str]:
    return tuple(sorted((source_id, target_id)))


def strengthen_edge(
    agents: dict[str, DocumentAgent],
    source_id: str,
    target_id: str,
    config: GraphUpdateConfig,
) -> dict:
    current_edge = agents[source_id].edges.get(target_id)
    current_weight = current_edge.weight if current_edge is not None else 0.0
    strengthened_weight = min(
        current_weight + config.strengthen_amount,
        config.max_edge_weight,
    )

    agents[source_id].add_edge(target_id, strengthened_weight)
    agents[target_id].add_edge(source_id, strengthened_weight)

    return {
        "source": source_id,
        "target": target_id,
        "previous_weight": current_weight,
        "new_weight": strengthened_weight,
        "change_type": "strengthened" if current_edge is not None else "created",
    }


def strengthen_retrieved_edges(
    agents: dict[str, DocumentAgent],
    updated_doc_ids: list[str],
    config: GraphUpdateConfig,
) -> list[dict]:
    edge_updates = []

    for source_position, source_id in enumerate(updated_doc_ids):
        for target_id in updated_doc_ids[source_position + 1:]:
            edge_updates.append(
                strengthen_edge(
                    agents=agents,
                    source_id=source_id,
                    target_id=target_id,
                    config=config,
                )
            )

    return edge_updates


def decay_local_edges(
    agents: dict[str, DocumentAgent],
    updated_doc_ids: list[str],
    strengthened_edges: set[tuple[str, str]],
    config: GraphUpdateConfig,
) -> list[dict]:
    updated_doc_id_set = set(updated_doc_ids)
    decayed_edge_keys = set()
    decayed_updates = []

    for source_id in updated_doc_ids:
        for target_id, edge in list(agents[source_id].edges.items()):
            edge_key = make_edge_key(source_id, target_id)

            if (
                target_id in updated_doc_id_set
                or edge_key in strengthened_edges
                or edge_key in decayed_edge_keys
            ):
                continue

            previous_weight = edge.weight
            decayed_weight = previous_weight * config.decay_rate

            if decayed_weight == previous_weight:
                continue

            agents[source_id].add_edge(target_id, decayed_weight)
            agents[target_id].add_edge(source_id, decayed_weight)

            decayed_edge_keys.add(edge_key)
            decayed_updates.append(
                {
                    "source": edge_key[0],
                    "target": edge_key[1],
                    "previous_weight": previous_weight,
                    "new_weight": decayed_weight,
                    "change_type": "decayed",
                }
            )

    return decayed_updates


def prune_weak_edges(
    agents: dict[str, DocumentAgent],
    candidate_edges: set[tuple[str, str]],
    config: GraphUpdateConfig,
) -> list[dict]:
    edges_to_remove = []

    for source_id, target_id in sorted(candidate_edges):
        edge = agents[source_id].edges.get(target_id)

        if edge is not None and edge.weight < config.prune_threshold:
            edges_to_remove.append((source_id, target_id, edge.weight))

    for source_id, target_id, _ in edges_to_remove:
        agents[source_id].edges.pop(target_id, None)
        agents[target_id].edges.pop(source_id, None)

    return [
        {
            "source": source_id,
            "target": target_id,
            "previous_weight": weight,
            "new_weight": None,
            "change_type": "pruned",
        }
        for source_id, target_id, weight in edges_to_remove
    ]


def update_graph_after_retrieval(
    agents: dict[str, DocumentAgent],
    retrieved_doc_ids: list[str],
    config: GraphUpdateConfig | None = None,
) -> dict:
    if config is None:
        config = GraphUpdateConfig()

    updated_doc_ids = list(dict.fromkeys(retrieved_doc_ids))

    if config.update_k is not None:
        updated_doc_ids = updated_doc_ids[:config.update_k]

    strengthened_updates = strengthen_retrieved_edges(
        agents=agents,
        updated_doc_ids=updated_doc_ids,
        config=config,
    )
    strengthened_edges = {
        make_edge_key(update["source"], update["target"])
        for update in strengthened_updates
    }
    decayed_updates = decay_local_edges(
        agents=agents,
        updated_doc_ids=updated_doc_ids,
        strengthened_edges=strengthened_edges,
        config=config,
    )
    decayed_edges = {
        make_edge_key(update["source"], update["target"])
        for update in decayed_updates
    }
    pruned_updates = []

    if config.enable_pruning:
        pruned_updates = prune_weak_edges(
            agents=agents,
            candidate_edges=decayed_edges,
            config=config,
        )

    created_edges = sum(
        update["change_type"] == "created"
        for update in strengthened_updates
    )
    pruned_edges = {
        make_edge_key(update["source"], update["target"])
        for update in pruned_updates
    }
    visible_decayed_updates = [
        update
        for update in decayed_updates
        if make_edge_key(update["source"], update["target"]) not in pruned_edges
    ]
    return {
        "retrieved_documents": len(retrieved_doc_ids),
        "updated_documents": len(updated_doc_ids),
        "strengthened_edges": len(strengthened_edges),
        "created_edges": created_edges,
        "decayed_edges": len(decayed_edges),
        "pruned_edges": len(pruned_updates),
        "edge_updates": strengthened_updates + visible_decayed_updates + pruned_updates,
    }
