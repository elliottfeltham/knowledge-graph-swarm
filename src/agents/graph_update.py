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
