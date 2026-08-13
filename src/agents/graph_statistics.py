from agents.document_agent import DocumentAgent
from agents.graph_update import make_edge_key
from common import average


def build_edge_snapshot(agents: dict[str, DocumentAgent]) -> dict[tuple[str, str], float]:
    edge_snapshot = {}

    for source_id, agent in agents.items():
        for target_id, edge in agent.edges.items():
            edge_key = make_edge_key(source_id, target_id)

            if edge_key in edge_snapshot:
                continue

            edge_snapshot[edge_key] = edge.weight

    return edge_snapshot


def calculate_graph_statistics(agents: dict[str, DocumentAgent]) -> dict:
    edge_snapshot = build_edge_snapshot(agents)
    edge_weights = list(edge_snapshot.values())
    agent_count = len(agents)
    edge_count = len(edge_snapshot)

    return {
        "agents": agent_count,
        "undirected_edges": edge_count,
        "average_degree": (2 * edge_count) / agent_count if agent_count else 0.0,
        "average_edge_weight": average(edge_weights),
        "min_edge_weight": min(edge_weights) if edge_weights else 0.0,
        "max_edge_weight": max(edge_weights) if edge_weights else 0.0,
    }

def calculate_graph_deltas(before_stats: dict, after_stats: dict) -> dict:
    return {
        "agents": after_stats["agents"] - before_stats["agents"],
        "undirected_edges": after_stats["undirected_edges"] - before_stats["undirected_edges"],
        "average_degree": after_stats["average_degree"] - before_stats["average_degree"],
        "average_edge_weight": after_stats["average_edge_weight"] - before_stats["average_edge_weight"],
        "min_edge_weight": after_stats["min_edge_weight"] - before_stats["min_edge_weight"],
        "max_edge_weight": after_stats["max_edge_weight"] - before_stats["max_edge_weight"],
    }
