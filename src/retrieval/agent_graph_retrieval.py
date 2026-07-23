from pathlib import Path

from agents import DocumentAgent, load_agents
from common import PROJECT_ROOT


AGENT_GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "document_agents.jsonl"


def load_agent_graph(path: Path = AGENT_GRAPH_PATH) -> dict[str, DocumentAgent]:
    return load_agents(path)
