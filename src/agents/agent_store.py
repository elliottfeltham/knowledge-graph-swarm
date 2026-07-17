import json
from pathlib import Path

from agents.document_agent import DocumentAgent


def save_agents(path: Path, agents: dict[str, DocumentAgent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        for agent in agents.values():
            file.write(json.dumps(agent.to_dict(), ensure_ascii=False) + "\n")


def load_agents(path: Path) -> dict[str, DocumentAgent]:
    agents = {}

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            agent = DocumentAgent.from_dict(json.loads(line))
            agents[agent.doc_id] = agent

    return agents
