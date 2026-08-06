import hashlib
import shutil
from pathlib import Path

from agents import DocumentAgent, save_agents
from graph.initialize_agent_graph import build_initial_agent_graph


NEIGHBOURS_PER_DOCUMENT = 5
INITIAL_WEIGHT_SCALE = 0.2


def file_checksum(path: Path) -> str:
    checksum = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            checksum.update(chunk)

    return checksum.hexdigest()


def create_initial_evaluation_graph(path: Path) -> dict[str, DocumentAgent]:
    agents = build_initial_agent_graph(
        neighbours_per_doc=NEIGHBOURS_PER_DOCUMENT,
        initial_weight_scale=INITIAL_WEIGHT_SCALE,
    )
    save_agents(path, agents)

    return agents


def copy_agent_graph(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target_path)


def graph_file_is_unchanged(path: Path, original_checksum: str) -> bool:
    return file_checksum(path) == original_checksum
