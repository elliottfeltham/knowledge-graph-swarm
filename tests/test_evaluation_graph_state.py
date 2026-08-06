from agents import DocumentAgent, load_agents
import evaluation.graph_state as graph_state


def test_create_initial_evaluation_graph_saves_a_fresh_graph(
    monkeypatch,
    tmp_path,
) -> None:
    agents = {
        "doc-1": DocumentAgent(doc_id="doc-1", title="Document 1", text=""),
    }
    output_path = tmp_path / "initial_graph.jsonl"

    monkeypatch.setattr(
        graph_state,
        "build_initial_agent_graph",
        lambda neighbours_per_doc, initial_weight_scale: agents,
    )

    created_agents = graph_state.create_initial_evaluation_graph(output_path)

    assert created_agents is agents
    assert list(load_agents(output_path)) == ["doc-1"]


def test_copied_graph_can_change_without_modifying_its_source(tmp_path) -> None:
    source_path = tmp_path / "initial_graph.jsonl"
    copied_path = tmp_path / "working_graph.jsonl"
    source_path.write_text('{"doc_id": "doc-1"}\n', encoding="utf-8")
    source_checksum = graph_state.file_checksum(source_path)

    graph_state.copy_agent_graph(source_path, copied_path)
    copied_path.write_text('{"doc_id": "updated"}\n', encoding="utf-8")

    assert graph_state.graph_file_is_unchanged(source_path, source_checksum)
    assert not graph_state.graph_file_is_unchanged(copied_path, source_checksum)
