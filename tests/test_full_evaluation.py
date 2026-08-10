import subprocess

from run_full_evaluation import run_full_evaluation


def test_full_evaluation_runs_each_stage_in_order(monkeypatch) -> None:
    commands = []

    def record_command(command, cwd, check) -> None:
        commands.append(command)
        assert check is True

    monkeypatch.setattr(subprocess, "run", record_command)

    run_full_evaluation(skip_preparation=False)

    assert commands[0][1].endswith("prepare_system.py")
    assert commands[1][1].endswith("run_experiments.py")
    assert commands[2][-1] == "--pruning-ablation"
    assert commands[3][1].endswith("create_results_report.py")


def test_full_evaluation_can_skip_preparation(monkeypatch) -> None:
    commands = []
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, cwd, check: commands.append(command),
    )

    run_full_evaluation(skip_preparation=True)

    assert len(commands) == 3
    assert all(not command[1].endswith("prepare_system.py") for command in commands)
