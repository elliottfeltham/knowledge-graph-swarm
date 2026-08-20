import argparse
import subprocess
import sys

from common import PROJECT_ROOT


def build_evaluation_commands(skip_preparation: bool) -> list[list[str]]:
    commands = []

    if not skip_preparation:
        commands.append([sys.executable, str(PROJECT_ROOT / "src" / "prepare_system.py")])

    commands.extend(
        [
            [sys.executable, str(PROJECT_ROOT / "src" / "run_experiments.py")],
            [
                sys.executable,
                str(PROJECT_ROOT / "src" / "run_experiments.py"),
                "--pruning-ablation",
            ],
            [
                sys.executable,
                str(
                    PROJECT_ROOT
                    / "src"
                    / "evaluation"
                    / "evaluate_adaptation_over_time.py"
                ),
            ],
            [
                sys.executable,
                str(
                    PROJECT_ROOT
                    / "src"
                    / "evaluation"
                    / "evaluate_repeated_adaptation.py"
                ),
            ],
            [
                sys.executable,
                str(PROJECT_ROOT / "src" / "evaluation" / "create_results_report.py"),
            ],
        ]
    )

    return commands


def run_full_evaluation(skip_preparation: bool = False) -> None:
    commands = build_evaluation_commands(skip_preparation)
    stage_names = [
        "Running the main evaluation",
        "Running the pruning ablation",
        "Evaluating adaptation over time",
        "Evaluating repeated context",
        "Creating tables and figures",
    ]

    if not skip_preparation:
        stage_names.insert(0, "Preparing the system")

    for stage_number, (stage_name, command) in enumerate(
        zip(stage_names, commands), start=1
    ):
        print(f"\n[{stage_number}/{len(commands)}] {stage_name}", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    print("\nFull evaluation complete.")
    print(f"Results and report files: {PROJECT_ROOT / 'results'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the system and reproduce the full dissertation evaluation."
    )
    parser.add_argument(
        "--skip-preparation",
        action="store_true",
        help="Use the existing dataset, embeddings and graphs.",
    )
    args = parser.parse_args()

    run_full_evaluation(skip_preparation=args.skip_preparation)


if __name__ == "__main__":
    main()
