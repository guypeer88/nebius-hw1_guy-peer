from pathlib import Path
import argparse

from src.judge.judge_pipeline import run_judge_pipeline
from src.judge.judge_prompts import build_judge_system_prompt, build_single_criterion_judge_system_prompt

EXPERIMENT_PROMPTS = {
    "fluency_only": lambda: build_single_criterion_judge_system_prompt("fluency"),
    "grammar_only": lambda: build_single_criterion_judge_system_prompt("grammar"),
    "tone_only": lambda: build_single_criterion_judge_system_prompt("tone"),
    "length_only": lambda: build_single_criterion_judge_system_prompt("length"),
    "grounding_only": lambda: build_single_criterion_judge_system_prompt("grounding"),
}


def resolve_system_prompt(experiment: str | None) -> str:
    if experiment is None or experiment not in EXPERIMENT_PROMPTS:
        return build_judge_system_prompt()
    return EXPERIMENT_PROMPTS[experiment]()

def resolve_input_path(project_root: Path) -> Path:
    return project_root / "data" / "assignment_01.csv"


def resolve_output_path(project_root: Path, experiment: str | None) -> Path:
    judge_dir = project_root / "data" / "judge_results"
    judge_dir.mkdir(parents=True, exist_ok=True)

    if experiment is None:
        return judge_dir / "assignment_01_judged.csv"

    if experiment in EXPERIMENT_PROMPTS:
        judge_experiments_dir = judge_dir / "criterions-isolated"
        judge_experiments_dir.mkdir(parents=True, exist_ok=True)
        return judge_experiments_dir / f"{experiment}_judged.csv"

    return judge_dir / f"{experiment}_judged.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run judge pipeline.")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--take-rows", type=int, default=None)
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    input_csv = resolve_input_path(project_root)
    output_csv = resolve_output_path(project_root, args.experiment)
    system_prompt = resolve_system_prompt(args.experiment)

    result_df = run_judge_pipeline(
        input_csv=input_csv,
        output_csv=output_csv,
        system_prompt=system_prompt,
        take_rows=args.take_rows,
        llm_model=args.llm_model,
        temperature=args.temperature,
        experiment=args.experiment,
    )

    print(f"Finished judging {len(result_df)} rows.")
    print(f"Output saved to: {output_csv}")


if __name__ == "__main__":
    main()