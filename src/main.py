from pathlib import Path
import argparse

from src.pipeline import run_generation_pipeline
from src.prompts import build_system_prompt, build_stricter_grounding_system_prompt


EXPERIMENT_PROMPTS = {
    "grounding_v2": build_stricter_grounding_system_prompt,
    "low_temperature": build_stricter_grounding_system_prompt,
    "faster_model": build_system_prompt,
}


def resolve_system_prompt(experiment: str | None) -> str:
    if experiment is None:
        return build_system_prompt()

    if experiment not in EXPERIMENT_PROMPTS:
        raise ValueError(f"Unknown experiment: {experiment}")

    return EXPERIMENT_PROMPTS[experiment]()


def resolve_output_path(project_root: Path, experiment: str | None) -> Path:
    if experiment is None:
        return project_root / "data" / "assignment_01.xlsx"

    experiments_dir = project_root / "data" / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    return experiments_dir / f"{experiment}.xlsx"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run product description generation pipeline.")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--take-rows", type=int, default=None)
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    input_csv = project_root / "data" / "product_dataset.csv"

    system_prompt = resolve_system_prompt(args.experiment)
    output_xlsx = resolve_output_path(project_root, args.experiment)

    result_df = run_generation_pipeline(
        input_csv=input_csv,
        output_xlsx=output_xlsx,
        system_prompt=system_prompt,
        take_rows=args.take_rows,
        llm_model=args.llm_model,
        temperature=args.temperature,
    )

    print(f"Finished generating descriptions for {len(result_df)} rows.")
    print(f"Output saved to: {output_xlsx}")


if __name__ == "__main__":
    main()