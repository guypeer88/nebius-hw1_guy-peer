from pathlib import Path
import pandas as pd

from src.judge.judge_client import generate_judge_result
from src.judge.judge_schemas import JudgeInput, JudgeResult, CriterionJudgeResult


JUDGE_OUTPUT_COLUMNS = [
    "fluency_explanation_judge",
    "fluency_verdict_judge",
    "grammar_explanation_judge",
    "grammar_verdict_judge",
    "tone_explanation_judge",
    "tone_verdict_judge",
    "length_explanation_judge",
    "length_verdict_judge",
    "grounding_explanation_judge",
    "grounding_verdict_judge",
    "final_score_explanation_judge",
    "final_score_verdict_judge",
    "judge_status",
    "judge_error_message",
]

JUDGE_ISOLATED_CRITERION_EXPERIMENTS = {
    "fluency_only": "fluency",
    "grammar_only": "grammar",
    "tone_only": "tone",
    "length_only": "length",
    "grounding_only": "grounding",
}


def resolve_focus_criterion(experiment: str | None) -> str | None:
    if experiment is None:
        return None
    return JUDGE_ISOLATED_CRITERION_EXPERIMENTS.get(experiment)


def load_and_validate_judge_dataset(input_csv: str | Path) -> pd.DataFrame:
    input_csv = Path(input_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV file not found: {input_csv}")

    df = pd.read_csv(input_csv)

    if df.empty:
        raise ValueError("The input dataset is empty.")

    df.columns = df.columns.str.strip()

    required_columns = list(JudgeInput.model_fields.keys())
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df


def build_judge_input(row: dict) -> JudgeInput:
    return JudgeInput.model_validate(row)


def build_full_judge_result_row(judge_result: JudgeResult) -> dict:
    return {
        "fluency_explanation_judge": judge_result.fluency_explanation,
        "fluency_verdict_judge": judge_result.fluency_verdict.value,
        "grammar_explanation_judge": judge_result.grammar_explanation,
        "grammar_verdict_judge": judge_result.grammar_verdict.value,
        "tone_explanation_judge": judge_result.tone_explanation,
        "tone_verdict_judge": judge_result.tone_verdict.value,
        "length_explanation_judge": judge_result.length_explanation,
        "length_verdict_judge": judge_result.length_verdict.value,
        "grounding_explanation_judge": judge_result.grounding_explanation,
        "grounding_verdict_judge": judge_result.grounding_verdict.value,
        "final_score_explanation_judge": judge_result.final_score_explanation,
        "final_score_verdict_judge": judge_result.final_score_verdict.value,
        "judge_status": "success",
        "judge_error_message": "",
    }


def build_single_criterion_judge_result_row(
    focus_criterion: str,
    judge_result: CriterionJudgeResult,
) -> dict:
    return {
        f"{focus_criterion}_explanation_judge": judge_result.explanation,
        f"{focus_criterion}_verdict_judge": judge_result.verdict.value,
        "judge_status": "success",
        "judge_error_message": "",
    }


def build_judge_error_row(error_message: str) -> dict:
    return {
        "judge_status": "error",
        "judge_error_message": error_message,
    }


def ensure_judge_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in JUDGE_OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df


def run_judge_pipeline(
    input_csv: str | Path,
    output_csv: str | Path,
    system_prompt: str | None = None,
    take_rows: int | None = None,
    llm_model: str | None = None,
    temperature: float | None = None,
    experiment: str | None = None,
) -> pd.DataFrame:
    df = load_and_validate_judge_dataset(input_csv)

    if take_rows is not None:
        df = df.head(take_rows).copy()

    focus_criterion = resolve_focus_criterion(experiment)
    is_isolated_criterion_experiment = focus_criterion is not None

    judge_results: list[dict] = []

    for _, row in df.iterrows():
        row_dict = row.to_dict()

        try:
            judge_input = build_judge_input(row_dict)
            judge_result = generate_judge_result(
                product_llm_result=judge_input,
                system_prompt=system_prompt,
                llm_model=llm_model,
                temperature=temperature,
                is_isolated_criterion_experiment=is_isolated_criterion_experiment,
            )

            if is_isolated_criterion_experiment:
                judge_row = build_single_criterion_judge_result_row(
                    focus_criterion=focus_criterion,
                    judge_result=judge_result,
                )
            else:
                judge_row = build_full_judge_result_row(judge_result)

        except Exception as exc:
            judge_row = build_judge_error_row(str(exc))

        judge_results.append(judge_row)

    judge_df = pd.DataFrame(judge_results)
    result_df = pd.concat([df.reset_index(drop=True), judge_df.reset_index(drop=True)], axis=1)
    result_df = ensure_judge_output_columns(result_df)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(output_csv, index=False)

    return result_df