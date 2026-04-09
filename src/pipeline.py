import pandas as pd
from pathlib import Path
from src.llm_client import generate_product_description

REQUIRED_COLUMNS = ["name", "attributes", "material", "warranty"]

RUBRIC_COLUMNS = [
    "fluency",
    "grammar",
    "tone",
    "length",
    "grounding",
    "latency",
    "cost",
    "final_score",
]

def load_and_validate_dataset(input_csv: str | Path) -> pd.DataFrame:
    input_csv = Path(input_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV file not found: {input_csv}")

    df = pd.read_csv(input_csv)

    if df.empty:
        raise ValueError("The input dataset is empty.")

    df.columns = df.columns.str.strip()

    df = df.rename(
        columns={
            "product_name": "name",
            "Product_attribute_list": "attributes",
        }
    )

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df


def build_result_row(product: dict, generation_result) -> dict:
    return {
        **product,
        "generated_description": generation_result.generated_description,
        "latency_ms": generation_result.latency_ms,
        "input_tokens": generation_result.input_tokens,
        "output_tokens": generation_result.output_tokens,
        "fluency": "",
        "grammar": "",
        "tone": "",
        "length": "",
        "grounding": "",
        "latency": "",
        "cost": "",
        "final_score": "",
    }


def build_error_row(product: dict, error_message: str) -> dict:
    return {
        **product,
        "generated_description": "",
        "latency_ms": None,
        "input_tokens": None,
        "output_tokens": None,
        "fluency": "",
        "grammar": "",
        "tone": "",
        "length": "",
        "grounding": "",
        "latency": "",
        "cost": "",
        "final_score": "",
        "status": "error",
        "error_message": error_message,
    }


def ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    required_output_columns = [
        "generated_description",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        *RUBRIC_COLUMNS,
    ]

    for col in required_output_columns:
        if col not in df.columns:
            df[col] = ""

    return df


def run_generation_pipeline(
    input_csv: str | Path,
    output_xlsx: str | Path,
    system_prompt: str | None = None,
    take_rows: int | None = None,
    llm_model: str | None = None,
    temperature: float | None = None,
) -> pd.DataFrame:
    df = load_and_validate_dataset(input_csv)

    if take_rows is not None:
        df = df.head(take_rows).copy()

    results: list[dict] = []

    for _, row in df.iterrows():
        product = row.to_dict()

        try:
            generation_result = generate_product_description(
                product=product, 
                system_prompt=system_prompt,
                llm_model=llm_model,
                temperature=temperature,
            )
            result_row = build_result_row(product, generation_result)
            result_row["status"] = "success"
            result_row["error_message"] = ""
        except Exception as exc:
            result_row = build_error_row(product, str(exc))

        results.append(result_row)

    result_df = pd.DataFrame(results)
    result_df = ensure_output_columns(result_df)

    output_xlsx = Path(output_xlsx)
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)

    result_df.to_excel(output_xlsx, index=False)

    return result_df