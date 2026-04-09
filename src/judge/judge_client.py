from openai import OpenAI
from src.config import get_settings
from src.judge.judge_schemas import JudgeInput, JudgeResult, CriterionJudgeResult
from src.judge.judge_prompts import build_judge_system_prompt, build_judge_user_prompt

def generate_judge_result(
    product_llm_result: JudgeInput, 
    system_prompt: str | None = None,
    llm_model: str | None = None,
    temperature: float | None = None,
    is_isolated_criterion_experiment: bool = False,
) -> JudgeResult | CriterionJudgeResult:
    settings = get_settings()
    client = OpenAI(
        base_url=settings.nebius_api_endpoint, 
        api_key=settings.nebius_api_key
    )

    final_temperature = 0.1 if temperature is None else temperature

    completion = client.chat.completions.create(
        model=llm_model or settings.judge_llm_model,
        temperature=final_temperature,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "judge_result",
                "schema": CriterionJudgeResult.model_json_schema() if is_isolated_criterion_experiment else JudgeResult.model_json_schema(),
            },
        },
        messages=[
            {
                "role": "system",
                "content": system_prompt or build_judge_system_prompt()
            },
            {
                "role": "user",
                "content": build_judge_user_prompt(product_llm_result)
            }
        ]
    )

    content = completion.choices[0].message.content
    return CriterionJudgeResult.model_validate_json(content) if is_isolated_criterion_experiment else JudgeResult.model_validate_json(content)