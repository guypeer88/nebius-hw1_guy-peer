import time
from openai import OpenAI
from src.config import get_settings
from src.schemas import GenerationResult, ProductInput
from src.prompts import build_system_prompt, build_user_prompt

def generate_product_description(
    product: ProductInput, 
    system_prompt: str | None = None,
    llm_model: str | None = None,
    temperature: float | None = None,
) -> GenerationResult:
    settings = get_settings()
    client = OpenAI(
        base_url=settings.nebius_api_endpoint, 
        api_key=settings.nebius_api_key
    )

    start_time = time.perf_counter()

    final_temperature = 0.7 if temperature is None else temperature

    completion = client.chat.completions.create(
        model=llm_model or settings.llm_model,
        temperature=final_temperature,
        messages=[
            {
                "role": "system",
                "content": system_prompt or build_system_prompt()
            },
            {
                "role": "user",
                "content": build_user_prompt(product)
            }
        ]
    )

    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000
    input_tokens = completion.usage.prompt_tokens
    output_tokens = completion.usage.completion_tokens
    generated_product_description = completion.choices[0].message.content.strip()

    return GenerationResult(
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        generated_description=generated_product_description,
    )