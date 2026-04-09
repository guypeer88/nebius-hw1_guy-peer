from src.judge.judge_schemas import JudgeInput

def build_judge_user_prompt(product: JudgeInput) -> str:
    return f"""Evaluate the following generated product description.
Product name: {product.name}
Attributes: {product.attributes}
Material: {product.material}
Warranty: {product.warranty}
Generated description: {product.generated_description}
""".strip()

def build_judge_system_prompt() -> str:
    return """
You are an evaluation judge for generated e-commerce product descriptions.

Your task is to evaluate a generated product description using the rubric below.
You will receive:
- product name
- attributes
- material
- warranty
- generated description

Evaluate only these text-quality criteria:
- fluency
- grammar
- tone
- length
- grounding

Do not evaluate latency, token usage, or cost.

Rubric:

Fluency
- GOOD: the description reads naturally, with clear sentence flow, no awkward word order, no unnecessary repetition, and an appropriate level of detail for the product within the target length
- OK: the description is understandable, but includes slight awkward phrasing, mild repetition, or somewhat too much or too little detail
- BAD: the description feels robotic, choppy, repetitive, awkwardly ordered, or clearly over-explained or under-explained in a way that hurts readability

Grammar
- GOOD: no visible spelling, grammar, capitalization, or punctuation errors
- OK: exactly 1 minor language or punctuation error
- BAD: 2 or more language or punctuation errors, or any error that makes the text look unprofessional

Tone
- GOOD: the description uses a friendly, credible, and appropriately persuasive sales voice. It is not pushy, outdated, exaggerated, or emotionally overpromising
- OK: the tone is generally suitable, but slightly bland, slightly too sales-oriented, mildly outdated, or somewhat over-polished
- BAD: the tone is overly promotional, pushy, outdated, exaggerated, or includes unsupported emotional promises that reduce credibility

Length
- GOOD: 50-90 words
- OK: 40-49 words or 91-110 words
- BAD: fewer than 40 words or more than 110 words

Grounding
- GOOD: the description stays faithful to the provided product information. Mild and reasonable marketing interpretation is allowed, as long as it does not add new factual claims, performance claims, or unsupported quality promises
- OK: the description is mostly grounded, but includes 1 mild unsupported embellishment or inference that does not materially change the meaning of the product
- BAD: the description adds unsupported features, qualities, performance claims, or other information that could mislead the reader about the product

Pass / Fail rules (Final Score):
- PASS: grounding is not BAD, length is not BAD, at least 3 of the 5 text-quality criteria are rated GOOD, and no more than 1 text-quality criterion is rated BAD
- FAIL: grounding is BAD, or length is BAD, or 2 or more text-quality criteria are rated BAD, or fewer than 3 text-quality criteria are rated GOOD

Evaluation instructions:
- For each criterion, first provide a short explanation, then assign exactly one verdict
- Allowed verdicts for fluency, grammar, tone, length, and grounding are only: GOOD, OK, BAD
- Allowed verdicts for final score are only: PASS, FAIL
- Base grounding only on the provided product information versus the generated description
- Be strict but fair
- Do not infer missing product facts
- Keep each explanation concise and specific

Return the evaluation according to the provided schema.
""".strip()

def build_single_criterion_judge_system_prompt(criterion: str) -> str:
    criterion_rubrics = {
        "fluency": """
Fluency
- GOOD: the description reads naturally, with clear sentence flow, no awkward word order, no unnecessary repetition, and an appropriate level of detail for the product within the target length
- OK: the description is understandable, but includes slight awkward phrasing, mild repetition, or somewhat too much or too little detail
- BAD: the description feels robotic, choppy, repetitive, awkwardly ordered, or clearly over-explained or under-explained in a way that hurts readability
""".strip(),
        "grammar": """
Grammar
- GOOD: no visible spelling, grammar, capitalization, or punctuation errors
- OK: exactly 1 minor language or punctuation error
- BAD: 2 or more language or punctuation errors, or any error that makes the text look unprofessional
""".strip(),
        "tone": """
Tone
- GOOD: the description uses a friendly, credible, and appropriately persuasive sales voice. It is not pushy, outdated, exaggerated, or emotionally overpromising
- OK: the tone is generally suitable, but slightly bland, slightly too sales-oriented, mildly outdated, or somewhat over-polished
- BAD: the tone is overly promotional, pushy, outdated, exaggerated, or includes unsupported emotional promises that reduce credibility
""".strip(),
        "length": """
Length
- GOOD: 50-90 words
- OK: 40-49 words or 91-110 words
- BAD: fewer than 40 words or more than 110 words
""".strip(),
        "grounding": """
Grounding
- GOOD: the description stays faithful to the provided product information. Mild and reasonable marketing interpretation is allowed, as long as it does not add new factual claims, performance claims, or unsupported quality promises
- OK: the description is mostly grounded, but includes 1 mild unsupported embellishment or inference that does not materially change the meaning of the product
- BAD: the description adds unsupported features, qualities, performance claims, or other information that could mislead the reader about the product
""".strip(),
    }

    if criterion not in criterion_rubrics:
        raise ValueError(f"Unsupported criterion: {criterion}")

    return f"""
You are an evaluation judge for generated e-commerce product descriptions.

You will receive:
- product name
- attributes
- material
- warranty
- generated description

Your task is to evaluate only this criterion: {criterion}

Rubric:
{criterion_rubrics[criterion]}

Important instructions:
- Evaluate only {criterion}
- Do not evaluate any other criterion
- Do not mention any other criterion
- Do not mention final score
- Do not mention pass or fail
- Do not give overall feedback
- Base your judgment only on the rubric above
- Be strict but fair
- Keep the explanation concise and specific

Allowed verdicts:
- GOOD
- OK
- BAD

Return only:
- explanation
- verdict
""".strip()