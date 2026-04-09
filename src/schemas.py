from pydantic import BaseModel

class ProductInput(BaseModel):
  name: str
  attributes: str
  material: str
  warranty: str

class GenerationResult(BaseModel):
  generated_description: str
  latency_ms: float
  input_tokens: int
  output_tokens: int