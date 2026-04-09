from enum import Enum
from pydantic import BaseModel
from src.schemas import ProductInput

class JudgeInput(ProductInput):
  generated_description: str

class RubricVerdict(str, Enum):
  GOOD = "GOOD"
  OK = "OK"
  BAD = "BAD"

class FinalScoreVerdict(str, Enum):
  PASS = "PASS"
  FAIL = "FAIL"

class JudgeResult(BaseModel):
  fluency_explanation: str
  fluency_verdict: RubricVerdict
  grammar_explanation: str
  grammar_verdict: RubricVerdict
  tone_explanation: str
  tone_verdict: RubricVerdict
  length_explanation: str
  length_verdict: RubricVerdict
  grounding_explanation: str
  grounding_verdict: RubricVerdict
  final_score_explanation: str
  final_score_verdict: FinalScoreVerdict

class CriterionJudgeResult(BaseModel):
    explanation: str
    verdict: RubricVerdict