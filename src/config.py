import os
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseModel):
    llm_model: str
    judge_llm_model: str
    nebius_api_key: str
    nebius_api_endpoint: str

def get_settings() -> Settings:
    return Settings(
        llm_model=os.environ.get("LLM_MODEL"),
        judge_llm_model=os.environ.get("JUDGE_LLM_MODEL"),
        nebius_api_key=os.environ.get("NEBIUS_TOKEN_FACTORY_API_KEY"),
        nebius_api_endpoint=os.environ.get("NEBIUS_TOKEN_FACTORY_API_ENDPOINT"),
    )