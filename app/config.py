from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    llm_temperature: float = 0.2

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vectorstore_dir: str = "./vectorstore"
    experience_top_k: int = 5
    knowledge_top_k: int = 5
    memory_min_sim: float = 0.35

    auto_threshold: float = 0.80
    review_threshold: float = 0.60
    max_retries: int = 2

    web_search_provider: str = "duckduckgo"
    web_search_timeout: int = 10
    tool_timeout: int = 15

    data_dir: str = "./data"
    knowledge_dir: str = "./data/knowledge"
    experiences_file: str = "./data/experiences/episodes.jsonl"
    strategy_file: str = "./data/experiences/strategies.json"

    @property
    def vectorstore_path(self) -> Path:
        p = Path(self.vectorstore_dir); p.mkdir(parents=True, exist_ok=True); return p

    @property
    def experiences_path(self) -> Path:
        p = Path(self.experiences_file); p.parent.mkdir(parents=True, exist_ok=True); return p

    @property
    def strategy_path(self) -> Path:
        p = Path(self.strategy_file); p.parent.mkdir(parents=True, exist_ok=True); return p

    @property
    def knowledge_path(self) -> Path:
        p = Path(self.knowledge_dir); p.mkdir(parents=True, exist_ok=True); return p


@lru_cache
def get_settings() -> Settings:
    return Settings()
