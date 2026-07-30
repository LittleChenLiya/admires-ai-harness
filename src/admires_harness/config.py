"""Environment-backed application configuration."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ADMIRE_", env_file=".env", extra="ignore")

    environment: str = "development"
    database_path: Path = Path(".admire/harness.db")
    workspace_root: Path = Path(".admire/workspaces")
    web_dist: Path = Path("web/dist")
    model_provider: str = "mock"
    model_base_url: str = "https://api.openai.com/v1"
    model_name: str = "gpt-4.1-mini"
    model_api_key: str | None = Field(default=None, repr=False)
    allow_network: bool = False
    log_level: str = "INFO"
    cors_origins: tuple[str, ...] = ("http://localhost:5173",)

    def ensure_directories(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.workspace_root.mkdir(parents=True, exist_ok=True)
