from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    region: str = "ap-south-1"
    bucket: str = "crop-advisor-images"
    cases_table: str = "cases"
    kb_index: str = "crop_knowledge"
    sagemaker_endpoint: str | None = None
    sagemaker_region: str = "ap-south-1"
    guardrail_id: str | None = None
    guardrail_version: str = "1"
    langfuse_host: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_public_key: str | None = None

    model_config = {"env_file": ".env", "env_prefix": "CA_"}


settings = Settings()
