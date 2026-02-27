from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    aws_region: str = "eu-west-1"
    s3_bucket: str
    sagemaker_role_arn: str
    sagemaker_image_uri: str
    sagemaker_instance_type: str = "ml.p3.2xlarge"
    hf_token: str | None = None
    app_password: str


@lru_cache
def get_settings() -> Settings:
    return Settings()
