from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuración centralizada de la aplicación.

    Se carga desde variables de entorno o desde un archivo .env local.
    No se debe subir ninguna API key real al repositorio.
    """

    azure_openai_api_key: str = Field(default="", alias="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: str = Field(default="", alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_version: str = Field(
        default="2024-10-21",
        alias="AZURE_OPENAI_API_VERSION",
    )
    azure_openai_deployment: str = Field(
        default="gpt-4o-mini",
        alias="AZURE_OPENAI_DEPLOYMENT",
    )

    request_timeout_seconds: float = Field(default=60.0, alias="REQUEST_TIMEOUT_SECONDS")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
