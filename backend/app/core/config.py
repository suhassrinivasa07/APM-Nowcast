from typing import List, Union
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "APM-Nowcast API"
    APP_VERSION: str = "0.1.0"
    APP_ENV: str = "development"
    DEBUG: bool = True
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_V1_STR: str = "/api/v1"

    # CORS
    ALLOWED_ORIGINS: Union[str, List[str]] = (
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,https://apm-nowcast.vercel.app,*"
    )

    @property
    def cors_origins(self) -> List[str]:
        if isinstance(self.ALLOWED_ORIGINS, list):
            origins = list(self.ALLOWED_ORIGINS)
        else:
            origins = [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]
        for req in ["https://apm-nowcast.vercel.app", "*"]:
            if req not in origins:
                origins.append(req)
        return origins

    # Hardware & ML configuration (Prepared for Day 2+)
    DEVICE: str = "cuda"
    GPU_MEMORY_FRACTION: float = 0.8

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
