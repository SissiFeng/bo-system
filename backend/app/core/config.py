from functools import lru_cache
from typing import Any, Dict, Optional
import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings

# Default configuration values
DEFAULT_APP_NAME = "BO-Engine-API"
DEFAULT_APP_VERSION = "0.1.0"
DEFAULT_APP_ENV = "development"
DEFAULT_APP_PORT = 8000
DEFAULT_APP_HOST = "0.0.0.0"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_DATA_DIR = "./data"
DEFAULT_TASK_DIR = f"{DEFAULT_DATA_DIR}/tasks"
DEFAULT_RANDOM_SEED = 42
DEFAULT_INITIAL_SAMPLES = 10
DEFAULT_ACQUISITION_FUNCTION = "ei"
DEFAULT_KERNEL = "matern"
DEFAULT_EXPLORATION_WEIGHT = 0.5
DEFAULT_WS_PING_INTERVAL = 30
DEFAULT_MAX_WORKERS = 4

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables with fallback to defaults.
    """
    # Application
    APP_NAME: str = Field(DEFAULT_APP_NAME, env="APP_NAME")
    APP_VERSION: str = Field(DEFAULT_APP_VERSION, env="APP_VERSION")
    APP_ENV: str = Field(DEFAULT_APP_ENV, env="APP_ENV")
    APP_PORT: int = Field(DEFAULT_APP_PORT, env="APP_PORT")
    APP_HOST: str = Field(DEFAULT_APP_HOST, env="APP_HOST")
    
    # Logging
    LOG_LEVEL: str = Field(DEFAULT_LOG_LEVEL, env="LOG_LEVEL")
    
    # Data storage
    DATA_DIR: str = Field(DEFAULT_DATA_DIR, env="DATA_DIR")
    TASK_DIR: str = Field(DEFAULT_TASK_DIR, env="TASK_DIR")
    
    # Default optimization settings
    DEFAULT_RANDOM_SEED: int = Field(DEFAULT_RANDOM_SEED, env="DEFAULT_RANDOM_SEED")
    DEFAULT_INITIAL_SAMPLES: int = Field(DEFAULT_INITIAL_SAMPLES, env="DEFAULT_INITIAL_SAMPLES")
    DEFAULT_ACQUISITION_FUNCTION: str = Field(DEFAULT_ACQUISITION_FUNCTION, env="DEFAULT_ACQUISITION_FUNCTION")
    DEFAULT_KERNEL: str = Field(DEFAULT_KERNEL, env="DEFAULT_KERNEL")
    DEFAULT_EXPLORATION_WEIGHT: float = Field(DEFAULT_EXPLORATION_WEIGHT, env="DEFAULT_EXPLORATION_WEIGHT")
    
    # WebSocket
    WS_PING_INTERVAL: int = Field(DEFAULT_WS_PING_INTERVAL, env="WS_PING_INTERVAL")
    
    # Performance
    MAX_WORKERS: int = Field(DEFAULT_MAX_WORKERS, env="MAX_WORKERS")
    
    # Optional security settings
    API_KEY: Optional[str] = Field(None, env="API_KEY")
    JWT_SECRET: Optional[str] = Field(None, env="JWT_SECRET")
    
    # Optional async task processing
    CELERY_BROKER_URL: Optional[str] = Field(None, env="CELERY_BROKER_URL")
    CELERY_BACKEND_URL: Optional[str] = Field(None, env="CELERY_BACKEND_URL")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True
    }

    def make_dirs(self) -> None:
        """Create necessary directories if they don't exist."""
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.TASK_DIR, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings with caching to avoid reloading from environment.
    
    Returns:
        Settings: Application settings
    """
    settings = Settings()
    settings.make_dirs()
    return settings 
