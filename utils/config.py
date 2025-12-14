"""
Configuration Management

Pydantic-based configuration system with environment variable
support and validation for all system components.
"""

import os
from typing import Dict, Any, Optional

try:
    from pydantic_settings import BaseSettings, Field
except ImportError:
    # Fallback for older pydantic versions
    try:
        from pydantic import BaseSettings, Field
    except ImportError:
        # Final fallback - create a simple settings class
        BaseSettings = object
        def Field(default=None, **kwargs):
            return default


class Settings(BaseSettings):
    """Main settings class for the ETL Sentiment system."""
    
    # Application settings
    app_name: str = Field(default="ETL Sentiment", env="APP_NAME")
    app_version: str = Field(default="0.1.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Ollama settings
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3", env="OLLAMA_MODEL")
    ollama_timeout: int = Field(default=30, env="OLLAMA_TIMEOUT")
    
    # LangGraph settings
    langgraph_timeout: int = Field(default=60, env="LANGGRAPH_TIMEOUT")
    langgraph_max_retries: int = Field(default=3, env="LANGGRAPH_MAX_RETRIES")
    
    # Selenium settings
    selenium_driver_path: Optional[str] = Field(default=None, env="SELENIUM_DRIVER_PATH")
    selenium_headless: bool = Field(default=True, env="SELENIUM_HEADLESS")
    selenium_timeout: int = Field(default=30, env="SELENIUM_TIMEOUT")
    
    # Healing settings
    healing_enabled: bool = Field(default=True, env="HEALING_ENABLED")
    healing_max_attempts: int = Field(default=3, env="HEALING_MAX_ATTEMPTS")
    healing_backup_enabled: bool = Field(default=True, env="HEALING_BACKUP_ENABLED")
    
    # Logging settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default="data/logs/system.log", env="LOG_FILE")
    
    # Data settings
    data_dir: str = Field(default="data", env="DATA_DIR")
    metrics_file: str = Field(default="data/metrics/healing_metrics.json", env="METRICS_FILE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()