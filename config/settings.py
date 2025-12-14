"""
Main Settings Configuration

Central configuration management using pydantic for validation
and environment variable support.
"""

import os
from typing import Dict, Any, Optional


class Settings:
    """Main settings class for the ETL Sentiment system."""
    
    def __init__(self):
        # Application settings
        self.app_name = os.getenv("APP_NAME", "ETL Sentiment")
        self.app_version = os.getenv("APP_VERSION", "0.1.0")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # Ollama settings
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
        self.ollama_timeout = int(os.getenv("OLLAMA_TIMEOUT", "30"))
        
        # LangGraph settings
        self.langgraph_timeout = int(os.getenv("LANGGRAPH_TIMEOUT", "60"))
        self.langgraph_max_retries = int(os.getenv("LANGGRAPH_MAX_RETRIES", "3"))
        
        # Selenium settings
        self.selenium_driver_path = os.getenv("SELENIUM_DRIVER_PATH")
        self.selenium_headless = os.getenv("SELENIUM_HEADLESS", "true").lower() == "true"
        self.selenium_timeout = int(os.getenv("SELENIUM_TIMEOUT", "30"))
        
        # Healing settings
        self.healing_enabled = os.getenv("HEALING_ENABLED", "true").lower() == "true"
        self.healing_max_attempts = int(os.getenv("HEALING_MAX_ATTEMPTS", "3"))
        self.healing_backup_enabled = os.getenv("HEALING_BACKUP_ENABLED", "true").lower() == "true"
        
        # Logging settings
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.log_file = os.getenv("LOG_FILE", "data/logs/system.log")
        
        # Data settings
        self.data_dir = os.getenv("DATA_DIR", "data")
        self.metrics_file = os.getenv("METRICS_FILE", "data/metrics/healing_metrics.json")
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        # Basic validation - can be expanded
        required_dirs = [
            self.data_dir,
            os.path.dirname(self.log_file),
            os.path.dirname(self.metrics_file)
        ]
        
        for dir_path in required_dirs:
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
        
        return True


# Global settings instance
settings = Settings()