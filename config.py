"""
Configuration module for the OpenAI API Proxy.

This module handles loading environment variables and provides a central
configuration object that can be imported throughout the application.
"""

import os
import secrets
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AppConfig(BaseModel):
    """
    Configuration class for the application.
    
    This provides a single source of truth for all configuration values
    and makes it easy to add new configuration options.
    """
    # OpenAI API configuration
    openai_api_key: str
    openai_base_url: str = Field(default="https://api.openai.com/v1")
    
    # Authentication settings
    # Authentication is disabled by default, but can be enabled if needed
    enable_auth: bool = Field(default=False)
    api_key_secret: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    
    # Server settings
    port: int = Field(default=8000)
    
    # Model restrictions (empty list means allow all models)
    allowed_models: List[str] = Field(default_factory=list)

    class Config:
        case_sensitive = False


def load_config() -> AppConfig:
    """
    Load the application configuration from environment variables.
    
    Returns:
        AppConfig: The application configuration object.
    
    Raises:
        ValueError: If required environment variables are missing.
    """
    # Required variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    # Optional variables with defaults
    enable_auth = os.getenv("ENABLE_AUTH", "false").lower() == "true"
    api_key_secret = os.getenv("API_KEY_SECRET", secrets.token_urlsafe(32))
    port = int(os.getenv("PORT", "8000"))
    openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    # Parse allowed models
    allowed_models_str = os.getenv("ALLOWED_MODELS", "")
    allowed_models = [model.strip() for model in allowed_models_str.split(",")] if allowed_models_str else []
    
    return AppConfig(
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        enable_auth=enable_auth,
        api_key_secret=api_key_secret,
        port=port,
        allowed_models=allowed_models
    )


# Create a global configuration instance
config = load_config()