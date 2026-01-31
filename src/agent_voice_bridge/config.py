"""Configuration management."""

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Main settings - all config in one flat class for proper .env loading."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Twilio
    twilio_account_sid: str = Field(default="", alias="TWILIO_ACCOUNT_SID")
    twilio_auth_token: str = Field(default="", alias="TWILIO_AUTH_TOKEN")
    twilio_phone_number: str = Field(default="", alias="TWILIO_PHONE_NUMBER")
    
    # AI Provider
    ai_provider: str = Field(default="gemini", alias="AI_PROVIDER")
    
    # Gemini
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.0-flash-exp", alias="GEMINI_MODEL")
    
    # OpenAI
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-realtime-preview", alias="OPENAI_MODEL")
    
    # Server
    server_host: str = Field(default="0.0.0.0", alias="SERVER_HOST")
    server_port: int = Field(default=8082, alias="SERVER_PORT")
    public_url: str = Field(default="", alias="VOICE_BRIDGE_URL")

    # System prompt for the AI
    system_prompt: str = Field(
        default="You are a helpful AI assistant on a phone call. "
        "Be conversational, concise, and natural. "
        "Speak clearly and at a moderate pace.",
        alias="SYSTEM_PROMPT"
    )


# Singleton
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get application settings."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
