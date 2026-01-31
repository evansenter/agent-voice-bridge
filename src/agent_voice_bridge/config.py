"""Configuration management."""

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TwilioSettings(BaseSettings):
    """Twilio configuration."""

    account_sid: str = Field(default="", alias="TWILIO_ACCOUNT_SID")
    auth_token: str = Field(default="", alias="TWILIO_AUTH_TOKEN")
    phone_number: str = Field(default="", alias="TWILIO_PHONE_NUMBER")


class GeminiSettings(BaseSettings):
    """Gemini configuration."""

    api_key: str = Field(default="", alias="GEMINI_API_KEY")
    model: str = Field(default="gemini-2.0-flash-exp")


class OpenAISettings(BaseSettings):
    """OpenAI configuration."""

    api_key: str = Field(default="", alias="OPENAI_API_KEY")
    model: str = Field(default="gpt-4o-realtime-preview")


class ServerSettings(BaseSettings):
    """Server configuration."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8082)
    public_url: str = Field(default="", alias="VOICE_BRIDGE_URL")


class Settings(BaseSettings):
    """Main settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ai_provider: str = Field(default="gemini", alias="AI_PROVIDER")
    twilio: TwilioSettings = Field(default_factory=TwilioSettings)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    server: ServerSettings = Field(default_factory=ServerSettings)

    # System prompt for the AI
    system_prompt: str = Field(
        default="You are a helpful AI assistant on a phone call. "
        "Be conversational, concise, and natural. "
        "Speak clearly and at a moderate pace."
    )


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()
