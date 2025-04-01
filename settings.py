from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file = ".env",
        extra = "ignore"
    )

    VOYAGE_API_KEY: str
    PINECONE_API_KEY: str
    GEMINI_API_KEY: str
    GROQ_API_KEY: str

Config = Settings()