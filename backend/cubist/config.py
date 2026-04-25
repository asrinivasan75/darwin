from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env", extra="ignore")

    # LLM provider — selects which SDK handles every complete() call.
    # "claude" uses Anthropic (ANTHROPIC_API_KEY); "gemini" uses Google
    # GenAI (GOOGLE_API_KEY). Switching providers does NOT rewrite model
    # IDs below — set them to provider-appropriate values in .env.
    llm_provider: Literal["claude", "gemini"] = "claude"

    anthropic_api_key: str = ""
    google_api_key: str = ""

    strategist_model: str = "claude-opus-4-6"
    player_model: str = "claude-sonnet-4-6"
    builder_model: str = "claude-sonnet-4-6"

    database_url: str = "sqlite:///./cubist.db"

    time_per_move_ms: int = 20_000
    games_per_pairing: int = 2
    max_parallel_games: int = 2
    max_moves_per_game: int = 120

    # Where tournament games execute. "local" runs them here via
    # asyncio.gather (current behavior). "modal" dispatches each game
    # to a Modal container — real OS-level parallelism, no GIL, frees
    # this machine's CPU. Requires `modal token` to be configured and
    # `modal deploy backend/cubist/tournament/modal_runner.py` to have
    # been run at least once.
    tournament_backend: str = "local"

    api_host: str = "127.0.0.1"
    api_port: int = 8000


settings = Settings()
