from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Provider name alias — used by per-role overrides + complete() dispatch.
Provider = Literal["claude", "gemini", "openai"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env", extra="ignore")

    # Default LLM provider — selects which SDK handles a complete() call
    # when no per-role override is set. "claude" uses Anthropic
    # (ANTHROPIC_API_KEY); "gemini" uses Google GenAI (GOOGLE_API_KEY);
    # "openai" uses OpenAI (OPENAI_API_KEY). Switching providers does
    # NOT rewrite model IDs below — set them to provider-appropriate
    # values in .env.
    llm_provider: Provider = "claude"

    # Per-role provider overrides. When unset, fall back to llm_provider.
    # Lets you mix providers across roles, e.g. strategist=claude (deeper
    # reasoning) + builder=gemini (faster code generation). The model ID
    # set in <role>_model below MUST match the provider chosen for that
    # role, since each provider only knows its own model namespace.
    strategist_provider: Provider | None = None
    player_provider: Provider | None = None
    builder_provider: Provider | None = None
    adversary_provider: Provider | None = None

    anthropic_api_key: str = ""
    google_api_key: str = ""
    openai_api_key: str = ""

    # Default model IDs are anchored to currently-shipping aliases:
    # opus-4-7 is the latest Anthropic flagship at time of writing;
    # sonnet-4-6 is the workhorse code-generation model. Override per-
    # role in .env when you want a non-Anthropic provider, since the
    # ID namespace is not shared across providers.
    strategist_model: str = "claude-opus-4-7"
    player_model: str = "claude-sonnet-4-6"
    builder_model: str = "claude-sonnet-4-6"
    # The adversary critiques builder output before validation; pairing
    # it with a different provider/family from the builder is the point
    # — homogeneous critique tends to rubber-stamp homogeneous code.
    adversary_model: str = "claude-opus-4-7"

    # Toggle for the adversary → fixer chain. When false, the orchestrator
    # validates builder output as-is, matching the pre-adversary behavior.
    enable_adversary: bool = True

    def provider_for(
        self,
        role: Literal["strategist", "player", "builder", "adversary"],
    ) -> Provider:
        """Resolve the provider for a role, defaulting to ``llm_provider``."""
        override = getattr(self, f"{role}_provider")
        return override or self.llm_provider

    database_url: str = "sqlite:///./darwin.db"

    time_per_move_ms: int = 20_000
    games_per_pairing: int = 2
    # Concurrency cap for local-tournament asyncio.gather. NOT used by
    # the Modal backend (which scales via max_containers in modal_runner.py).
    # 16 is fine for typical 90-game cohorts on a laptop with the LLM-driven
    # path; pure-code games are CPU-bound and can hit this if the local
    # backend is in use.
    max_parallel_games: int = 16
    max_moves_per_game: int = 120

    # Where tournament games execute. "local" runs them here via
    # asyncio.gather (current behavior). "modal" dispatches each game
    # to a Modal container — real OS-level parallelism, no GIL, frees
    # this machine's CPU. Requires `modal token` to be configured and
    # `modal deploy backend/darwin/tournament/modal_runner.py` to have
    # been run at least once.
    tournament_backend: str = "local"

    api_host: str = "127.0.0.1"
    api_port: int = 8000


settings = Settings()
