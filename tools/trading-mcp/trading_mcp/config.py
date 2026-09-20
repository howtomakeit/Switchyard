"""Environment-driven settings for the trading MCP server.

Every knob is read from the process environment so the server can be started by
a supervisor or a tunnel wrapper without a config file. The defaults are
deliberately safe: paper trading, a small notional cap, and no credentials.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Polymarket's CLOB serves quotes, books, and order placement.
DEFAULT_CLOB_URL = "https://clob.polymarket.com"
# The public data API serves positions and needs no credentials.
DEFAULT_DATA_URL = "https://data-api.polymarket.com"
DEFAULT_LLM_URL = "https://api.x.ai/v1"
DEFAULT_LLM_MODEL = "grok-4"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


@dataclass(frozen=True)
class Settings:
    """Resolved server configuration.

    `live` is the only switch that lets real capital move. It is false unless
    `TRADING_MCP_MODE=live` is set explicitly, so an accidental start, a stale
    shell, or a copied systemd unit cannot place a real order.
    """

    live: bool
    max_order_usd: float
    clob_url: str
    data_url: str
    private_key: str | None
    funder_address: str | None
    signature_type: int
    llm_url: str
    llm_model: str
    llm_api_key: str | None
    auth_token: str | None
    request_timeout: float
    cache_path: str

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from the process environment."""
        return cls(
            live=os.getenv("TRADING_MCP_MODE", "paper").strip().lower() == "live",
            max_order_usd=_env_float("TRADING_MCP_MAX_ORDER_USD", 50.0),
            clob_url=os.getenv("POLYMARKET_CLOB_URL", DEFAULT_CLOB_URL).rstrip("/"),
            data_url=os.getenv("POLYMARKET_DATA_URL", DEFAULT_DATA_URL).rstrip("/"),
            private_key=os.getenv("POLYMARKET_PRIVATE_KEY"),
            funder_address=os.getenv("POLYMARKET_FUNDER_ADDRESS"),
            signature_type=int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "0")),
            llm_url=os.getenv("TRADING_MCP_LLM_URL", DEFAULT_LLM_URL).rstrip("/"),
            llm_model=os.getenv("TRADING_MCP_LLM_MODEL", DEFAULT_LLM_MODEL),
            llm_api_key=os.getenv("XAI_API_KEY") or os.getenv("TRADING_MCP_LLM_API_KEY"),
            auth_token=os.getenv("TRADING_MCP_AUTH_TOKEN"),
            request_timeout=_env_float("TRADING_MCP_TIMEOUT", 20.0),
            cache_path=os.getenv("TRADING_MCP_CACHE", ".dependency-cache.json"),
        )

    def require_live_credentials(self) -> str:
        """Return the signing key, or raise if live trading is not fully configured."""
        if not self.live:
            raise RuntimeError("live trading is disabled; set TRADING_MCP_MODE=live to enable it")
        if not self.private_key:
            raise RuntimeError("POLYMARKET_PRIVATE_KEY is required when TRADING_MCP_MODE=live")
        return self.private_key
