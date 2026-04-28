"""Configuration settings and pricing for DeepSeek V4 Context Benchmark.

This module defines Pydantic settings for the benchmark, including
API configuration, model pricing (April 2026), and benchmark parameters.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelProvider(StrEnum):
    """Supported model providers via OpenRouter."""

    DEEPSEEK_FLASH = "deepseek/deepseek-v4-flash"
    DEEPSEEK_PRO = "deepseek/deepseek-v4-pro"
    LLAMA_SCOUT = "meta-llama/llama-4-scout-17b-16e-instruct"


# Pricing per 1M tokens (input/output) in USD.
# Source: live OpenRouter `/api/v1/models` snapshot taken on 2026-04-28.
PRICING_TABLE: Final[dict[ModelProvider, tuple[float, float]]] = {
    ModelProvider.DEEPSEEK_FLASH: (0.14, 0.28),     # deepseek/deepseek-v4-flash
    ModelProvider.DEEPSEEK_PRO:   (0.435, 0.87),    # deepseek/deepseek-v4-pro
    ModelProvider.LLAMA_SCOUT:    (0.08, 0.30),     # meta-llama/llama-4-scout
}

# Maximum context lengths per model (in tokens)
MAX_CONTEXT_LENGTHS: Final[dict[ModelProvider, int]] = {
    ModelProvider.DEEPSEEK_FLASH: 1_000_000,
    ModelProvider.DEEPSEEK_PRO: 1_000_000,
    ModelProvider.LLAMA_SCOUT: 1_000_000,
}


class BenchmarkConfig(BaseSettings):
    """Benchmark configuration settings.

    All settings can be overridden via environment variables
    with the prefix DSV4CTX_.
    """

    model_config = SettingsConfigDict(
        env_prefix="DSV4CTX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Configuration
    openrouter_api_key: str = Field(
        default="",
        description="OpenRouter API key",
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL",
    )
    openrouter_timeout: int = Field(
        default=300,
        ge=10,
        le=600,
        description="API request timeout in seconds",
    )

    # Benchmark Parameters
    max_tokens: int = Field(
        default=1_000_000,
        ge=1000,
        le=2_000_000,
        description="Maximum context length in tokens",
    )
    output_tokens: int = Field(
        default=1024,
        ge=1,
        le=8192,
        description="Maximum output tokens per request",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )

    # Retry Configuration
    max_retries: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of retries for failed requests",
    )
    retry_delay: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Initial retry delay in seconds",
    )
    retry_backoff: float = Field(
        default=2.0,
        ge=1.0,
        le=5.0,
        description="Exponential backoff multiplier",
    )

    # Budget Configuration
    max_budget_usd: float = Field(
        default=100.0,
        ge=0.0,
        description="Maximum budget in USD (0 = unlimited)",
    )
    dry_run: bool = Field(
        default=False,
        description="Run in dry-run mode (no API calls)",
    )

    # Output Configuration
    output_dir: str = Field(
        default="./results",
        description="Directory for benchmark results",
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose output",
    )

    @field_validator("openrouter_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate API key format."""
        if v and not v.startswith("sk-"):
            raise ValueError("OpenRouter API key must start with 'sk-'")
        return v

    def get_model_pricing(self, model: ModelProvider) -> tuple[float, float]:
        """Get pricing for a specific model.

        Args:
            model: The model provider enum.

        Returns:
            Tuple of (input_price_per_1m, output_price_per_1m) in USD.
        """
        return PRICING_TABLE.get(model, (0.0, 0.0))

    def estimate_cost(
        self,
        model: ModelProvider,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate API call cost.

        Args:
            model: The model provider.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Estimated cost in USD.
        """
        input_price, output_price = self.get_model_pricing(model)
        input_cost = (input_tokens / 1_000_000) * input_price
        output_cost = (output_tokens / 1_000_000) * output_price
        return input_cost + output_cost

    def get_max_context_length(self, model: ModelProvider) -> int:
        """Get maximum context length for a model.

        Args:
            model: The model provider.

        Returns:
            Maximum context length in tokens.
        """
        return min(MAX_CONTEXT_LENGTHS.get(model, 1_000_000), self.max_tokens)


# Global config instance
config = BenchmarkConfig()
