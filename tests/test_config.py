"""Tests for config module."""

import pytest
from pydantic import ValidationError

from deepseek_v4_context_bench.config import (
    MAX_CONTEXT_LENGTHS,
    PRICING_TABLE,
    BenchmarkConfig,
    ModelProvider,
    config,
)


class TestModelProvider:
    """Tests for ModelProvider enum."""

    def test_model_provider_values(self):
        """Test model provider enum values."""
        assert ModelProvider.DEEPSEEK_FLASH.value == "deepseek/deepseek-v4-flash"
        assert ModelProvider.DEEPSEEK_PRO.value == "deepseek/deepseek-v4-pro"
        assert ModelProvider.LLAMA_SCOUT.value == "meta-llama/llama-4-scout-17b-16e-instruct"

    def test_pricing_table_has_all_models(self):
        """Test that pricing table has all models."""
        for model in ModelProvider:
            assert model in PRICING_TABLE

    def test_max_context_lengths_has_all_models(self):
        """Test that max context lengths has all models."""
        for model in ModelProvider:
            assert model in MAX_CONTEXT_LENGTHS


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        cfg = BenchmarkConfig()
        assert cfg.openrouter_base_url == "https://openrouter.ai/api/v1"
        assert cfg.openrouter_timeout == 300
        assert cfg.max_tokens == 1_000_000
        assert cfg.output_tokens == 1024
        assert cfg.temperature == 0.0
        assert cfg.max_retries == 5
        assert cfg.retry_delay == 1.0
        assert cfg.retry_backoff == 2.0
        assert cfg.max_budget_usd == 100.0
        assert cfg.dry_run is False
        assert cfg.output_dir == "./results"
        assert cfg.verbose is False

    def test_config_with_custom_values(self):
        """Test configuration with custom values."""
        cfg = BenchmarkConfig(
            max_tokens=500_000,
            temperature=0.5,
            dry_run=True,
        )
        assert cfg.max_tokens == 500_000
        assert cfg.temperature == 0.5
        assert cfg.dry_run is True

    def test_invalid_temperature(self):
        """Test that invalid temperature raises error."""
        with pytest.raises(ValidationError):
            BenchmarkConfig(temperature=3.0)

    def test_invalid_max_tokens(self):
        """Test that invalid max_tokens raises error."""
        with pytest.raises(ValidationError):
            BenchmarkConfig(max_tokens=500)

    def test_invalid_retries(self):
        """Test that invalid max_retries raises error."""
        with pytest.raises(ValidationError):
            BenchmarkConfig(max_retries=15)

    def test_api_key_validation(self):
        """Test API key validation."""
        # Valid key starting with 'sk-'
        cfg = BenchmarkConfig(openrouter_api_key="sk-test123")
        assert cfg.openrouter_api_key == "sk-test123"

        # Invalid key should raise error
        with pytest.raises(ValidationError):
            BenchmarkConfig(openrouter_api_key="invalid-key")

    def test_get_model_pricing(self):
        """Test getting model pricing."""
        cfg = BenchmarkConfig()

        flash_input, flash_output = cfg.get_model_pricing(ModelProvider.DEEPSEEK_FLASH)
        assert flash_input == 0.14
        assert flash_output == 0.28

        pro_input, pro_output = cfg.get_model_pricing(ModelProvider.DEEPSEEK_PRO)
        assert pro_input == 0.435
        assert pro_output == 0.87

    def test_estimate_cost(self):
        """Test cost estimation."""
        cfg = BenchmarkConfig()

        # Test with 1M input tokens and 1K output tokens
        cost = cfg.estimate_cost(ModelProvider.DEEPSEEK_FLASH, 1_000_000, 1_000)
        expected = (1_000_000 / 1_000_000) * 0.14 + (1_000 / 1_000_000) * 0.28
        assert cost == pytest.approx(expected, rel=1e-6)

    def test_get_max_context_length(self):
        """Test getting max context length."""
        cfg = BenchmarkConfig(max_tokens=500_000)

        # Should return min of model max and config max
        length = cfg.get_max_context_length(ModelProvider.DEEPSEEK_FLASH)
        assert length == 500_000

        # Test with higher config max
        cfg2 = BenchmarkConfig(max_tokens=2_000_000)
        length2 = cfg2.get_max_context_length(ModelProvider.DEEPSEEK_FLASH)
        assert length2 == 1_000_000  # Model limit

    def test_global_config_instance(self):
        """Test that global config instance exists."""
        assert isinstance(config, BenchmarkConfig)
