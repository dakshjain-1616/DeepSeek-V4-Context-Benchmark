"""Tests for card module."""


from deepseek_v4_context_bench.card import generate_dataset_card, generate_model_card


class TestGenerateDatasetCard:
    """Tests for generate_dataset_card function."""

    def test_basic_card(self):
        """Test basic dataset card generation."""
        card = generate_dataset_card()
        assert "DeepSeek V4 Context Benchmark" in card
        assert "deepseek-v4-flash" in card
        assert "deepseek-v4-pro" in card
        assert "llama-4-scout" in card

    def test_custom_title(self):
        """Test card with custom title."""
        card = generate_dataset_card(title="Custom Title")
        assert "Custom Title" in card

    def test_custom_version(self):
        """Test card with custom version."""
        card = generate_dataset_card(version="1.0.0")
        assert "1.0.0" in card

    def test_contains_model_info(self):
        """Test that card contains model information."""
        card = generate_dataset_card()
        assert "deepseek/deepseek-v4-flash" in card
        assert "deepseek/deepseek-v4-pro" in card
        assert "meta-llama/llama-4-scout-17b-16e-instruct" in card

    def test_contains_corpus_types(self):
        """Test that card contains corpus types."""
        card = generate_dataset_card()
        assert "NIAH" in card
        assert "Multi-hop Reasoning" in card
        assert "Codebase Analysis" in card
        assert "Synthetic Data" in card

    def test_contains_pricing(self):
        """Test that card contains pricing information."""
        card = generate_dataset_card()
        assert "$0.10" in card
        assert "1M tokens" in card

    def test_contains_usage_examples(self):
        """Test that card contains usage examples."""
        card = generate_dataset_card()
        assert "dsv4ctx" in card
        assert "run" in card

    def test_contains_citation(self):
        """Test that card contains citation information."""
        card = generate_dataset_card()
        assert "@software" in card
        assert "bibtex" in card.lower()
        assert "deepseek_v4_context_bench" in card

    def test_contains_architecture_diagram(self):
        """Test that card contains architecture diagram."""
        card = generate_dataset_card()
        assert "mermaid" in card


class TestGenerateModelCard:
    """Tests for generate_model_card function."""

    def test_basic_model_card(self):
        """Test basic model card generation."""
        results = {
            "statistics": {
                "total_tasks": 100,
                "completed_tasks": 95,
                "failed_tasks": 5,
                "accuracy": 0.85,
                "avg_latency_ms": 150.0,
            }
        }
        card = generate_model_card("test-model", results)
        assert "test-model" in card
        assert "85.00%" in card
        assert "150.0" in card

    def test_contains_performance_metrics(self):
        """Test that card contains performance metrics."""
        results = {
            "statistics": {
                "total_tasks": 10,
                "completed_tasks": 8,
                "failed_tasks": 2,
                "accuracy": 0.8,
                "avg_latency_ms": 100.0,
            }
        }
        card = generate_model_card("model", results)
        assert "Accuracy" in card
        assert "Latency" in card
        assert "Total Tasks" in card

    def test_contains_corpus_breakdown(self):
        """Test that card contains corpus breakdown."""
        results = {"statistics": {}}
        card = generate_model_card("model", results)
        assert "Corpus Breakdown" in card

    def test_contains_context_length_performance(self):
        """Test that card contains context length performance."""
        results = {"statistics": {}}
        card = generate_model_card("model", results)
        assert "Context Length" in card
