"""Tests for tokenizer module."""

import pytest

from deepseek_v4_context_bench.tokenizer import Tokenizer, get_tokenizer, tokenizer


class TestTokenizer:
    """Tests for Tokenizer class."""

    def test_init_default_encoding(self):
        """Test tokenizer initialization with default encoding."""
        t = Tokenizer()
        assert t.encoding_name == "cl100k_base"

    def test_init_custom_encoding(self):
        """Test tokenizer initialization with custom encoding."""
        t = Tokenizer("p50k_base")
        assert t.encoding_name == "p50k_base"

    def test_count_tokens_empty(self):
        """Test token counting with empty string."""
        assert tokenizer.count_tokens("") == 0

    def test_count_tokens_simple(self):
        """Test token counting with simple text."""
        tokens = tokenizer.count_tokens("Hello world")
        assert tokens > 0

    def test_count_tokens_longer(self):
        """Test token counting with longer text."""
        text = "The quick brown fox jumps over the lazy dog."
        tokens = tokenizer.count_tokens(text)
        assert tokens > 5  # Should be several tokens

    def test_count_message_tokens(self):
        """Test message token counting."""
        tokens = tokenizer.count_message_tokens("user", "Hello")
        assert tokens > 0

    def test_count_messages_tokens(self):
        """Test multiple messages token counting."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        tokens = tokenizer.count_messages_tokens(messages)
        assert tokens > 0

    def test_truncate_to_tokens_exact(self):
        """Test truncation when text is exact size."""
        text = "Hello world"
        original_tokens = tokenizer.count_tokens(text)
        result = tokenizer.truncate_to_tokens(text, original_tokens)
        assert result == text

    def test_truncate_to_tokens_longer(self):
        """Test truncation when text is longer."""
        text = "The quick brown fox jumps over the lazy dog multiple times"
        max_tokens = 5
        result = tokenizer.truncate_to_tokens(text, max_tokens)
        result_tokens = tokenizer.count_tokens(result)
        assert result_tokens <= max_tokens

    def test_pad_to_tokens_already_sufficient(self):
        """Test padding when text already meets target."""
        text = "Hello world"
        original_tokens = tokenizer.count_tokens(text)
        result = tokenizer.pad_to_tokens(text, original_tokens)
        assert result == text

    def test_pad_to_tokens_increases(self):
        """Test padding increases token count."""
        text = "Hi"
        original_tokens = tokenizer.count_tokens(text)
        target = original_tokens + 10
        result = tokenizer.pad_to_tokens(text, target)
        result_tokens = tokenizer.count_tokens(result)
        assert result_tokens >= original_tokens

    def test_create_filler_text(self):
        """Test filler text creation."""
        target = 100
        filler = tokenizer.create_filler_text(target)
        tokens = tokenizer.count_tokens(filler)
        assert tokens >= target - 5  # Allow small variance

    def test_create_filler_text_with_pattern(self):
        """Test filler text creation with custom pattern."""
        pattern = "Test pattern. "
        filler = tokenizer.create_filler_text(50, pattern)
        assert pattern in filler

    def test_build_context_prompt(self):
        """Test context prompt building."""
        context = "This is the context."
        question = "What is this?"
        messages = tokenizer.build_context_prompt(context, question)

        assert len(messages) >= 1
        assert "context" in messages[0]["content"].lower()
        assert "question" in messages[0]["content"].lower()

    def test_build_context_prompt_with_system(self):
        """Test context prompt building with system prompt."""
        context = "Context"
        question = "Question?"
        system = "You are helpful."
        messages = tokenizer.build_context_prompt(context, question, system)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_split_into_chunks(self):
        """Test text chunking."""
        text = " ".join(["word"] * 100)
        chunks = tokenizer.split_into_chunks(text, 10)
        assert len(chunks) > 0

    def test_split_into_chunks_with_overlap(self):
        """Test text chunking with overlap."""
        text = " ".join(["word"] * 100)
        chunks = tokenizer.split_into_chunks(text, 10, overlap=5)
        assert len(chunks) > 0

    def test_estimate_cost(self):
        """Test cost estimation."""
        cost = tokenizer.estimate_cost(
            input_tokens=1_000_000,
            output_tokens=1_000,
            price_per_1m_input=0.10,
            price_per_1m_output=0.25,
        )
        expected = 0.10 + 0.00025
        assert cost == pytest.approx(expected, rel=1e-6)


def test_get_tokenizer_factory():
    """Test get_tokenizer factory function."""
    t = get_tokenizer("cl100k_base")
    assert isinstance(t, Tokenizer)
    assert t.encoding_name == "cl100k_base"
