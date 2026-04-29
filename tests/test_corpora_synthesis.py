"""Tests for Synthesis corpus generator."""


from deepseek_v4_context_bench.corpora.synthesis import (
    SynthesisConfig,
    SynthesisCorpus,
    SynthesisSample,
)


class TestSynthesisConfig:
    """Tests for SynthesisConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = SynthesisConfig()
        assert config.seed == 42
        assert config.content_type == "mixed"
        assert config.total_tokens_target == 10000
        assert config.paragraphs == 50
        assert config.include_structured_data is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = SynthesisConfig(seed=123, content_type="narrative", paragraphs=30)
        assert config.seed == 123
        assert config.content_type == "narrative"
        assert config.paragraphs == 30


class TestSynthesisSample:
    """Tests for SynthesisSample."""

    def test_sample_creation(self):
        """Test creating a sample."""
        sample = SynthesisSample(
            content="Test content",
            content_type="narrative",
            question="What?",
            expected_answer="Answer",
            metadata={"key": "value"},
        )
        assert sample.content == "Test content"
        assert sample.content_type == "narrative"
        assert sample.question == "What?"
        assert sample.expected_answer == "Answer"
        assert sample.metadata == {"key": "value"}


class TestSynthesisCorpus:
    """Tests for SynthesisCorpus."""

    def test_init_default(self):
        """Test initialization with default config."""
        corpus = SynthesisCorpus()
        assert corpus.config.content_type == "mixed"

    def test_init_custom(self):
        """Test initialization with custom config."""
        config = SynthesisConfig(seed=123, content_type="dialogue")
        corpus = SynthesisCorpus(config)
        assert corpus.config.seed == 123
        assert corpus.config.content_type == "dialogue"

    def test_deterministic_generation(self):
        """Test that same seed produces identical output."""
        corpus1 = SynthesisCorpus(SynthesisConfig(seed=42, paragraphs=10))
        corpus2 = SynthesisCorpus(SynthesisConfig(seed=42, paragraphs=10))

        samples1 = corpus1.generate(3)
        samples2 = corpus2.generate(3)

        assert len(samples1) == len(samples2) == 3
        for s1, s2 in zip(samples1, samples2, strict=True):
            assert s1.content == s2.content
            assert s1.question == s2.question
            assert s1.expected_answer == s2.expected_answer
            assert s1.content_type == s2.content_type

    def test_generate_single(self):
        """Test generating a single sample."""
        corpus = SynthesisCorpus(SynthesisConfig(seed=42, paragraphs=10))
        sample = corpus.generate_single()

        assert isinstance(sample, SynthesisSample)
        assert len(sample.content) > 0
        assert len(sample.question) > 0
        assert len(sample.expected_answer) > 0

    def test_generate_multiple(self):
        """Test generating multiple samples."""
        corpus = SynthesisCorpus(SynthesisConfig(seed=42, paragraphs=10))
        samples = corpus.generate(3)

        assert len(samples) == 3
        for sample in samples:
            assert isinstance(sample, SynthesisSample)
            assert len(sample.content) > 0

    def test_different_content_types(self):
        """Test different content type settings."""
        for content_type in ["narrative", "dialogue", "structured", "mixed"]:
            corpus = SynthesisCorpus(SynthesisConfig(
                seed=42,
                content_type=content_type,
                paragraphs=10
            ))
            sample = corpus.generate_single()
            assert len(sample.content) > 0

    def test_narrative_content(self):
        """Test narrative content generation."""
        corpus = SynthesisCorpus(SynthesisConfig(
            seed=42,
            content_type="narrative",
            paragraphs=5
        ))
        sample = corpus.generate_single()

        assert sample.content_type == "narrative"
        assert "Once upon a time" in sample.content

    def test_dialogue_content(self):
        """Test dialogue content generation."""
        corpus = SynthesisCorpus(SynthesisConfig(
            seed=42,
            content_type="dialogue",
            paragraphs=5
        ))
        sample = corpus.generate_single()

        assert sample.content_type == "dialogue"
        assert '"' in sample.content and "said" in sample.content

    def test_structured_content(self):
        """Test structured data generation."""
        corpus = SynthesisCorpus(SynthesisConfig(
            seed=42,
            content_type="structured",
            paragraphs=3
        ))
        sample = corpus.generate_single()

        assert sample.content_type == "structured"
        assert "[" in sample.content and "{" in sample.content

    def test_metadata_populated(self):
        """Test that metadata is populated."""
        corpus = SynthesisCorpus(SynthesisConfig(seed=42, paragraphs=10))
        sample = corpus.generate_single()

        assert len(sample.metadata) > 0
        assert "content_type" in sample.metadata
        assert "paragraphs" in sample.metadata
        assert "entities" in sample.metadata

    def test_vocabulary_usage(self):
        """Test that vocabulary words are used."""
        from deepseek_v4_context_bench.corpora.synthesis import VOCABULARY

        corpus = SynthesisCorpus(SynthesisConfig(seed=42, paragraphs=20))
        sample = corpus.generate_single()

        # At least some vocabulary should appear
        vocab_words = (
            VOCABULARY["adjective"] +
            VOCABULARY["noun"] +
            VOCABULARY["location"]
        )
        found = any(word.lower() in sample.content.lower() for word in vocab_words)
        assert found

    def test_planted_marker_in_content(self):
        """Test that the unique marker is physically present in the content."""
        corpus = SynthesisCorpus(SynthesisConfig(seed=42, paragraphs=5))
        sample = corpus.generate_single()
        assert sample.expected_answer in sample.content

    def test_expected_answer_matches_metadata_marker(self):
        """Test that expected_answer equals metadata['planted_marker']."""
        corpus = SynthesisCorpus(SynthesisConfig(seed=42, paragraphs=5))
        sample = corpus.generate_single()
        assert sample.expected_answer == sample.metadata["planted_marker"]

    def test_metadata_has_entities_list(self):
        """Test that metadata['entities'] is a populated list."""
        corpus = SynthesisCorpus(SynthesisConfig(seed=42, paragraphs=5))
        sample = corpus.generate_single()
        assert "entities" in sample.metadata
        assert isinstance(sample.metadata["entities"], list)
        assert len(sample.metadata["entities"]) > 0

    def test_different_seeds_produce_different_content(self):
        """Test that different seeds produce different generated content."""
        s1 = SynthesisCorpus(SynthesisConfig(seed=42, paragraphs=5)).generate_single()
        s2 = SynthesisCorpus(SynthesisConfig(seed=99, paragraphs=5)).generate_single()
        assert s1.content != s2.content

    def test_content_grows_with_more_paragraphs(self):
        """Test that requesting more paragraphs produces strictly longer content."""
        short = SynthesisCorpus(SynthesisConfig(seed=42, paragraphs=5)).generate_single()
        long = SynthesisCorpus(SynthesisConfig(seed=42, paragraphs=20)).generate_single()
        assert len(long.content) > len(short.content)

    def test_generate_zero_returns_empty_list(self):
        """Test that generate(0) returns an empty list."""
        corpus = SynthesisCorpus(SynthesisConfig(seed=42, paragraphs=5))
        assert corpus.generate(0) == []
