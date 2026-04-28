"""Tests for NIAH corpus generator."""


from deepseek_v4_context_bench.corpora.niah import NIAHConfig, NIAHCorpus, NIAHSample


class TestNIAHConfig:
    """Tests for NIAHConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = NIAHConfig()
        assert config.seed == 42
        assert config.needle_count == 1
        assert config.haystack_sentences == 1000
        assert config.needle_position == "random"
        assert config.code_length == 8

    def test_custom_config(self):
        """Test custom configuration."""
        config = NIAHConfig(seed=123, needle_count=3, haystack_sentences=500)
        assert config.seed == 123
        assert config.needle_count == 3
        assert config.haystack_sentences == 500


class TestNIAHSample:
    """Tests for NIAHSample."""

    def test_sample_creation(self):
        """Test creating a sample."""
        sample = NIAHSample(
            text="Test text",
            needles=[{"text": "needle", "question": "What?", "answer": "ABC"}],
            question="What is the code?",
            expected_answer="ABC123",
            needle_positions=[10],
        )
        assert sample.text == "Test text"
        assert sample.question == "What is the code?"
        assert sample.expected_answer == "ABC123"
        assert len(sample.needles) == 1


class TestNIAHCorpus:
    """Tests for NIAHCorpus."""

    def test_init_default(self):
        """Test initialization with default config."""
        corpus = NIAHCorpus()
        assert corpus.config.seed == 42

    def test_init_custom(self):
        """Test initialization with custom config."""
        config = NIAHConfig(seed=123)
        corpus = NIAHCorpus(config)
        assert corpus.config.seed == 123

    def test_deterministic_generation(self):
        """Test that same seed produces identical output."""
        corpus1 = NIAHCorpus(NIAHConfig(seed=42, haystack_sentences=100))
        corpus2 = NIAHCorpus(NIAHConfig(seed=42, haystack_sentences=100))

        samples1 = corpus1.generate(5)
        samples2 = corpus2.generate(5)

        assert len(samples1) == len(samples2) == 5
        for s1, s2 in zip(samples1, samples2, strict=True):
            assert s1.text == s2.text
            assert s1.question == s2.question
            assert s1.expected_answer == s2.expected_answer
            assert s1.needle_positions == s2.needle_positions

    def test_different_seeds_produce_different_output(self):
        """Test that different seeds produce different output."""
        corpus1 = NIAHCorpus(NIAHConfig(seed=42, haystack_sentences=100))
        corpus2 = NIAHCorpus(NIAHConfig(seed=43, haystack_sentences=100))

        samples1 = corpus1.generate(1)
        samples2 = corpus2.generate(1)

        # Texts should be different (or at least positions)
        assert (
            samples1[0].text != samples2[0].text
            or samples1[0].needle_positions != samples2[0].needle_positions
        )

    def test_generate_single(self):
        """Test generating a single sample."""
        corpus = NIAHCorpus(NIAHConfig(seed=42, haystack_sentences=50))
        sample = corpus.generate_single()

        assert isinstance(sample, NIAHSample)
        assert len(sample.text) > 0
        assert len(sample.question) > 0
        assert len(sample.expected_answer) > 0
        assert len(sample.needles) > 0

    def test_generate_multiple(self):
        """Test generating multiple samples."""
        corpus = NIAHCorpus(NIAHConfig(seed=42, haystack_sentences=50))
        samples = corpus.generate(3)

        assert len(samples) == 3
        for sample in samples:
            assert isinstance(sample, NIAHSample)
            assert len(sample.text) > 0

    def test_needle_in_text(self):
        """Test that needle is actually in the text."""
        corpus = NIAHCorpus(NIAHConfig(seed=42, haystack_sentences=50, needle_count=1))
        sample = corpus.generate_single()

        # The expected answer should be findable in the text
        needle_text = sample.needles[0]["text"]
        assert needle_text in sample.text

    def test_multiple_needles(self):
        """Test generating with multiple needles."""
        corpus = NIAHCorpus(NIAHConfig(seed=42, haystack_sentences=100, needle_count=3))
        sample = corpus.generate_single()

        assert len(sample.needles) == 3
        for needle in sample.needles:
            assert needle["text"] in sample.text

    def test_code_format(self):
        """Test that generated codes have correct format."""
        corpus = NIAHCorpus(NIAHConfig(seed=42, code_length=8))
        sample = corpus.generate_single()

        # Code should be uppercase alphanumeric
        code = sample.expected_answer
        assert len(code) == 8
        assert code.isalnum()
        assert code.isupper()

    def test_different_needle_positions(self):
        """Test different needle position settings."""
        for position in ["start", "middle", "end", "random"]:
            corpus = NIAHCorpus(NIAHConfig(
                seed=42,
                haystack_sentences=100,
                needle_position=position
            ))
            sample = corpus.generate_single()
            assert len(sample.text) > 0

    def test_text_contains_context(self):
        """Test that generated text contains meaningful context."""
        corpus = NIAHCorpus(NIAHConfig(seed=42, haystack_sentences=50))
        sample = corpus.generate_single()

        # Should contain some of the haystack sentences
        assert "The quick brown fox" in sample.text or "In 1492" in sample.text
