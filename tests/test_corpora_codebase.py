"""Tests for Codebase corpus generator."""


from deepseek_v4_context_bench.corpora.codebase import (
    LANGUAGES,
    CodebaseConfig,
    CodebaseCorpus,
    CodebaseSample,
)


class TestCodebaseConfig:
    """Tests for CodebaseConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = CodebaseConfig()
        assert config.seed == 42
        assert config.language == "python"
        assert config.files_count == 20
        assert config.lines_per_file == 50
        assert config.patterns_per_sample == 3

    def test_custom_config(self):
        """Test custom configuration."""
        config = CodebaseConfig(seed=123, language="javascript", files_count=10)
        assert config.seed == 123
        assert config.language == "javascript"
        assert config.files_count == 10


class TestCodebaseSample:
    """Tests for CodebaseSample."""

    def test_sample_creation(self):
        """Test creating a sample."""
        sample = CodebaseSample(
            code="def test(): pass",
            language="python",
            question="What function?",
            expected_answer="test",
            file_structure=["file1.py"],
            pattern_locations=[{"file": "file1.py", "patterns": [], "positions": []}],
        )
        assert sample.code == "def test(): pass"
        assert sample.language == "python"
        assert sample.question == "What function?"
        assert sample.expected_answer == "test"


class TestCodebaseCorpus:
    """Tests for CodebaseCorpus."""

    def test_init_default(self):
        """Test initialization with default config."""
        corpus = CodebaseCorpus()
        assert corpus.config.language == "python"

    def test_init_custom(self):
        """Test initialization with custom config."""
        config = CodebaseConfig(seed=123, language="javascript")
        corpus = CodebaseCorpus(config)
        assert corpus.config.seed == 123
        assert corpus.config.language == "javascript"

    def test_invalid_language_defaults_to_python(self):
        """Test that invalid language defaults to Python."""
        config = CodebaseConfig(language="invalid_lang")
        corpus = CodebaseCorpus(config)
        assert corpus.config.language == "python"

    def test_deterministic_generation(self):
        """Test that same seed produces identical output."""
        corpus1 = CodebaseCorpus(CodebaseConfig(seed=42, files_count=5))
        corpus2 = CodebaseCorpus(CodebaseConfig(seed=42, files_count=5))

        samples1 = corpus1.generate(3)
        samples2 = corpus2.generate(3)

        assert len(samples1) == len(samples2) == 3
        for s1, s2 in zip(samples1, samples2, strict=True):
            assert s1.code == s2.code
            assert s1.question == s2.question
            assert s1.expected_answer == s2.expected_answer
            assert s1.file_structure == s2.file_structure

    def test_generate_single(self):
        """Test generating a single sample."""
        corpus = CodebaseCorpus(CodebaseConfig(seed=42, files_count=5))
        sample = corpus.generate_single()

        assert isinstance(sample, CodebaseSample)
        assert len(sample.code) > 0
        assert len(sample.question) > 0
        assert len(sample.expected_answer) > 0
        assert len(sample.file_structure) > 0

    def test_generate_multiple(self):
        """Test generating multiple samples."""
        corpus = CodebaseCorpus(CodebaseConfig(seed=42, files_count=5))
        samples = corpus.generate(3)

        assert len(samples) == 3
        for sample in samples:
            assert isinstance(sample, CodebaseSample)
            assert len(sample.code) > 0

    def test_file_structure_generation(self):
        """Test that file structure is generated correctly."""
        corpus = CodebaseCorpus(CodebaseConfig(seed=42, files_count=10))
        sample = corpus.generate_single()

        assert len(sample.file_structure) == 10
        for filename in sample.file_structure:
            assert filename.endswith(".py")

    def test_different_languages(self):
        """Test generation with different languages."""
        for lang in ["python", "javascript", "typescript"]:
            if lang in LANGUAGES:
                corpus = CodebaseCorpus(CodebaseConfig(seed=42, language=lang, files_count=3))
                sample = corpus.generate_single()
                assert sample.language == lang
                assert len(sample.code) > 0

    def test_code_contains_patterns(self):
        """Test that generated code contains named patterns from CODE_PATTERNS."""
        corpus = CodebaseCorpus(CodebaseConfig(
            seed=42,
            language="python",
            files_count=5,
            patterns_per_sample=2
        ))
        sample = corpus.generate_single()

        assert "calculate_sum" in sample.code or "DataProcessor" in sample.code

    def test_pattern_locations_tracked(self):
        """Test that pattern locations are tracked."""
        corpus = CodebaseCorpus(CodebaseConfig(
            seed=42,
            files_count=5,
            patterns_per_sample=2
        ))
        sample = corpus.generate_single()

        # Should have pattern locations
        assert len(sample.pattern_locations) > 0

    def test_file_extensions_correct(self):
        """Test that file extensions match language."""
        test_cases = [
            ("python", ".py"),
            ("javascript", ".js"),
            ("typescript", ".ts"),
            ("java", ".java"),
            ("cpp", ".cpp"),
            ("rust", ".rs"),
        ]

        for lang, ext in test_cases:
            if lang in LANGUAGES:
                corpus = CodebaseCorpus(CodebaseConfig(seed=42, language=lang, files_count=3))
                sample = corpus.generate_single()
                for filename in sample.file_structure:
                    assert filename.endswith(ext)

    def test_language_set_in_sample(self):
        """Test that sample.language matches the configured language."""
        for lang in ["python", "javascript", "typescript"]:
            if lang in LANGUAGES:
                corpus = CodebaseCorpus(CodebaseConfig(seed=42, language=lang, files_count=3))
                sample = corpus.generate_single()
                assert sample.language == lang

    def test_all_standard_languages_in_constant(self):
        """Test that the LANGUAGES constant contains every expected language."""
        for lang in ["python", "javascript", "typescript", "java", "cpp", "rust"]:
            assert lang in LANGUAGES

    def test_file_count_matches_config(self):
        """Test that file_structure length always equals files_count."""
        for count in [3, 5, 10]:
            corpus = CodebaseCorpus(CodebaseConfig(seed=42, files_count=count))
            sample = corpus.generate_single()
            assert len(sample.file_structure) == count

    def test_code_has_file_section_markers(self):
        """Test that combined code string contains per-file headers."""
        corpus = CodebaseCorpus(CodebaseConfig(seed=42, files_count=3))
        sample = corpus.generate_single()
        assert "// File:" in sample.code

    def test_pattern_locations_have_required_keys(self):
        """Test that every pattern_locations entry has file, patterns and positions."""
        corpus = CodebaseCorpus(CodebaseConfig(
            seed=42, files_count=5, patterns_per_sample=2,
        ))
        sample = corpus.generate_single()
        for loc in sample.pattern_locations:
            assert "file" in loc
            assert "patterns" in loc
            assert "positions" in loc

    def test_different_seeds_produce_different_code(self):
        """Test that different seeds produce different generated code."""
        s1 = CodebaseCorpus(CodebaseConfig(seed=42, files_count=3)).generate_single()
        s2 = CodebaseCorpus(CodebaseConfig(seed=99, files_count=3)).generate_single()
        assert s1.code != s2.code
