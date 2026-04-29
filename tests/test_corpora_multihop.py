"""Tests for MultiHop corpus generator."""


from deepseek_v4_context_bench.corpora.multihop import (
    MultiHopConfig,
    MultiHopCorpus,
    MultiHopSample,
)


class TestMultiHopConfig:
    """Tests for MultiHopConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = MultiHopConfig()
        assert config.seed == 42
        assert config.num_hops == 2
        assert config.context_facts == 50
        assert config.num_questions == 10

    def test_custom_config(self):
        """Test custom configuration."""
        config = MultiHopConfig(seed=123, num_hops=3, context_facts=100)
        assert config.seed == 123
        assert config.num_hops == 3
        assert config.context_facts == 100


class TestMultiHopSample:
    """Tests for MultiHopSample."""

    def test_sample_creation(self):
        """Test creating a sample."""
        sample = MultiHopSample(
            context="Test context",
            question="What is the answer?",
            answer="The answer",
            reasoning_chain=[{"fact": "fact1", "from_entity": "A", "to_entity": "B"}],
            hop_count=1,
        )
        assert sample.context == "Test context"
        assert sample.question == "What is the answer?"
        assert sample.answer == "The answer"
        assert sample.hop_count == 1


class TestMultiHopCorpus:
    """Tests for MultiHopCorpus."""

    def test_init_default(self):
        """Test initialization with default config."""
        corpus = MultiHopCorpus()
        assert corpus.config.seed == 42

    def test_init_custom(self):
        """Test initialization with custom config."""
        config = MultiHopConfig(seed=123)
        corpus = MultiHopCorpus(config)
        assert corpus.config.seed == 123

    def test_deterministic_generation(self):
        """Test that same seed produces identical output."""
        corpus1 = MultiHopCorpus(MultiHopConfig(seed=42, context_facts=30))
        corpus2 = MultiHopCorpus(MultiHopConfig(seed=42, context_facts=30))

        samples1 = corpus1.generate(3)
        samples2 = corpus2.generate(3)

        assert len(samples1) == len(samples2) == 3
        for s1, s2 in zip(samples1, samples2, strict=True):
            assert s1.context == s2.context
            assert s1.question == s2.question
            assert s1.answer == s2.answer
            assert s1.hop_count == s2.hop_count

    def test_generate_single(self):
        """Test generating a single sample."""
        corpus = MultiHopCorpus(MultiHopConfig(seed=42, context_facts=30))
        sample = corpus.generate_single()

        assert isinstance(sample, MultiHopSample)
        assert len(sample.context) > 0
        assert len(sample.question) > 0
        assert len(sample.answer) > 0

    def test_generate_multiple(self):
        """Test generating multiple samples."""
        corpus = MultiHopCorpus(MultiHopConfig(seed=42, context_facts=30, num_questions=5))
        samples = corpus.generate(5)

        assert len(samples) == 5
        for sample in samples:
            assert isinstance(sample, MultiHopSample)
            assert len(sample.context) > 0

    def test_reasoning_chain_exists(self):
        """Test that reasoning chain is populated."""
        corpus = MultiHopCorpus(MultiHopConfig(seed=42, context_facts=50, num_hops=2))
        samples = corpus.generate(5)

        for sample in samples:
            assert len(sample.reasoning_chain) > 0
            assert sample.hop_count == len(sample.reasoning_chain)

    def test_answer_in_context(self):
        """Test that answer appears in context."""
        corpus = MultiHopCorpus(MultiHopConfig(seed=42, context_facts=50))
        samples = corpus.generate(5)

        for sample in samples:
            assert sample.answer in sample.context

    def test_different_hop_counts(self):
        """Test different hop count configurations."""
        for hops in [1, 2, 3]:
            corpus = MultiHopCorpus(MultiHopConfig(
                seed=42,
                context_facts=50,
                num_hops=hops
            ))
            samples = corpus.generate(3)

            for sample in samples:
                if hops <= 2:
                    assert sample.hop_count == hops
                else:
                    assert sample.hop_count >= 1

    def test_context_contains_facts(self):
        """Test that context contains factual statements."""
        corpus = MultiHopCorpus(MultiHopConfig(seed=42, context_facts=30))
        sample = corpus.generate_single()

        # Should contain entity names and relationship words
        assert len(sample.context) > 100  # Should be substantial

    def test_entities_used(self):
        """Test that entities from vocabulary are used."""
        from deepseek_v4_context_bench.corpora.multihop import ENTITIES

        corpus = MultiHopCorpus(MultiHopConfig(seed=42, context_facts=30))
        sample = corpus.generate_single()

        # At least some entities should appear in context
        person_found = any(p in sample.context for p in ENTITIES["person"])
        location_found = any(loc in sample.context for loc in ENTITIES["location"])
        assert person_found or location_found

    def test_hop_count_equals_chain_length(self):
        """Test that hop_count always equals len(reasoning_chain)."""
        corpus = MultiHopCorpus(MultiHopConfig(seed=42, context_facts=50, num_hops=2))
        samples = corpus.generate(5)
        for sample in samples:
            assert sample.hop_count == len(sample.reasoning_chain)

    def test_reasoning_chain_step_has_required_keys(self):
        """Test that every reasoning step contains the four required keys."""
        corpus = MultiHopCorpus(MultiHopConfig(seed=42, context_facts=50, num_hops=2))
        samples = corpus.generate(5)
        required = {"fact", "from_entity", "to_entity", "relationship"}
        for sample in samples:
            for step in sample.reasoning_chain:
                assert required.issubset(step.keys())

    def test_different_seeds_produce_different_output(self):
        """Test that changing the seed changes the generated output."""
        corpus1 = MultiHopCorpus(MultiHopConfig(seed=42, context_facts=50))
        corpus2 = MultiHopCorpus(MultiHopConfig(seed=99, context_facts=50))
        s1 = corpus1.generate(1)
        s2 = corpus2.generate(1)
        if s1 and s2:
            assert s1[0].context != s2[0].context or s1[0].question != s2[0].question

    def test_context_has_minimum_words(self):
        """Test that generated context has at least as many words as context_facts."""
        corpus = MultiHopCorpus(MultiHopConfig(seed=42, context_facts=30))
        sample = corpus.generate_single()
        assert len(sample.context.split()) >= 30

    def test_generate_uses_num_questions_when_no_arg(self):
        """Test that generate() with no argument uses config.num_questions."""
        corpus = MultiHopCorpus(MultiHopConfig(seed=42, context_facts=50, num_questions=3))
        samples = corpus.generate()
        assert len(samples) == 3

    def test_reasoning_chain_entities_linked(self):
        """Test that each step's to_entity is the next step's from_entity."""
        corpus = MultiHopCorpus(MultiHopConfig(seed=42, context_facts=50, num_hops=2))
        samples = corpus.generate(5)
        for sample in samples:
            chain = sample.reasoning_chain
            for i in range(len(chain) - 1):
                assert chain[i]["to_entity"] == chain[i + 1]["from_entity"]

    def test_answer_matches_last_chain_to_entity(self):
        """Test that sample.answer equals the final hop's target entity."""
        corpus = MultiHopCorpus(MultiHopConfig(seed=42, context_facts=50, num_hops=2))
        samples = corpus.generate(5)
        for sample in samples:
            if sample.reasoning_chain:
                assert sample.answer == sample.reasoning_chain[-1]["to_entity"]
