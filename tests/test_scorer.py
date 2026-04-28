"""Tests for scorer module."""

from unittest.mock import AsyncMock

import pytest

from deepseek_v4_context_bench.scorer import (
    ContainsMatchScorer,
    ExactMatchScorer,
    F1Scorer,
    ScoreResult,
    get_scorer,
)


class TestScoreResult:
    """Tests for ScoreResult dataclass."""

    def test_creation(self):
        """Test creating ScoreResult."""
        result = ScoreResult(
            score=0.85,
            correct=True,
            method="test",
            details={"key": "value"},
            explanation="Test explanation",
        )
        assert result.score == 0.85
        assert result.correct is True
        assert result.method == "test"


class TestExactMatchScorer:
    """Tests for ExactMatchScorer."""

    @pytest.fixture
    def scorer(self):
        """Create exact match scorer."""
        return ExactMatchScorer()

    @pytest.mark.asyncio
    async def test_exact_match(self, scorer):
        """Test exact match scoring."""
        result = await scorer.score("hello", "hello")
        assert result.score == 1.0
        assert result.correct is True
        assert result.method == "exact_match"

    @pytest.mark.asyncio
    async def test_no_match(self, scorer):
        """Test non-matching prediction."""
        result = await scorer.score("hello", "world")
        assert result.score == 0.0
        assert result.correct is False

    @pytest.mark.asyncio
    async def test_case_insensitive(self, scorer):
        """Test case insensitive matching."""
        result = await scorer.score("Hello", "hello")
        assert result.score == 1.0
        assert result.correct is True

    @pytest.mark.asyncio
    async def test_whitespace_stripping(self, scorer):
        """Test whitespace stripping."""
        result = await scorer.score("  hello  ", "hello")
        assert result.score == 1.0
        assert result.correct is True

    def test_normalize_text(self, scorer):
        """Test text normalization."""
        normalized = scorer.normalize_text("  Hello, World!  ")
        assert normalized == "hello world"


class TestContainsMatchScorer:
    """Tests for ContainsMatchScorer."""

    @pytest.fixture
    def scorer(self):
        """Create contains match scorer."""
        return ContainsMatchScorer()

    @pytest.mark.asyncio
    async def test_contains_match(self, scorer):
        """Test contains match scoring."""
        result = await scorer.score("the answer is hello world", "hello")
        assert result.score == 1.0
        assert result.correct is True

    @pytest.mark.asyncio
    async def test_not_contains(self, scorer):
        """Test when reference not in prediction."""
        result = await scorer.score("hello world", "goodbye")
        assert result.score == 0.0
        assert result.correct is False

    @pytest.mark.asyncio
    async def test_case_insensitive(self, scorer):
        """Test case insensitive contains."""
        result = await scorer.score("The Answer Is HELLO", "hello")
        assert result.score == 1.0
        assert result.correct is True


class TestF1Scorer:
    """Tests for F1Scorer."""

    @pytest.fixture
    def scorer(self):
        """Create F1 scorer."""
        return F1Scorer()

    @pytest.mark.asyncio
    async def test_perfect_match(self, scorer):
        """Test perfect F1 score."""
        result = await scorer.score("hello world", "hello world")
        assert result.score == 1.0
        assert result.correct is True

    @pytest.mark.asyncio
    async def test_partial_match(self, scorer):
        """Test partial F1 score."""
        result = await scorer.score("hello world test", "hello world")
        assert 0.0 < result.score < 1.0

    @pytest.mark.asyncio
    async def test_no_match(self, scorer):
        """Test zero F1 score."""
        result = await scorer.score("abc def", "ghi jkl")
        assert result.score == 0.0
        assert result.correct is False

    @pytest.mark.asyncio
    async def test_empty_prediction(self, scorer):
        """Test empty prediction."""
        result = await scorer.score("", "hello")
        assert result.score == 0.0
        assert result.correct is False

    @pytest.mark.asyncio
    async def test_empty_reference(self, scorer):
        """Test empty reference."""
        result = await scorer.score("hello", "")
        assert result.score == 0.0
        assert result.correct is False

    @pytest.mark.asyncio
    async def test_both_empty(self, scorer):
        """Test both empty."""
        result = await scorer.score("", "")
        assert result.score == 1.0
        assert result.correct is True

    def test_tokenize(self, scorer):
        """Test tokenization."""
        tokens = scorer._tokenize("Hello, World!")
        assert "hello" in tokens
        assert "world" in tokens


class TestGetScorer:
    """Tests for get_scorer factory function."""

    def test_exact_match_scorer(self):
        """Test getting exact match scorer."""
        scorer = get_scorer("exact_match")
        assert isinstance(scorer, ExactMatchScorer)

    def test_contains_scorer(self):
        """Test getting contains scorer."""
        scorer = get_scorer("contains")
        assert isinstance(scorer, ContainsMatchScorer)

    def test_f1_scorer(self):
        """Test getting F1 scorer."""
        scorer = get_scorer("f1")
        assert isinstance(scorer, F1Scorer)

    def test_invalid_scorer(self):
        """Test invalid scorer type."""
        with pytest.raises(ValueError):
            get_scorer("invalid")


class TestV4ProJudgeScorer:
    """Tests for V4ProJudgeScorer."""

    @pytest.fixture
    def mock_client(self):
        """Create mock OpenRouter client."""
        mock = AsyncMock()
        mock.create_completion = AsyncMock(return_value=AsyncMock(
            content="Score: 8\nExplanation: Good answer"
        ))
        return mock

    @pytest.fixture
    def scorer(self, mock_client):
        """Create V4ProJudge scorer."""
        from deepseek_v4_context_bench.scorer import V4ProJudgeScorer
        return V4ProJudgeScorer(client=mock_client)

    def test_parse_score_with_pattern(self, scorer):
        """Test parsing score from judge response."""
        text = "Score: 8\nExplanation: Good answer"
        score = scorer._parse_score(text)
        assert score == 0.8

    def test_parse_score_fallback_number(self, scorer):
        """Test parsing score fallback to first number."""
        text = "The answer is 7 out of 10"
        score = scorer._parse_score(text)
        assert score == 0.7

    def test_parse_score_no_number(self, scorer):
        """Test parsing score with no number."""
        text = "No score here"
        score = scorer._parse_score(text)
        assert score == 0.0

    def test_parse_score_decimal(self, scorer):
        """Test parsing decimal score."""
        text = "Score: 8.5"
        score = scorer._parse_score(text)
        assert score == 0.85

    @pytest.mark.asyncio
    async def test_score_success(self, scorer, mock_client):
        """Test successful judge scoring."""
        result = await scorer.score("prediction", "reference", "context")
        assert result.score == 0.8
        assert result.correct is True
        assert result.method == "v4_pro_judge"
        mock_client.create_completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_score_low_score(self, scorer, mock_client):
        """Test judge scoring with low score."""
        mock_client.create_completion = AsyncMock(return_value=AsyncMock(
            content="Score: 4\nExplanation: Poor answer"
        ))
        result = await scorer.score("prediction", "reference")
        assert result.score == 0.4
        assert result.correct is False

    @pytest.mark.asyncio
    async def test_score_exception(self, scorer, mock_client):
        """Test judge scoring with exception."""
        mock_client.create_completion = AsyncMock(side_effect=Exception("API error"))
        result = await scorer.score("prediction", "reference")
        assert result.score == 0.0
        assert result.correct is False
        assert "error" in result.details


class TestMultiScorer:
    """Tests for MultiScorer."""

    @pytest.fixture
    def scorer(self):
        """Create multi-scorer."""
        from deepseek_v4_context_bench.scorer import MultiScorer
        em_scorer = ExactMatchScorer()
        contains_scorer = ContainsMatchScorer()
        return MultiScorer([(em_scorer, 0.6), (contains_scorer, 0.4)])

    @pytest.mark.asyncio
    async def test_score_weighted_average(self, scorer):
        """Test weighted average scoring."""
        result = await scorer.score("hello", "hello")
        # Both scorers give 1.0, weighted average is 1.0
        assert result.score == 1.0
        assert result.correct is True
        assert result.method == "multi_scorer"

    @pytest.mark.asyncio
    async def test_score_partial_match(self, scorer):
        """Test partial match with different scores."""
        # "hello world" vs "hello" - exact match fails, contains passes
        result = await scorer.score("hello world", "hello")
        # ExactMatch: 0.0, Contains: 1.0
        # Weighted: 0.0*0.6 + 1.0*0.4 = 0.4
        assert result.score == pytest.approx(0.4, abs=0.01)

    @pytest.mark.asyncio
    async def test_score_zero_weight(self, scorer):
        """Test scoring with zero total weight."""
        from deepseek_v4_context_bench.scorer import MultiScorer
        empty_scorer = MultiScorer([])
        result = await empty_scorer.score("hello", "hello")
        assert result.score == 0.0


class TestBaseScorerNormalize:
    """Tests for BaseScorer.normalize_text."""

    def test_normalize_lowercase(self):
        """Test normalization lowercases text."""
        scorer = ExactMatchScorer()
        result = scorer.normalize_text("HELLO World")
        assert result == "hello world"

    def test_normalize_whitespace(self):
        """Test normalization handles whitespace."""
        scorer = ExactMatchScorer()
        result = scorer.normalize_text("  hello   world  ")
        assert result == "hello world"

    def test_normalize_punctuation(self):
        """Test normalization removes punctuation."""
        scorer = ExactMatchScorer()
        result = scorer.normalize_text("hello, world!")
        assert result == "hello world"
