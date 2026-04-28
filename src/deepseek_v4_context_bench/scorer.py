"""Scoring module with exact-match and V4-Pro judge rubrics.

This module provides scoring functionality for benchmark results,
including exact-match scoring and judge-based evaluation using
DeepSeek V4-Pro as a judge.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .client import OpenRouterClient


@dataclass
class ScoreResult:
    """Result of scoring a single prediction."""

    score: float  # 0.0 to 1.0
    correct: bool
    method: str
    details: dict[str, Any]
    explanation: str = ""


class BaseScorer(ABC):
    """Base class for all scorers."""

    @abstractmethod
    async def score(
        self,
        prediction: str,
        reference: str,
        context: str | None = None,
    ) -> ScoreResult:
        """Score a prediction against a reference.

        Args:
            prediction: The model's prediction.
            reference: The expected reference answer.
            context: Optional context for scoring.

        Returns:
            ScoreResult with score and metadata.
        """
        pass

    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison.

        Args:
            text: Text to normalize.

        Returns:
            Normalized text.
        """
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = " ".join(text.split())
        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        return text.strip()


class ExactMatchScorer(BaseScorer):
    """Exact match scorer with normalization options."""

    def __init__(self, case_sensitive: bool = False, strip_whitespace: bool = True):
        """Initialize exact match scorer.

        Args:
            case_sensitive: Whether matching is case sensitive.
            strip_whitespace: Whether to strip whitespace before matching.
        """
        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace

    async def score(
        self,
        prediction: str,
        reference: str,
        context: str | None = None,
    ) -> ScoreResult:
        """Score using exact match.

        Args:
            prediction: The model's prediction.
            reference: The expected reference answer.
            context: Optional context (ignored for exact match).

        Returns:
            ScoreResult with binary score.
        """
        pred = prediction
        ref = reference

        if self.strip_whitespace:
            pred = pred.strip()
            ref = ref.strip()

        if not self.case_sensitive:
            pred = pred.lower()
            ref = ref.lower()

        is_correct = pred == ref

        return ScoreResult(
            score=1.0 if is_correct else 0.0,
            correct=is_correct,
            method="exact_match",
            details={
                "prediction_normalized": pred,
                "reference_normalized": ref,
                "case_sensitive": self.case_sensitive,
                "strip_whitespace": self.strip_whitespace,
            },
            explanation="Exact match" if is_correct else "No exact match",
        )


class ContainsMatchScorer(BaseScorer):
    """Scorer that checks if reference is contained in prediction."""

    def __init__(self, case_sensitive: bool = False):
        """Initialize contains match scorer.

        Args:
            case_sensitive: Whether matching is case sensitive.
        """
        self.case_sensitive = case_sensitive

    async def score(
        self,
        prediction: str,
        reference: str,
        context: str | None = None,
    ) -> ScoreResult:
        """Score using contains match.

        Args:
            prediction: The model's prediction.
            reference: The expected reference answer.
            context: Optional context (ignored).

        Returns:
            ScoreResult with binary score.
        """
        pred = prediction
        ref = reference

        if not self.case_sensitive:
            pred = pred.lower()
            ref = ref.lower()

        is_correct = ref in pred

        return ScoreResult(
            score=1.0 if is_correct else 0.0,
            correct=is_correct,
            method="contains_match",
            details={
                "prediction_normalized": pred,
                "reference_normalized": ref,
                "case_sensitive": self.case_sensitive,
            },
            explanation="Reference found in prediction" if is_correct else "Reference not found",
        )


class F1Scorer(BaseScorer):
    """F1 score based on token overlap."""

    def __init__(self, delimiter: str | None = None):
        """Initialize F1 scorer.

        Args:
            delimiter: Delimiter to split tokens. If None, splits on whitespace.
        """
        self.delimiter = delimiter

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text.

        Args:
            text: Text to tokenize.

        Returns:
            List of tokens.
        """
        normalized = self.normalize_text(text)
        if self.delimiter:
            return [t.strip() for t in normalized.split(self.delimiter) if t.strip()]
        return normalized.split()

    async def score(
        self,
        prediction: str,
        reference: str,
        context: str | None = None,
    ) -> ScoreResult:
        """Score using F1 metric.

        Args:
            prediction: The model's prediction.
            reference: The expected reference answer.
            context: Optional context (ignored).

        Returns:
            ScoreResult with F1 score.
        """
        pred_tokens = set(self._tokenize(prediction))
        ref_tokens = set(self._tokenize(reference))

        if not pred_tokens and not ref_tokens:
            # Both empty, perfect match
            return ScoreResult(
                score=1.0,
                correct=True,
                method="f1",
                details={"precision": 1.0, "recall": 1.0, "f1": 1.0},
                explanation="Both prediction and reference are empty",
            )

        if not pred_tokens or not ref_tokens:
            # One is empty, other is not
            return ScoreResult(
                score=0.0,
                correct=False,
                method="f1",
                details={"precision": 0.0, "recall": 0.0, "f1": 0.0},
                explanation="Empty prediction or reference",
            )

        # Calculate F1
        common_tokens = pred_tokens & ref_tokens
        precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0.0
        recall = len(common_tokens) / len(ref_tokens) if ref_tokens else 0.0

        f1 = (
            0.0
            if precision + recall == 0
            else 2 * (precision * recall) / (precision + recall)
        )

        return ScoreResult(
            score=f1,
            correct=f1 >= 0.5,  # Consider correct if F1 >= 0.5
            method="f1",
            details={
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "common_tokens": len(common_tokens),
                "pred_tokens": len(pred_tokens),
                "ref_tokens": len(ref_tokens),
            },
            explanation=f"F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}",
        )


class V4ProJudgeScorer(BaseScorer):
    """Judge-based scorer using DeepSeek V4-Pro.

    Uses V4-Pro as a judge to evaluate answer quality.
    """

    JUDGE_PROMPT = """You are an expert evaluator. Evaluate the following model response \
against the expected answer.

Context: {context}

Expected Answer: {reference}

Model Response: {prediction}

Rate the model response on a scale of 0-10 where:
- 10: Perfect match, completely correct
- 7-9: Mostly correct with minor issues
- 4-6: Partially correct
- 1-3: Mostly incorrect
- 0: Completely wrong or irrelevant

Provide your rating and a brief explanation.

Format your response as:
Score: [0-10]
Explanation: [your reasoning]"""

    def __init__(
        self,
        client: OpenRouterClient,
        model: str = "deepseek/deepseek-v4-pro",
        temperature: float = 0.0,
    ):
        """Initialize V4-Pro judge scorer.

        Args:
            client: OpenRouter client for API calls.
            model: Model to use as judge.
            temperature: Sampling temperature for judge.
        """
        self.client = client
        self.model = model
        self.temperature = temperature

    def _parse_score(self, text: str) -> float:
        """Parse score from judge response.

        Args:
            text: Judge response text.

        Returns:
            Parsed score between 0 and 1.
        """
        # Look for "Score: X" pattern
        match = re.search(r"score[:\s]+(\d+(?:\.\d+)?)", text.lower())
        if match:
            try:
                score = float(match.group(1))
                # Normalize to 0-1 range
                return min(max(score / 10.0, 0.0), 1.0)
            except ValueError:
                pass

        # Fallback: look for any number
        numbers = re.findall(r"\d+(?:\.\d+)?", text)
        if numbers:
            try:
                score = float(numbers[0])
                return min(max(score / 10.0, 0.0), 1.0)
            except ValueError:
                pass

        return 0.0

    async def score(
        self,
        prediction: str,
        reference: str,
        context: str | None = None,
    ) -> ScoreResult:
        """Score using V4-Pro as judge.

        Args:
            prediction: The model's prediction.
            reference: The expected reference answer.
            context: Optional context for the question.

        Returns:
            ScoreResult with judge score.
        """
        context = context or "No context provided"

        prompt = self.JUDGE_PROMPT.format(
            context=context[:2000],  # Limit context length
            reference=reference,
            prediction=prediction,
        )

        try:

            result = await self.client.create_completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=self.temperature,
            )

            judge_response = result.content
            score = self._parse_score(judge_response)

            return ScoreResult(
                score=score,
                correct=score >= 0.7,  # Consider correct if score >= 7/10
                method="v4_pro_judge",
                details={
                    "judge_response": judge_response,
                    "judge_model": self.model,
                },
                explanation=judge_response,
            )

        except Exception as e:
            return ScoreResult(
                score=0.0,
                correct=False,
                method="v4_pro_judge",
                details={"error": str(e)},
                explanation=f"Judge evaluation failed: {e}",
            )


class MultiScorer(BaseScorer):
    """Combines multiple scorers with weighted voting."""

    def __init__(self, scorers: list[tuple[BaseScorer, float]]):
        """Initialize multi-scorer.

        Args:
            scorers: List of (scorer, weight) tuples.
        """
        self.scorers = scorers

    async def score(
        self,
        prediction: str,
        reference: str,
        context: str | None = None,
    ) -> ScoreResult:
        """Score using multiple scorers.

        Args:
            prediction: The model's prediction.
            reference: The expected reference answer.
            context: Optional context.

        Returns:
            ScoreResult with weighted average score.
        """
        scores = []
        total_weight = 0.0
        details = {}

        for scorer, weight in self.scorers:
            result = await scorer.score(prediction, reference, context)
            scores.append(result.score * weight)
            total_weight += weight
            details[result.method] = {
                "score": result.score,
                "correct": result.correct,
                "details": result.details,
            }

        weighted_score = (
            0.0 if total_weight == 0 else sum(scores) / total_weight
        )

        return ScoreResult(
            score=weighted_score,
            correct=weighted_score >= 0.5,
            method="multi_scorer",
            details=details,
            explanation=f"Weighted average score: {weighted_score:.3f}",
        )


def get_scorer(
    scorer_type: str,
    **kwargs: Any,
) -> BaseScorer:
    """Factory function to get a scorer by type.

    Args:
        scorer_type: Type of scorer (exact_match, contains, f1, v4_pro_judge).
        **kwargs: Additional arguments for scorer initialization.

    Returns:
        Configured scorer instance.
    """
    if scorer_type == "exact_match":
        return ExactMatchScorer(**kwargs)
    elif scorer_type == "contains":
        return ContainsMatchScorer(**kwargs)
    elif scorer_type == "f1":
        return F1Scorer(**kwargs)
    elif scorer_type == "v4_pro_judge":
        return V4ProJudgeScorer(**kwargs)
    else:
        raise ValueError(f"Unknown scorer type: {scorer_type}")
