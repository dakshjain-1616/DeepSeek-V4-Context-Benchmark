"""Tests for runner module."""

from pathlib import Path

import pytest

from deepseek_v4_context_bench.config import BenchmarkConfig, ModelProvider
from deepseek_v4_context_bench.runner import (
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSummary,
    BenchmarkTask,
    BudgetEstimator,
    PreflightChecker,
)


class TestBudgetEstimator:
    """Tests for BudgetEstimator."""

    @pytest.fixture
    def estimator(self):
        """Create budget estimator."""
        config = BenchmarkConfig(max_budget_usd=50.0)
        return BudgetEstimator(config)

    def test_estimate_task_cost(self, estimator):
        """Test task cost estimation."""
        cost = estimator.estimate_task_cost(
            ModelProvider.DEEPSEEK_FLASH,
            1_000_000,
            1_000,
        )
        # $0.14 per 1M input + $0.28 per 1M output
        expected = 0.14 + 0.00028
        assert cost == pytest.approx(expected, rel=1e-6)

    def test_estimate_benchmark_cost(self, estimator):
        """Test benchmark cost estimation."""
        estimate = estimator.estimate_benchmark_cost(
            ModelProvider.DEEPSEEK_FLASH,
            10,
            100_000,
        )
        assert "per_task_cost" in estimate
        assert "total_cost" in estimate
        assert "max_budget" in estimate
        assert "within_budget" in estimate
        assert estimate["max_budget"] == 50.0


class TestBenchmarkTask:
    """Tests for BenchmarkTask dataclass."""

    def test_creation(self):
        """Test creating a task."""
        task = BenchmarkTask(
            task_id="test_1",
            corpus_type="niah",
            model="test-model",
            context="test context",
            question="test question?",
            expected_answer="answer",
        )
        assert task.task_id == "test_1"
        assert task.corpus_type == "niah"
        assert task.model == "test-model"


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_creation(self):
        """Test creating a result."""
        result = BenchmarkResult(
            task_id="test_1",
            corpus_type="niah",
            model="test-model",
            prediction="pred",
            expected_answer="answer",
            score_result=None,
            latency_ms=100.0,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            timestamp="2024-01-01T00:00:00",
        )
        assert result.task_id == "test_1"
        assert result.prediction == "pred"
        assert result.error is None


class TestBenchmarkSummary:
    """Tests for BenchmarkSummary dataclass."""

    def test_creation(self):
        """Test creating a summary."""
        summary = BenchmarkSummary(
            model="test-model",
            corpus_type="niah",
            total_tasks=10,
            completed_tasks=8,
            failed_tasks=2,
            accuracy=0.8,
            avg_latency_ms=100.0,
            total_tokens=1000,
            estimated_cost_usd=0.5,
        )
        assert summary.model == "test-model"
        assert summary.accuracy == 0.8
        assert summary.total_tasks == 10


class TestPreflightChecker:
    """Tests for PreflightChecker."""

    @pytest.fixture
    def checker(self):
        """Create preflight checker."""
        config = BenchmarkConfig(dry_run=True)
        return PreflightChecker(config)

    def test_check_budget_dry_run(self, checker):
        """Test budget check in dry run mode."""
        passed, message = checker.check_budget(1000.0)
        assert passed is True
        assert "dry run" in message.lower()

    def test_check_budget_within_limit(self):
        """Test budget check within limit."""
        config = BenchmarkConfig(max_budget_usd=100.0, dry_run=False)
        checker = PreflightChecker(config)
        passed, message = checker.check_budget(50.0)
        assert passed is True
        assert "within budget" in message.lower()

    def test_check_budget_exceeds_limit(self):
        """Test budget check exceeds limit."""
        config = BenchmarkConfig(max_budget_usd=10.0, dry_run=False)
        checker = PreflightChecker(config)
        passed, message = checker.check_budget(50.0)
        assert passed is False
        assert "exceeds" in message.lower()

    def test_check_output_dir(self, checker, tmp_path):
        """Test output directory check."""
        checker.config.output_dir = str(tmp_path)
        passed, message = checker.check_output_dir()
        assert passed is True


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner."""

    @pytest.fixture
    def runner(self):
        """Create benchmark runner."""
        from deepseek_v4_context_bench.client import OpenRouterClient
        from deepseek_v4_context_bench.scorer import ExactMatchScorer

        config = BenchmarkConfig(dry_run=True)
        client = OpenRouterClient(api_key="sk-test")
        scorer = ExactMatchScorer()
        return BenchmarkRunner(config, client, scorer)

    @pytest.mark.asyncio
    async def test_run_task_dry_run(self, runner):
        """Test running task in dry run mode."""
        task = BenchmarkTask(
            task_id="test_1",
            corpus_type="niah",
            model="test-model",
            context="context",
            question="question?",
            expected_answer="answer",
        )
        result = await runner.run_task(task)
        assert result.task_id == "test_1"
        assert result.error is None
        assert "DRY RUN" in result.prediction

    def test_calculate_summary(self, runner):
        """Test summary calculation."""
        results = [
            BenchmarkResult(
                task_id="1",
                corpus_type="niah",
                model="test",
                prediction="answer",
                expected_answer="answer",
                score_result=None,
                latency_ms=100.0,
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                timestamp="2024-01-01T00:00:00",
            ),
            BenchmarkResult(
                task_id="2",
                corpus_type="niah",
                model="test",
                prediction="wrong",
                expected_answer="answer",
                score_result=None,
                latency_ms=200.0,
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                timestamp="2024-01-01T00:00:00",
            ),
        ]
        summary = runner.calculate_summary("test", "niah", results)
        assert summary.total_tasks == 2
        assert summary.completed_tasks == 2
        assert summary.failed_tasks == 0

    def test_save_results(self, runner, tmp_path):
        """Test saving results."""
        summary = BenchmarkSummary(
            model="test-model",
            corpus_type="niah",
            total_tasks=1,
            completed_tasks=1,
            failed_tasks=0,
            accuracy=1.0,
            avg_latency_ms=100.0,
            total_tokens=100,
            estimated_cost_usd=0.1,
        )
        output_path = tmp_path / "test_results.json"
        saved_path = runner.save_results(summary, str(output_path))
        assert Path(saved_path).exists()
