"""Benchmark runner with budget estimation, pre-flight checks, and orchestration.

This module provides the main benchmark execution logic including
budget estimation, pre-flight validation, and result orchestration.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table

if TYPE_CHECKING:
    from .client import OpenRouterClient
    from .config import BenchmarkConfig, ModelProvider
    from .scorer import BaseScorer


@dataclass
class BenchmarkTask:
    """A single benchmark task."""

    task_id: str
    corpus_type: str
    model: str
    context: str
    question: str
    expected_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark task."""

    task_id: str
    corpus_type: str
    model: str
    prediction: str
    expected_answer: str
    score_result: Any  # ScoreResult
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    timestamp: str
    error: str | None = None


@dataclass
class BenchmarkSummary:
    """Summary of benchmark results."""

    model: str
    corpus_type: str
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    accuracy: float
    avg_latency_ms: float
    total_tokens: int
    estimated_cost_usd: float
    results: list[BenchmarkResult] = field(default_factory=list)


class BudgetEstimator:
    """Estimates benchmark costs before execution."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize budget estimator.

        Args:
            config: Benchmark configuration.
        """
        self.config = config

    def estimate_task_cost(
        self,
        model: ModelProvider,
        input_tokens: int,
        output_tokens: int | None = None,
    ) -> float:
        """Estimate cost for a single task.

        Args:
            model: Model provider.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens (uses config default if None).

        Returns:
            Estimated cost in USD.
        """
        if output_tokens is None:
            output_tokens = self.config.output_tokens
        return self.config.estimate_cost(model, input_tokens, output_tokens)

    def estimate_benchmark_cost(
        self,
        model: ModelProvider,
        num_tasks: int,
        avg_input_tokens: int,
    ) -> dict[str, float]:
        """Estimate total benchmark cost.

        Args:
            model: Model provider.
            num_tasks: Number of tasks.
            avg_input_tokens: Average input tokens per task.

        Returns:
            Dictionary with cost breakdown.
        """
        task_cost = self.estimate_task_cost(model, avg_input_tokens)
        total_cost = task_cost * num_tasks

        return {
            "per_task_cost": task_cost,
            "total_cost": total_cost,
            "max_budget": self.config.max_budget_usd,
            "within_budget": (
                total_cost <= self.config.max_budget_usd
                or self.config.max_budget_usd == 0
            ),
        }


class PreflightChecker:
    """Pre-flight checks before running benchmark."""

    def __init__(self, config: BenchmarkConfig, console: Console | None = None):
        """Initialize pre-flight checker.

        Args:
            config: Benchmark configuration.
            console: Rich console for output.
        """
        self.config = config
        self.console = console or Console()

    async def check_api_key(self, client: OpenRouterClient) -> tuple[bool, str]:
        """Check if API key is valid.

        Args:
            client: OpenRouter client.

        Returns:
            Tuple of (passed, message).
        """
        if self.config.dry_run:
            return True, "Dry run mode - API key check skipped"

        if not self.config.openrouter_api_key:
            return False, "OpenRouter API key not configured"

        try:
            is_valid = await client.validate_api_key()
            if is_valid:
                return True, "API key is valid"
            else:
                return False, "API key validation failed"
        except Exception as e:
            return False, f"API key check failed: {e}"

    def check_budget(self, estimated_cost: float) -> tuple[bool, str]:
        """Check if estimated cost is within budget.

        Args:
            estimated_cost: Estimated total cost.

        Returns:
            Tuple of (passed, message).
        """
        if self.config.dry_run:
            return True, "Dry run mode - budget check skipped"

        if self.config.max_budget_usd == 0:
            return True, "No budget limit set"

        if estimated_cost > self.config.max_budget_usd:
            return (
                False,
                f"Estimated cost (${estimated_cost:.2f}) exceeds budget "
                f"(${self.config.max_budget_usd:.2f})"
            )

        return True, f"Estimated cost (${estimated_cost:.2f}) within budget"

    def check_output_dir(self) -> tuple[bool, str]:
        """Check if output directory is writable.

        Returns:
            Tuple of (passed, message).
        """
        output_path = Path(self.config.output_dir)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            test_file = output_path / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
            return True, f"Output directory '{output_path}' is writable"
        except Exception as e:
            return False, f"Output directory check failed: {e}"

    async def run_all_checks(
        self,
        client: OpenRouterClient,
        estimated_cost: float,
    ) -> tuple[bool, list[tuple[bool, str]]]:
        """Run all pre-flight checks.

        Args:
            client: OpenRouter client.
            estimated_cost: Estimated total cost.

        Returns:
            Tuple of (all_passed, list of (passed, message) tuples).
        """
        checks = []

        # API key check
        api_ok, api_msg = await self.check_api_key(client)
        checks.append((api_ok, api_msg))

        # Budget check
        budget_ok, budget_msg = self.check_budget(estimated_cost)
        checks.append((budget_ok, budget_msg))

        # Output directory check
        dir_ok, dir_msg = self.check_output_dir()
        checks.append((dir_ok, dir_msg))

        all_passed = all(c[0] for c in checks)
        return all_passed, checks


class BenchmarkRunner:
    """Main benchmark runner with orchestration logic."""

    def __init__(
        self,
        config: BenchmarkConfig,
        client: OpenRouterClient,
        scorer: BaseScorer,
        console: Console | None = None,
    ):
        """Initialize benchmark runner.

        Args:
            config: Benchmark configuration.
            client: OpenRouter client.
            scorer: Scorer for evaluating results.
            console: Rich console for output.
        """
        self.config = config
        self.client = client
        self.scorer = scorer
        self.console = console or Console()
        self.budget_estimator = BudgetEstimator(config)
        self.preflight = PreflightChecker(config, console)

    async def run_task(self, task: BenchmarkTask) -> BenchmarkResult:
        """Run a single benchmark task.

        Args:
            task: The task to run.

        Returns:
            BenchmarkResult with results.
        """
        if self.config.dry_run:
            # Return mock result in dry-run mode
            return BenchmarkResult(
                task_id=task.task_id,
                corpus_type=task.corpus_type,
                model=task.model,
                prediction="[DRY RUN] Mock prediction",
                expected_answer=task.expected_answer,
                score_result=None,
                latency_ms=0.0,
                prompt_tokens=len(task.context) // 4,  # Rough estimate
                completion_tokens=100,
                total_tokens=len(task.context) // 4 + 100,
                timestamp=datetime.now().isoformat(),
                error=None,
            )

        try:
            # Build messages
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Answer the question "
                        "based on the provided context."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{task.context}\n\nQuestion: {task.question}",
                },
            ]

            # Call API
            result = await self.client.create_completion(
                model=task.model,
                messages=messages,
                max_tokens=self.config.output_tokens,
                temperature=self.config.temperature,
            )

            # Score result
            score_result = await self.scorer.score(
                prediction=result.content,
                reference=task.expected_answer,
                context=task.context,
            )

            return BenchmarkResult(
                task_id=task.task_id,
                corpus_type=task.corpus_type,
                model=task.model,
                prediction=result.content,
                expected_answer=task.expected_answer,
                score_result=score_result,
                latency_ms=result.latency_ms,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.total_tokens,
                timestamp=datetime.now().isoformat(),
                error=None,
            )

        except Exception as e:
            return BenchmarkResult(
                task_id=task.task_id,
                corpus_type=task.corpus_type,
                model=task.model,
                prediction="",
                expected_answer=task.expected_answer,
                score_result=None,
                latency_ms=0.0,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                timestamp=datetime.now().isoformat(),
                error=str(e),
            )

    async def run_benchmark(
        self,
        tasks: list[BenchmarkTask],
        progress: Progress | None = None,
    ) -> list[BenchmarkResult]:
        """Run a batch of benchmark tasks.

        Args:
            tasks: List of tasks to run.
            progress: Optional progress bar.

        Returns:
            List of BenchmarkResult objects.
        """
        results = []
        task_id: TaskID | None = None

        if progress:
            task_id = progress.add_task("Running benchmark...", total=len(tasks))

        for task in tasks:
            result = await self.run_task(task)
            results.append(result)

            if progress and task_id is not None:
                progress.advance(task_id)

        return results

    def calculate_summary(
        self,
        model: str,
        corpus_type: str,
        results: list[BenchmarkResult],
    ) -> BenchmarkSummary:
        """Calculate summary statistics from results.

        Args:
            model: Model name.
            corpus_type: Corpus type.
            results: List of results.

        Returns:
            BenchmarkSummary with statistics.
        """
        total = len(results)
        completed = sum(1 for r in results if r.error is None)
        failed = total - completed

        correct = sum(
            1 for r in results
            if r.error is None and r.score_result and r.score_result.correct
        )
        accuracy = correct / completed if completed > 0 else 0.0

        latencies = [r.latency_ms for r in results if r.error is None]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        total_tokens = sum(r.total_tokens for r in results)

        # Estimate cost
        from .config import ModelProvider
        try:
            model_provider = ModelProvider(model)
        except ValueError:
            model_provider = ModelProvider.DEEPSEEK_FLASH  # Default

        estimated_cost = self.budget_estimator.estimate_task_cost(
            model_provider,
            total_tokens,
        )

        return BenchmarkSummary(
            model=model,
            corpus_type=corpus_type,
            total_tasks=total,
            completed_tasks=completed,
            failed_tasks=failed,
            accuracy=accuracy,
            avg_latency_ms=avg_latency,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost,
            results=results,
        )

    def save_results(
        self,
        summary: BenchmarkSummary,
        output_path: str | None = None,
    ) -> str:
        """Save benchmark results to file.

        Args:
            summary: Benchmark summary.
            output_path: Output file path (uses config default if None).

        Returns:
            Path to saved file.
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                self.config.output_dir,
                f"benchmark_{summary.model.replace('/', '_')}_{timestamp}.json"
            )

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        data = {
            "model": summary.model,
            "corpus_type": summary.corpus_type,
            "timestamp": datetime.now().isoformat(),
            "statistics": {
                "total_tasks": summary.total_tasks,
                "completed_tasks": summary.completed_tasks,
                "failed_tasks": summary.failed_tasks,
                "accuracy": summary.accuracy,
                "avg_latency_ms": summary.avg_latency_ms,
                "total_tokens": summary.total_tokens,
                "estimated_cost_usd": summary.estimated_cost_usd,
            },
            "results": [
                {
                    "task_id": r.task_id,
                    "corpus_type": r.corpus_type,
                    "model": r.model,
                    "prediction": r.prediction,
                    "expected_answer": r.expected_answer,
                    "score": r.score_result.score if r.score_result else 0.0,
                    "correct": r.score_result.correct if r.score_result else False,
                    "latency_ms": r.latency_ms,
                    "total_tokens": r.total_tokens,
                    "error": r.error,
                }
                for r in summary.results
            ],
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        return str(output_file)

    def display_summary(self, summary: BenchmarkSummary) -> None:
        """Display benchmark summary in console.

        Args:
            summary: Benchmark summary.
        """
        table = Table(title=f"Benchmark Results: {summary.model}")

        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Corpus Type", summary.corpus_type)
        table.add_row("Total Tasks", str(summary.total_tasks))
        table.add_row("Completed", str(summary.completed_tasks))
        table.add_row("Failed", str(summary.failed_tasks))
        table.add_row("Accuracy", f"{summary.accuracy:.2%}")
        table.add_row("Avg Latency", f"{summary.avg_latency_ms:.2f} ms")
        table.add_row("Total Tokens", f"{summary.total_tokens:,}")
        table.add_row("Est. Cost", f"${summary.estimated_cost_usd:.4f}")

        self.console.print(table)
