"""Command-line interface for DeepSeek V4 Context Benchmark.

Provides Click commands for running benchmarks, generating reports,
and creating dataset cards.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .client import OpenRouterClient
from .config import BenchmarkConfig, ModelProvider, config
from .corpora import (
    CodebaseConfig,
    CodebaseCorpus,
    MultiHopConfig,
    MultiHopCorpus,
    NIAHConfig,
    NIAHCorpus,
    SynthesisConfig,
    SynthesisCorpus,
)
from .runner import BenchmarkResult, BenchmarkRunner, BenchmarkTask
from .scorer import ContainsMatchScorer, ExactMatchScorer

console = Console()


def get_version() -> str:
    """Get package version."""
    return "0.1.0"


@click.group()
@click.version_option(version=get_version(), prog_name="dsv4ctx")
@click.option(
    "--config-file",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx: click.Context, config_file: str | None, verbose: bool) -> None:
    """DeepSeek V4 Context Benchmark CLI.

    A production-ready 1M-token context benchmark for DeepSeek V4.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["config_file"] = config_file

    if verbose:
        console.print("[blue]Verbose mode enabled[/blue]")


@cli.command()
@click.option(
    "--model",
    "-m",
    type=click.Choice([m.value for m in ModelProvider]),
    required=True,
    help="Model to benchmark",
)
@click.option(
    "--corpus",
    "-c",
    type=click.Choice(["niah", "multihop", "codebase", "synthesis", "all"]),
    default="all",
    help="Corpus type to use",
)
@click.option(
    "--tasks",
    "-n",
    type=int,
    default=10,
    help="Number of tasks to run",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Run in dry-run mode (no API calls)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path",
)
@click.option(
    "--max-tokens",
    type=int,
    default=100000,
    help="Maximum context tokens",
)
@click.option(
    "--scorer",
    type=click.Choice(["contains", "exact"]),
    default="contains",
    help="Scoring rubric. 'contains' (default) credits answers that include the needle "
         "anywhere in the response — suitable for NIAH/multihop where models wrap "
         "answers in prose. 'exact' requires the response to equal the reference.",
)
@click.pass_context
def run(
    ctx: click.Context,
    model: str,
    corpus: str,
    tasks: int,
    dry_run: bool,
    output: str | None,
    max_tokens: int,
    scorer: str,  # noqa: ARG001 — re-bound below to a scorer instance
) -> None:
    """Run benchmark on specified model and corpus."""
    verbose = ctx.obj.get("verbose", False)

    # Display header
    console.print(Panel.fit(
        f"[bold green]DeepSeek V4 Context Benchmark[/bold green]\n"
        f"Model: [cyan]{model}[/cyan] | "
        f"Corpus: [cyan]{corpus}[/cyan] | "
        f"Tasks: [cyan]{tasks}[/cyan]"
    ))

    # Create config
    bench_config = BenchmarkConfig(
        dry_run=dry_run,
        max_tokens=max_tokens,
        verbose=verbose,
    )

    # Create client
    client = OpenRouterClient(
        api_key=bench_config.openrouter_api_key,
        base_url=bench_config.openrouter_base_url,
        timeout=bench_config.openrouter_timeout,
        max_retries=bench_config.max_retries,
    )

    # Create scorer
    scorer_instance: ContainsMatchScorer | ExactMatchScorer = (
        ContainsMatchScorer() if scorer == "contains" else ExactMatchScorer()
    )

    # Create runner
    runner = BenchmarkRunner(bench_config, client, scorer_instance, console)

    # Generate tasks based on corpus type
    benchmark_tasks = []
    corpus_types = ["niah", "multihop", "codebase", "synthesis"] if corpus == "all" else [corpus]

    for corpus_type in corpus_types:
        if corpus_type == "niah":
            niah_corpus = NIAHCorpus(NIAHConfig(seed=42, haystack_sentences=max_tokens // 10))
            niah_samples = niah_corpus.generate(tasks)
            for i, niah_sample in enumerate(niah_samples):
                benchmark_tasks.append(BenchmarkTask(
                    task_id=f"niah_{i}",
                    corpus_type="niah",
                    model=model,
                    context=niah_sample.text,
                    question=niah_sample.question,
                    expected_answer=niah_sample.expected_answer,
                    metadata={"needle_positions": niah_sample.needle_positions},
                ))
        elif corpus_type == "multihop":
            multihop_corpus = MultiHopCorpus(MultiHopConfig(seed=42))
            multihop_samples = multihop_corpus.generate(tasks)
            for i, multihop_sample in enumerate(multihop_samples):
                benchmark_tasks.append(BenchmarkTask(
                    task_id=f"multihop_{i}",
                    corpus_type="multihop",
                    model=model,
                    context=multihop_sample.context,
                    question=multihop_sample.question,
                    expected_answer=multihop_sample.answer,
                    metadata={"hop_count": multihop_sample.hop_count},
                ))
        elif corpus_type == "codebase":
            codebase_corpus = CodebaseCorpus(CodebaseConfig(seed=42))
            codebase_samples = codebase_corpus.generate(tasks)
            for i, codebase_sample in enumerate(codebase_samples):
                benchmark_tasks.append(BenchmarkTask(
                    task_id=f"codebase_{i}",
                    corpus_type="codebase",
                    model=model,
                    context=codebase_sample.code,
                    question=codebase_sample.question,
                    expected_answer=codebase_sample.expected_answer,
                    metadata={"language": codebase_sample.language},
                ))
        elif corpus_type == "synthesis":
            synthesis_corpus = SynthesisCorpus(SynthesisConfig(seed=42))
            synthesis_samples = synthesis_corpus.generate(tasks)
            for i, synthesis_sample in enumerate(synthesis_samples):
                benchmark_tasks.append(BenchmarkTask(
                    task_id=f"synthesis_{i}",
                    corpus_type="synthesis",
                    model=model,
                    context=synthesis_sample.content,
                    question=synthesis_sample.question,
                    expected_answer=synthesis_sample.expected_answer,
                    metadata={"content_type": synthesis_sample.content_type},
                ))

    # Run pre-flight checks
    if not dry_run:
        console.print("\n[yellow]Running pre-flight checks...[/yellow]")
        estimated_cost = runner.budget_estimator.estimate_task_cost(
            ModelProvider(model),
            max_tokens * len(benchmark_tasks),
        )

        async def run_checks() -> tuple[bool, list[tuple[bool, str]]]:
            return await runner.preflight.run_all_checks(client, estimated_cost)

        all_passed, checks = asyncio.run(run_checks())

        for passed, message in checks:
            status = "[green]✓[/green]" if passed else "[red]✗[/red]"
            console.print(f"{status} {message}")

        if not all_passed:
            console.print("\n[red]Pre-flight checks failed. Aborting.[/red]")
            sys.exit(1)

    # Run benchmark
    console.print(f"\n[blue]Running {len(benchmark_tasks)} tasks...[/blue]")

    async def run_benchmark() -> list[BenchmarkResult]:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            return await runner.run_benchmark(benchmark_tasks, progress)

    results = asyncio.run(run_benchmark())

    # Calculate and display summary
    for corpus_type in corpus_types:
        corpus_results = [r for r in results if r.corpus_type == corpus_type]
        if corpus_results:
            summary = runner.calculate_summary(model, corpus_type, corpus_results)
            runner.display_summary(summary)

            # Save results
            output_path = runner.save_results(summary, output)
            console.print(f"\n[green]Results saved to:[/green] {output_path}")

    console.print("\n[bold green]Benchmark complete![/bold green]")


@cli.command()
@click.argument("results_file", type=click.Path(exists=True))
@click.option(
    "--format",
    "-f",
    type=click.Choice(["markdown", "json", "csv"]),
    default="markdown",
    help="Output format",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path",
)
def report(results_file: str, format: str, output: str | None) -> None:
    """Generate report from benchmark results."""
    from .report import generate_report

    console.print(f"[blue]Generating {format} report from {results_file}...[/blue]")

    report_content = generate_report(results_file, format)

    if output:
        Path(output).write_text(report_content)
        console.print(f"[green]Report saved to:[/green] {output}")
    else:
        console.print(report_content)


@cli.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="DATASET_CARD.md",
    help="Output file path",
)
@click.option(
    "--title",
    "-t",
    default="DeepSeek V4 Context Benchmark Dataset",
    help="Dataset title",
)
def card(output: str, title: str) -> None:
    """Generate HuggingFace Dataset Card."""
    from .card import generate_dataset_card

    console.print("[blue]Generating dataset card...[/blue]")

    card_content = generate_dataset_card(title)

    Path(output).write_text(card_content)
    console.print(f"[green]Dataset card saved to:[/green] {output}")


@cli.command()
def models() -> None:
    """List available models and their pricing."""
    table = Table(title="Available Models")

    table.add_column("Model", style="cyan")
    table.add_column("Input ($/1M)", style="green")
    table.add_column("Output ($/1M)", style="green")
    table.add_column("Max Context", style="yellow")

    for model in ModelProvider:
        input_price, output_price = config.get_model_pricing(model)
        max_context = config.get_max_context_length(model)
        table.add_row(
            model.value,
            f"${input_price:.2f}",
            f"${output_price:.2f}",
            f"{max_context:,} tokens",
        )

    console.print(table)


@cli.command()
@click.option(
    "--model",
    "-m",
    type=click.Choice([m.value for m in ModelProvider]),
    required=True,
    help="Model to estimate",
)
@click.option(
    "--tasks",
    "-n",
    type=int,
    default=100,
    help="Number of tasks",
)
@click.option(
    "--tokens",
    "-t",
    type=int,
    default=100_000,
    help="Average tokens per task",
)
def estimate(model: str, tasks: int, tokens: int) -> None:
    """Estimate benchmark cost."""
    from .runner import BudgetEstimator

    bench_config = BenchmarkConfig()
    estimator = BudgetEstimator(bench_config)

    model_provider = ModelProvider(model)
    cost_estimate = estimator.estimate_benchmark_cost(
        model_provider,
        tasks,
        tokens,
    )

    table = Table(title=f"Cost Estimate: {model}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Number of Tasks", str(tasks))
    table.add_row("Avg Tokens per Task", f"{tokens:,}")
    table.add_row("Cost per Task", f"${cost_estimate['per_task_cost']:.4f}")
    table.add_row("Total Estimated Cost", f"${cost_estimate['total_cost']:.2f}")
    budget_limit = (
        f"${cost_estimate['max_budget']:.2f}"
        if cost_estimate['max_budget'] > 0
        else "Unlimited"
    )
    table.add_row("Budget Limit", budget_limit)
    table.add_row("Within Budget", "Yes" if cost_estimate['within_budget'] else "No")

    console.print(table)


def main() -> None:
    """Entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
