"""Report generation module for benchmark results.

Generates reports in various formats (Markdown, JSON, CSV) from
benchmark results.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from typing import Any


def generate_markdown_report(data: dict[str, Any]) -> str:
    """Generate Markdown report from benchmark data.

    Args:
        data: Benchmark results data.

    Returns:
        Markdown formatted report.
    """
    stats = data.get("statistics", {})
    results = data.get("results", [])

    lines = [
        "# DeepSeek V4 Context Benchmark Report",
        "",
        f"**Model:** `{data.get('model', 'unknown')}`",
        f"**Corpus Type:** {data.get('corpus_type', 'unknown')}",
        f"**Generated:** {data.get('timestamp', datetime.now().isoformat())}",
        "",
        "## Summary Statistics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Tasks | {stats.get('total_tasks', 0)} |",
        f"| Completed | {stats.get('completed_tasks', 0)} |",
        f"| Failed | {stats.get('failed_tasks', 0)} |",
        f"| Accuracy | {stats.get('accuracy', 0.0):.2%} |",
        f"| Avg Latency | {stats.get('avg_latency_ms', 0.0):.2f} ms |",
        f"| Total Tokens | {stats.get('total_tokens', 0):,} |",
        f"| Est. Cost | ${stats.get('estimated_cost_usd', 0.0):.4f} |",
        "",
        "## Detailed Results",
        "",
    ]

    for i, result in enumerate(results, 1):
        pred = result.get('prediction', 'N/A')
        pred_str = (
            f"- **Prediction:** {pred[:200]}..."
            if len(pred) > 200
            else f"- **Prediction:** {pred}"
        )
        lines.extend([
            f"### Task {i}: {result.get('task_id', 'unknown')}",
            "",
            f"- **Question:** {result.get('expected_answer', 'N/A')}",
            pred_str,
            f"- **Score:** {result.get('score', 0.0):.3f}",
            f"- **Correct:** {'✓' if result.get('correct', False) else '✗'}",
            f"- **Latency:** {result.get('latency_ms', 0.0):.2f} ms",
            f"- **Tokens:** {result.get('total_tokens', 0)}",
        ])
        if result.get('error'):
            lines.append(f"- **Error:** {result['error']}")
        lines.append("")

    return "\n".join(lines)


def generate_csv_report(data: dict[str, Any]) -> str:
    """Generate CSV report from benchmark data.

    Args:
        data: Benchmark results data.

    Returns:
        CSV formatted report.
    """
    import io

    output = io.StringIO()
    results = data.get("results", [])

    if not results:
        return ""

    # Get fieldnames from first result
    fieldnames = list(results[0].keys())
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

    return output.getvalue()


def generate_json_report(data: dict[str, Any], pretty: bool = True) -> str:
    """Generate JSON report from benchmark data.

    Args:
        data: Benchmark results data.
        pretty: Whether to format with indentation.

    Returns:
        JSON formatted report.
    """
    indent = 2 if pretty else None
    return json.dumps(data, indent=indent, ensure_ascii=False)


def generate_report(results_file: str, format: str = "markdown") -> str:
    """Generate report from results file.

    Args:
        results_file: Path to results JSON file.
        format: Output format (markdown, json, csv).

    Returns:
        Formatted report string.
    """
    # Load results
    with open(results_file) as f:
        data = json.load(f)

    if format == "markdown":
        return generate_markdown_report(data)
    elif format == "json":
        return generate_json_report(data)
    elif format == "csv":
        return generate_csv_report(data)
    else:
        raise ValueError(f"Unknown format: {format}")


def generate_comparison_report(
    results_files: list[str],
    output_format: str = "markdown",
) -> str:
    """Generate comparison report from multiple result files.

    Args:
        results_files: List of paths to results JSON files.
        output_format: Output format.

    Returns:
        Formatted comparison report.
    """
    all_data = []
    for f in results_files:
        with open(f) as fp:
            all_data.append(json.load(fp))

    if output_format == "markdown":
        lines = [
            "# DeepSeek V4 Context Benchmark Comparison",
            "",
            "## Model Comparison",
            "",
            "| Model | Corpus | Accuracy | Avg Latency | Total Tokens | Est. Cost |",
            "|-------|--------|----------|-------------|--------------|-----------|",
        ]

        for data in all_data:
            stats = data.get("statistics", {})
            lines.append(
                f"| {data.get('model', 'unknown')} | "
                f"{data.get('corpus_type', 'unknown')} | "
                f"{stats.get('accuracy', 0.0):.2%} | "
                f"{stats.get('avg_latency_ms', 0.0):.2f} ms | "
                f"{stats.get('total_tokens', 0):,} | "
                f"${stats.get('estimated_cost_usd', 0.0):.4f} |"
            )

        return "\n".join(lines)
    else:
        return json.dumps(all_data, indent=2)
