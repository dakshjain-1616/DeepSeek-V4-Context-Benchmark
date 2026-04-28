"""HuggingFace Dataset Card generator.

Generates standardized HuggingFace Dataset Card markdown
for the benchmark dataset.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


def generate_dataset_card(
    title: str = "DeepSeek V4 Context Benchmark Dataset",
    version: str = "0.1.0",
    **kwargs: Any,
) -> str:
    """Generate HuggingFace Dataset Card content.

    Args:
        title: Dataset title.
        version: Dataset version.
        **kwargs: Additional metadata.

    Returns:
        Dataset card markdown content.
    """
    card = f"""---
language:
- en
license: mit
library_name: deepseek-v4-context-bench
tags:
- deepseek
- llm-benchmark
- context-window
- 1m-tokens
- long-context
- needle-in-haystack
- multi-hop-reasoning
datasets:
- deepseek-v4-context-bench
metrics:
- accuracy
- f1
- exact-match
---

# {title}

> 🤖 **Made Autonomously Using [NEO](https://heyneo.com)** — Your Autonomous AI Engineering Agent
>
> [![VS Code Extension](https://img.shields.io/badge/VS%20Code-Install%20NEO-007ACC?logo=visualstudiocode&logoColor=white)](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo) [![Cursor Extension](https://img.shields.io/badge/Cursor-Install%20NEO-000000?logo=cursor&logoColor=white)](https://marketplace.cursorapi.com/items/?itemName=NeoResearchInc.heyneo)

## Dataset Description

This dataset contains benchmark results for evaluating large language models
on 1M-token context window tasks, specifically designed for DeepSeek V4
and comparable models.

### Supported Models

- **deepseek/deepseek-v4-flash**: Fast variant with 1M context window
- **deepseek/deepseek-v4-pro**: Professional variant with enhanced capabilities
- **meta-llama/llama-4-scout-17b-16e-instruct**: Meta's Llama 4 Scout model

### Corpus Types

1. **NIAH (Needle In A Haystack)**: Tests information retrieval
2. **Multi-hop Reasoning**: Tests multi-step reasoning
3. **Codebase Analysis**: Tests code understanding
4. **Synthetic Data**: Diverse synthetic content

## Dataset Structure

```json
{{
  "task_id": "string",
  "corpus_type": "niah|multihop|codebase|synthesis",
  "model": "string",
  "context": "string",
  "question": "string",
  "expected_answer": "string",
  "prediction": "string",
  "score": "float",
  "correct": "boolean",
  "latency_ms": "float",
  "total_tokens": "int"
}}
```

## Pricing (April 2026)

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| deepseek-v4-flash | $0.10 | $0.25 |
| deepseek-v4-pro | $0.50 | $1.50 |
| llama-4-scout | $0.15 | $0.40 |

## Usage

### CLI

```bash
# Run benchmark
dsv4ctx run --model deepseek/deepseek-v4-flash --corpus niah --tasks 100

# Generate report
dsv4ctx report --format markdown

# Generate dataset card
dsv4ctx card --output dataset_card.md
```

### Python API

```python
from deepseek_v4_context_bench import BenchmarkRunner

runner = BenchmarkRunner()
results = await runner.run_benchmark(
    model="deepseek/deepseek-v4-flash",
    corpus_type="niah",
    num_tasks=100,
)
```

## Architecture

```mermaid
graph TD
    A[CLI] --> B[BenchmarkRunner]
    B --> C[Corpus Generator]
    B --> D[OpenRouter Client]
    B --> E[Scorer]
    C --> F[NIAH]
    C --> G[MultiHop]
    C --> H[Codebase]
    C --> I[Synthesis]
    D --> J[Flash/Pro/Scout]
    E --> K[Exact Match]
    E --> L[F1 Score]
    B --> M[Report Generator]
```

## Citation

```bibtex
@software{{deepseek_v4_context_bench,
  title = {{{title}}},
  version = {{{version}}},
  year = {{{datetime.now().year}}},
}}
```
"""
    return card


def generate_model_card(model: str, results: dict[str, Any]) -> str:
    """Generate a model card for benchmark results.

    Args:
        model: Model identifier.
        results: Benchmark results dictionary.

    Returns:
        Model card markdown content.
    """
    stats = results.get("statistics", {})
    accuracy = stats.get("accuracy", 0.0)
    avg_latency = stats.get("avg_latency_ms", 0.0)
    total_tasks = stats.get("total_tasks", 0)
    completed = stats.get("completed_tasks", 0)

    card = f"""# Model Card: {model}

## Benchmark Results

### Performance Metrics

- **Accuracy**: {accuracy:.2%}
- **Average Latency**: {avg_latency:.1f} ms
- **Total Tasks**: {total_tasks}
- **Completed Tasks**: {completed}

### Corpus Breakdown

Results by corpus type and context length.

### Context Length Performance

Performance across different context lengths.
"""
    return card
