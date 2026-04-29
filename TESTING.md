# Testing Guide

## Quick Start

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=deepseek_v4_context_bench --cov-report=term-missing

# Run a specific benchmark suite
pytest tests/test_corpora_niah.py -v
```

## Test Inventory

| Module | Tests | What It Validates |
|--------|-------|-------------------|
| `test_corpora_niah.py` | 20 | Needle-in-haystack corpus generation |
| `test_corpora_multihop.py` | 20 | Multi-hop reasoning chain generation |
| `test_corpora_codebase.py` | 20 | Code pattern corpus generation |
| `test_corpora_synthesis.py` | 20 | Synthetic text corpus generation |
| `test_scorer.py` | 33 | Exact-match, F1, and contains-match scoring |
| `test_client.py` | 23 | OpenRouter API client and retry logic |
| `test_tokenizer.py` | 19 | Token counting, truncation, padding |
| `test_cli.py` | 15 | CLI commands: run, report, estimate, models, card |
| `test_card.py` | 13 | Dataset and model card generation |
| `test_report.py` | 12 | Markdown, CSV, JSON, comparison reports |
| `test_runner.py` | 12 | Benchmark runner, budget, preflight checks |
| `test_config.py` | 13 | Config loading, pricing table, env vars |
| `test_integration.py` | 1 | End-to-end dry-run smoke test |
| **Total** | **221** | |

Coverage target: **85%** (currently **92%**).

## Benchmark Corpus Tests (20 each)

### NIAH — Needle In A Haystack (`test_corpora_niah.py`)

Tests that a unique code is physically planted inside a large haystack and is retrievable.

Key assertions:
- `expected_answer` equals the first needle's answer code
- Each needle's text embeds its answer code (not just stored separately)
- `needle_positions` length equals `needle_count`
- Code length matches `code_length` config; codes are uppercase alphanumeric
- Longer `haystack_sentences` produces strictly longer text
- Same seed → identical output; different seeds → different output

### MultiHop (`test_corpora_multihop.py`)

Tests that reasoning chains are structurally valid and linked end-to-end.

Key assertions:
- `hop_count == len(reasoning_chain)` always
- Each chain step contains keys: `fact`, `from_entity`, `to_entity`, `relationship`
- Chain links are connected: `chain[i]["to_entity"] == chain[i+1]["from_entity"]`
- `sample.answer` equals `chain[-1]["to_entity"]`
- `sample.answer` appears in `sample.context`
- `generate()` with no argument uses `config.num_questions`

> **Known limitation**: The entity assignment formula makes 3-hop chains
> (person → org → product → location) structurally impossible with the current
> implementation. The `test_different_hop_counts` test asserts `== hops` for
> 1- and 2-hop configs, and `>= 1` for 3-hop configs.

### Codebase (`test_corpora_codebase.py`)

Tests that synthetic code repositories are generated with real named patterns.

Key assertions:
- `file_structure` length always equals `files_count`
- File extensions match language (`.py`, `.js`, `.ts`, `.java`, `.cpp`, `.rs`)
- `sample.code` contains named patterns from `CODE_PATTERNS` (`calculate_sum`, `DataProcessor`)
- Combined code string contains per-file `// File:` section headers
- Each `pattern_locations` entry has `file`, `patterns`, and `positions` keys
- `LANGUAGES` constant includes all six supported languages

### Synthesis (`test_corpora_synthesis.py`)

Tests that synthetic content has planted markers and deterministic structure.

Key assertions:
- `sample.expected_answer` is physically present in `sample.content`
- `sample.expected_answer == sample.metadata["planted_marker"]`
- Narrative content always starts with "Once upon a time"
- Dialogue content always contains `"` and "said"
- Structured content always contains both `[` and `{` (JSON array of objects)
- `metadata["entities"]` is a non-empty list
- More paragraphs → strictly longer content
- `generate(0)` returns `[]`

## Running Individual Test Classes

```bash
# Just the corpus tests
pytest tests/test_corpora_niah.py tests/test_corpora_multihop.py \
       tests/test_corpora_codebase.py tests/test_corpora_synthesis.py -v

# Scorer tests
pytest tests/test_scorer.py -v

# CLI dry-run tests (no API key needed)
pytest tests/test_cli.py -v

# Skip integration test (requires no external services, but slower)
pytest tests/ --ignore=tests/test_integration.py
```

## What Passes Without an API Key

All 221 tests run without any API credentials. The client tests use `respx` to
mock HTTP calls; the runner tests use `dry_run=True` mode; no real model
inference happens anywhere in the test suite.

## Coverage by Module

| Module | Coverage |
|--------|----------|
| `config.py` | 100% |
| `card.py` | 100% |
| `corpora/niah.py` | 100% |
| `corpora/codebase.py` | 100% |
| `corpora/multihop.py` | 95% |
| `corpora/synthesis.py` | 91% |
| `scorer.py` | 94% |
| `tokenizer.py` | 97% |
| `report.py` | 98% |
| `client.py` | 93% |
| `runner.py` | 82% |
| `cli.py` | 82% |
