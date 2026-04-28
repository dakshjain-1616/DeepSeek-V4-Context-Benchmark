# DeepSeek V4 1M-Token Context Benchmark

## Goal
Build a production-ready benchmarking suite to compare `deepseek/deepseek-v4-flash`, `deepseek/deepseek-v4-pro`, and `meta-llama/llama-4-scout-17b-16e-instruct` on 1M-token context tasks via OpenRouter.

## Research Summary
- **Model IDs**: Verified `deepseek/deepseek-v4-flash`, `deepseek/deepseek-v4-pro`, and `meta-llama/llama-4-scout-17b-16e-instruct` are the target strings.
- **Pricing (April 2026 estimates)**:
    - `deepseek/deepseek-v4-flash`: $0.10/1M (input), $0.20/1M (output)
    - `deepseek/deepseek-v4-pro`: $2.00/1M (input), $2.00/1M (output)
    - `meta-llama/llama-4-scout-17b-16e-instruct`: $0.15/1M (input), $0.15/1M (output)
    *Note: These are snapshot estimates for the budget calculator.*
- **Tokenizer**: `o200k_base` (GPT-4o) or `cl100k_base` are standard for tiktoken; DeepSeek models often use their own, but for context-length accounting in a benchmark, using a consistent tiktoken proxy is acceptable as per instructions.

## Approach
1.  **Project Initialization**: Use `uv` for dependency management.
2.  **Core Logic**:
    - `tokenizer.py`: Implements precise token packing for 100K, 500K, and 1M targets.
    - `client.py`: Robust OpenRouter wrapper with exponential backoff and structured error handling.
    - `corpora/`: Specialized generators for NIAH, Multi-hop, Codebase Q&A, and Synthesis.
3.  **Execution & Reporting**:
    - `runner.py`: Orchestrates the sweep with budget pre-flight checks.
    - `report.py` & `card.py`: Generate Markdown reports and HF Dataset Cards.
4.  **Testing**: 100% unit test coverage for deterministic logic; `respx` for mocking OpenRouter.

## Subtasks
1. Initialize project with `uv` and install dependencies (click, openai, tiktoken, rich, pydantic, pytest, etc.), expected output: `pyproject.toml`, `uv.lock` (verify: `uv sync` success)
2. Implement `config.py` with Pydantic settings and April 2026 pricing table, expected output: `src/deepseek_v4_context_bench/config.py` (verify: `python -m py_compile`)
3. Implement `tokenizer.py` for token-accurate prompt construction up to 1M tokens, expected output: `src/deepseek_v4_context_bench/tokenizer.py` (verify: unit tests for token counts)
4. Implement `client.py` with OpenRouter error handling and retry logic, expected output: `src/deepseek_v4_context_bench/client.py` (verify: `respx` mocks in `tests/test_client.py`)
5. Implement `corpora/` (niah, multihop, codebase, synthesis) with deterministic generators, expected output: `src/deepseek_v4_context_bench/corpora/*.py` (verify: unit tests for corpus generation)
6. Implement `scorer.py` with exact-match and V4-Pro judge rubrics, expected output: `src/deepseek_v4_context_bench/scorer.py` (verify: unit tests with sample outputs)
7. Implement `runner.py` with budget estimation and orchestration logic, expected output: `src/deepseek_v4_context_bench/runner.py` (verify: dry-run mode)
8. Implement `cli.py` with Click commands (run, report, card), expected output: `src/deepseek_v4_context_bench/cli.py` (verify: `dsv4ctx --help`)
9. Implement `report.py` and `card.py` for output generation, expected output: `src/deepseek_v4_context_bench/report.py`, `src/deepseek_v4_context_bench/card.py` (verify: markdown output format)
10. Finalize README.md with Mermaid diagram and NEO attribution, expected output: `README.md` (verify: markdown lint)
11. Run full test suite and static checks (ruff, mypy), expected output: test report, zero lint errors (verify: `pytest`, `ruff check`, `mypy`)

## Deliverables
| File Path | Description |
|-----------|-------------|
| `/app/deepseek_v4_benchmark_0941/pyproject.toml` | Project configuration |
| `/app/deepseek_v4_benchmark_0941/src/deepseek_v4_context_bench/` | Core source code |
| `/app/deepseek_v4_benchmark_0941/tests/` | Test suite |
| `/app/deepseek_v4_benchmark_0941/README.md` | Documentation |

## Evaluation Criteria
- `pytest` coverage ≥ 85% on `src/`.
- Zero `ruff` and `mypy --strict` errors.
- End-to-end dry-run success for 1M-token NIAH.
- Accurate budget estimation before API calls.
