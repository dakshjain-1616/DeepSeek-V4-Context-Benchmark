"""Integration tests against live OpenRouter — skipped when OPENROUTER_API_KEY is not set.

Run with: `OPENROUTER_API_KEY=sk-or-... uv run pytest -m integration`.
"""

from __future__ import annotations

import os

import pytest

from deepseek_v4_context_bench.client import OpenRouterClient
from deepseek_v4_context_bench.config import config

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set — skipping live OpenRouter integration tests",
)


@pytest.mark.integration
async def test_client_can_reach_openrouter() -> None:
    """Smoke: live client returns a non-empty completion from V4-Flash."""
    client = OpenRouterClient(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=config.openrouter_base_url,
        timeout=60.0,
        max_retries=2,
    )
    try:
        result = await client.create_completion(
            model="deepseek/deepseek-v4-flash",
            messages=[{"role": "user", "content": "Reply with the single word: pong"}],
            max_tokens=8,
            temperature=0.0,
        )
    finally:
        await client.close()

    assert result.content
    assert result.total_tokens > 0
