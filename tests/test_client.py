"""Tests for client module with respx mocks."""

import pytest
import respx
from httpx import Response

from deepseek_v4_context_bench.client import (
    CompletionResult,
    OpenRouterAuthError,
    OpenRouterClient,
    OpenRouterConnectionError,
    OpenRouterContextLengthError,
    OpenRouterError,
    OpenRouterRateLimitError,
    OpenRouterServerError,
)


class TestOpenRouterClient:
    """Tests for OpenRouterClient."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return OpenRouterClient(
            api_key="sk-test-key",
            base_url="https://api.test.com",
            timeout=30,
            max_retries=2,
            retry_delay=0.1,
        )

    @respx.mock
    async def test_create_completion_success(self, client):
        """Test successful completion."""
        route = respx.post("https://api.test.com/chat/completions").mock(
            return_value=Response(
                200,
                json={
                    "id": "test-id",
                    "model": "test-model",
                    "choices": [{
                        "message": {"content": "Test response"},
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                }
            )
        )

        result = await client.create_completion(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert isinstance(result, CompletionResult)
        assert result.content == "Test response"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5
        assert result.total_tokens == 15
        assert result.model == "test-model"
        assert result.finish_reason == "stop"
        assert route.called

    @respx.mock
    async def test_create_completion_rate_limit(self, client):
        """Test rate limit handling."""
        _ = respx.post("https://api.test.com/chat/completions").mock(
            return_value=Response(429, json={"error": "Rate limited"})
        )

        with pytest.raises(OpenRouterRateLimitError):
            await client.create_completion(
                model="test-model",
                messages=[{"role": "user", "content": "Hello"}],
            )

    @respx.mock
    async def test_create_completion_auth_error(self, client):
        """Test authentication error handling."""
        _ = respx.post("https://api.test.com/chat/completions").mock(
            return_value=Response(401, json={"error": "Unauthorized"})
        )

        with pytest.raises(OpenRouterAuthError):
            await client.create_completion(
                model="test-model",
                messages=[{"role": "user", "content": "Hello"}],
            )

    @respx.mock
    async def test_create_completion_server_error(self, client):
        """Test server error handling."""
        _ = respx.post("https://api.test.com/chat/completions").mock(
            return_value=Response(500, json={"error": "Internal server error"})
        )

        with pytest.raises(OpenRouterServerError):
            await client.create_completion(
                model="test-model",
                messages=[{"role": "user", "content": "Hello"}],
            )

    @respx.mock
    async def test_validate_api_key_success(self, client):
        """Test API key validation success."""
        _ = respx.get("https://api.test.com/models").mock(
            return_value=Response(200, json={"data": []})
        )

        is_valid = await client.validate_api_key()
        assert is_valid is True

    @respx.mock
    async def test_validate_api_key_failure(self, client):
        """Test API key validation failure."""
        _ = respx.get("https://api.test.com/models").mock(
            return_value=Response(401, json={"error": "Unauthorized"})
        )

        is_valid = await client.validate_api_key()
        assert is_valid is False

    @respx.mock
    async def test_get_available_models(self, client):
        """Test getting available models."""
        _ = respx.get("https://api.test.com/models").mock(
            return_value=Response(200, json={
                "data": [
                    {"id": "model-1"},
                    {"id": "model-2"},
                ]
            })
        )

        models = await client.get_available_models()
        assert models == ["model-1", "model-2"]

    async def test_context_manager(self):
        """Test async context manager."""
        client = OpenRouterClient(api_key="sk-test")

        async with client as c:
            assert c is client
            assert c._client is not None


class TestOpenRouterErrors:
    """Tests for OpenRouter error classes."""

    def test_base_error(self):
        """Test base error class."""
        err = OpenRouterError("Test error", status_code=400, response_body={"detail": "test"})
        assert str(err) == "Test error"
        assert err.status_code == 400
        assert err.response_body == {"detail": "test"}

    def test_rate_limit_error(self):
        """Test rate limit error."""
        err = OpenRouterRateLimitError("Rate limited", retry_after=60.0)
        assert err.status_code == 429
        assert err.retry_after == 60.0

    def test_auth_error(self):
        """Test auth error."""
        err = OpenRouterAuthError("Unauthorized")
        assert err.status_code == 401

    def test_context_length_error(self):
        """Test context length error."""
        err = OpenRouterContextLengthError("Too long", max_tokens=1000)
        assert err.status_code == 400
        assert err.max_tokens == 1000

    def test_server_error(self):
        """Test server error."""
        err = OpenRouterServerError("Server error", status_code=503)
        assert err.status_code == 503

    def test_connection_error(self):
        """Test connection error."""
        err = OpenRouterConnectionError("Connection failed")
        assert err.status_code is None


class TestOpenRouterClientExtended:
    """Additional tests for OpenRouterClient."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return OpenRouterClient(
            api_key="sk-test-key",
            base_url="https://api.test.com",
            timeout=30,
            max_retries=2,
            retry_delay=0.1,
        )

    @respx.mock
    async def test_create_completion_context_length_error(self, client):
        """Test context length error handling."""
        _ = respx.post("https://api.test.com/chat/completions").mock(
            return_value=Response(400, json={"error": "Context length exceeded"})
        )

        with pytest.raises(OpenRouterContextLengthError):
            await client.create_completion(
                model="test-model",
                messages=[{"role": "user", "content": "Hello"}],
            )

    @respx.mock
    async def test_create_completion_retry_then_success(self, client):
        """Test retry logic eventually succeeding."""
        route = respx.post("https://api.test.com/chat/completions").mock(
            side_effect=[
                Response(503, json={"error": "Service unavailable"}),
                Response(200, json={
                    "id": "test-id",
                    "model": "test-model",
                    "choices": [{
                        "message": {"content": "Success after retry"},
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                }),
            ]
        )

        result = await client.create_completion(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert result.content == "Success after retry"
        assert route.call_count == 2

    @respx.mock
    async def test_create_completion_with_retry_after_header(self, client):
        """Test rate limit with retry-after header."""
        _ = respx.post("https://api.test.com/chat/completions").mock(
            side_effect=[
                Response(429, json={"error": "Rate limited"}, headers={"retry-after": "0.1"}),
                Response(200, json={
                    "id": "test-id",
                    "model": "test-model",
                    "choices": [{
                        "message": {"content": "Success"},
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                }),
            ]
        )

        result = await client.create_completion(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert result.content == "Success"

    @respx.mock
    async def test_create_completion_empty_choices(self, client):
        """Test empty choices in response."""
        _ = respx.post("https://api.test.com/chat/completions").mock(
            return_value=Response(200, json={
                "id": "test-id",
                "model": "test-model",
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 0,
                    "total_tokens": 10,
                },
            })
        )

        with pytest.raises(OpenRouterError):
            await client.create_completion(
                model="test-model",
                messages=[{"role": "user", "content": "Hello"}],
            )

    @respx.mock
    async def test_create_completion_with_model_provider(self, client):
        """Test completion with ModelProvider enum."""
        from deepseek_v4_context_bench.config import ModelProvider

        _ = respx.post("https://api.test.com/chat/completions").mock(
            return_value=Response(200, json={
                "id": "test-id",
                "model": "deepseek/deepseek-v4-flash",
                "choices": [{
                    "message": {"content": "Test"},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            })
        )

        result = await client.create_completion(
            model=ModelProvider.DEEPSEEK_FLASH,
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert result.content == "Test"

    @respx.mock
    async def test_get_available_models_error(self, client):
        """Test get_available_models with error."""
        _ = respx.get("https://api.test.com/models").mock(
            return_value=Response(500, json={"error": "Server error"})
        )

        with pytest.raises(OpenRouterError):
            await client.get_available_models()

    async def test_close_client(self):
        """Test closing the client."""
        client = OpenRouterClient(api_key="sk-test")
        await client._get_client()  # Initialize client
        assert client._client is not None

        await client.close()
        assert client._client is None

    async def test_close_without_client(self):
        """Test closing when client is not initialized."""
        client = OpenRouterClient(api_key="sk-test")
        # Should not raise even if _client is None
        await client.close()


class TestCompletionResult:
    """Tests for CompletionResult dataclass."""

    def test_completion_result_creation(self):
        """Test creating CompletionResult."""
        result = CompletionResult(
            content="Test",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            model="test-model",
            finish_reason="stop",
            latency_ms=100.0,
        )
        assert result.content == "Test"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5
        assert result.total_tokens == 15
        assert result.model == "test-model"
        assert result.finish_reason == "stop"
        assert result.latency_ms == 100.0
