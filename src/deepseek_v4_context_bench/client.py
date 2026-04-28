"""OpenRouter API client with error handling and exponential backoff.

This module provides a robust client for making API calls to OpenRouter
with structured exceptions, retry logic, and proper error handling.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError

if TYPE_CHECKING:
    from .config import ModelProvider


class OpenRouterError(Exception):
    """Base exception for OpenRouter client errors."""

    def __init__(self, message: str, status_code: int | None = None, response_body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class OpenRouterRateLimitError(OpenRouterError):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class OpenRouterAuthError(OpenRouterError):
    """Raised when authentication fails."""

    def __init__(self, message: str):
        super().__init__(message, status_code=401)


class OpenRouterContextLengthError(OpenRouterError):
    """Raised when context length exceeds model limits."""

    def __init__(self, message: str, max_tokens: int | None = None):
        super().__init__(message, status_code=400)
        self.max_tokens = max_tokens


class OpenRouterServerError(OpenRouterError):
    """Raised when server returns 5xx error."""

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message, status_code=status_code)


class OpenRouterConnectionError(OpenRouterError):
    """Raised when connection to API fails."""

    def __init__(self, message: str):
        super().__init__(message, status_code=None)


@dataclass
class CompletionResult:
    """Result of a completion request."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    finish_reason: str | None = None
    latency_ms: float = 0.0


class OpenRouterClient:
    """Async client for OpenRouter API with retry logic.

    Provides robust error handling, exponential backoff, and structured
    exceptions for all API interactions.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: int = 300,
        max_retries: int = 5,
        retry_delay: float = 1.0,
        retry_backoff: float = 2.0,
    ):
        """Initialize the OpenRouter client.

        Args:
            api_key: OpenRouter API key.
            base_url: API base URL.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries for failed requests.
            retry_delay: Initial retry delay in seconds.
            retry_backoff: Exponential backoff multiplier.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff

        self._client: AsyncOpenAI | None = None

    async def _get_client(self) -> AsyncOpenAI:
        """Get or create the OpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    def _map_error(self, error: APIError) -> OpenRouterError:
        """Map OpenAI APIError to structured OpenRouterError.

        Args:
            error: The APIError from OpenAI client.

        Returns:
            Appropriate OpenRouterError subclass.
        """
        status_code = getattr(error, "status_code", None)
        message = str(error)

        if status_code == 401:
            return OpenRouterAuthError("Authentication failed. Check your API key.")
        elif status_code == 429:
            retry_after = None
            if hasattr(error, "headers") and error.headers:
                retry_after_str = error.headers.get("retry-after")
                if retry_after_str:
                    import contextlib
                    with contextlib.suppress(ValueError):
                        retry_after = float(retry_after_str)
            return OpenRouterRateLimitError(
                "Rate limit exceeded. Please retry later.",
                retry_after=retry_after,
            )
        elif status_code == 400:
            if "context length" in message.lower() or "too long" in message.lower():
                return OpenRouterContextLengthError(
                    f"Context length exceeded: {message}",
                )
            return OpenRouterError(f"Bad request: {message}", status_code=400)
        elif status_code and 500 <= status_code < 600:
            return OpenRouterServerError(
                f"Server error: {message}",
                status_code=status_code,
            )
        else:
            return OpenRouterError(
                f"API error: {message}",
                status_code=status_code,
            )

    async def create_completion(
        self,
        model: str | ModelProvider,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float | None = None,
        stream: bool = False,
    ) -> CompletionResult:
        """Create a chat completion with retry logic.

        Args:
            model: Model identifier or ModelProvider enum.
            messages: List of message dicts with 'role' and 'content'.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            stream: Whether to stream the response.

        Returns:
            CompletionResult with response data.

        Raises:
            OpenRouterError: On API errors after retries exhausted.
        """
        model_str = model.value if hasattr(model, "value") else model

        last_error: Exception | None = None
        delay = self.retry_delay

        for attempt in range(self.max_retries):
            start_time = time.time()
            try:
                client = await self._get_client()
                if stream:
                    raise OpenRouterError(
                        "Streaming responses are not supported by create_completion",
                    )
                response = await client.chat.completions.create(
                    model=model_str,
                    messages=cast(Any, messages),
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=False,
                )

                latency_ms = (time.time() - start_time) * 1000

                # Extract usage information
                usage = response.usage
                choice = response.choices[0] if response.choices else None

                if choice is None:
                    raise OpenRouterError("Empty response from API")

                return CompletionResult(
                    content=choice.message.content or "",
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0,
                    model=response.model or model_str,
                    finish_reason=choice.finish_reason,
                    latency_ms=latency_ms,
                )

            except (RateLimitError, APIConnectionError) as e:
                last_error = self._map_error(e)
                if isinstance(last_error, OpenRouterRateLimitError) and last_error.retry_after:
                    delay = last_error.retry_after

                if attempt < self.max_retries - 1:
                    await asyncio.sleep(delay)
                    delay *= self.retry_backoff
                else:
                    raise last_error from e

            except APIError as e:
                last_error = self._map_error(e)
                # Don't retry auth errors or context length errors
                if isinstance(last_error, (OpenRouterAuthError, OpenRouterContextLengthError)):
                    raise last_error from e

                if attempt < self.max_retries - 1:
                    await asyncio.sleep(delay)
                    delay *= self.retry_backoff
                else:
                    raise last_error from e

            except Exception as e:
                last_error = OpenRouterConnectionError(f"Unexpected error: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(delay)
                    delay *= self.retry_backoff
                else:
                    raise last_error from e

        # Should never reach here, but just in case
        raise last_error or OpenRouterError("All retries failed")

    async def validate_api_key(self) -> bool:
        """Validate the API key by making a minimal request.

        Returns:
            True if API key is valid, False otherwise.
        """
        try:
            client = await self._get_client()
            # Make a minimal request to validate key
            await client.models.list()
            return True
        except Exception:
            return False

    async def get_available_models(self) -> list[str]:
        """Get list of available models from OpenRouter.

        Returns:
            List of model identifiers.
        """
        try:
            client = await self._get_client()
            models = await client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            raise OpenRouterError(f"Failed to fetch models: {e}") from e

    async def close(self) -> None:
        """Close the client connection."""
        if self._client:
            await self._client.close()
            self._client = None

    async def __aenter__(self) -> OpenRouterClient:
        """Async context manager entry."""
        await self._get_client()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Async context manager exit."""
        await self.close()
