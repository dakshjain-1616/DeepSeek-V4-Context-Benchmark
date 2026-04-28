"""Token-accurate prompt construction using tiktoken.

This module provides utilities for counting tokens and constructing
prompts with precise token budgets up to 1M tokens.
"""

from __future__ import annotations

from typing import Final

import tiktoken

# Default encoding for token counting (cl100k_base is used by GPT-4, compatible with most models)
DEFAULT_ENCODING: Final[str] = "cl100k_base"

# Token overhead for different message formats
MESSAGE_OVERHEAD: Final[int] = 4  # <|im_start|>role\ncontent<|im_end|>
SYSTEM_MESSAGE_OVERHEAD: Final[int] = 3  # <|im_start|>system\n

class Tokenizer:
    """Token-accurate prompt construction utility.

    Uses tiktoken for precise token counting and prompt construction
    with exact token budgets.
    """

    def __init__(self, encoding_name: str = DEFAULT_ENCODING) -> None:
        """Initialize tokenizer with specified encoding.

        Args:
            encoding_name: The tiktoken encoding name to use.
        """
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.encoding_name = encoding_name

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string.

        Args:
            text: The text to count tokens for.

        Returns:
            Number of tokens.
        """
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def count_message_tokens(self, role: str, content: str) -> int:
        """Count tokens in a chat message.

        Args:
            role: The message role (system, user, assistant).
            content: The message content.

        Returns:
            Number of tokens including message overhead.
        """
        base_tokens = self.count_tokens(content)
        overhead = MESSAGE_OVERHEAD + self.count_tokens(role)
        return base_tokens + overhead

    def count_messages_tokens(self, messages: list[dict[str, str]]) -> int:
        """Count tokens in a list of chat messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.

        Returns:
            Total number of tokens including all overheads.
        """
        total = 0
        for msg in messages:
            total += self.count_message_tokens(msg.get("role", "user"), msg.get("content", ""))
        # Add reply overhead
        total += 3
        return total

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within max_tokens.

        Args:
            text: The text to truncate.
            max_tokens: Maximum number of tokens allowed.

        Returns:
            Truncated text.
        """
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.encoding.decode(tokens[:max_tokens])

    def pad_to_tokens(self, text: str, target_tokens: int, pad_char: str = " ") -> str:
        """Pad text to reach target token count.

        Args:
            text: The base text.
            target_tokens: Target number of tokens.
            pad_char: Character to use for padding.

        Returns:
            Padded text.
        """
        current_tokens = self.count_tokens(text)
        if current_tokens >= target_tokens:
            return text

        # Estimate tokens per character (approximate)
        tokens_needed = target_tokens - current_tokens
        # Average ~4 chars per token for English text
        chars_needed = tokens_needed * 4

        padding = pad_char * chars_needed
        padded = text + padding

        # Fine-tune to exact token count
        final_tokens = self.count_tokens(padded)
        while final_tokens < target_tokens:
            padded += pad_char * 4
            final_tokens = self.count_tokens(padded)
        while final_tokens > target_tokens:
            padded = padded[:-4]
            final_tokens = self.count_tokens(padded)

        return padded

    def create_filler_text(self, target_tokens: int, pattern: str | None = None) -> str:
        """Create filler text to reach exact token count.

        Args:
            target_tokens: Target number of tokens.
            pattern: Optional pattern to repeat (defaults to lorem ipsum).

        Returns:
            Filler text with exact token count.
        """
        if pattern is None:
            # Use lorem ipsum paragraphs
            pattern = (
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
                "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
                "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. "
            )

        # Calculate repetitions needed
        pattern_tokens = self.count_tokens(pattern)
        repetitions = (target_tokens // pattern_tokens) + 1

        filler = pattern * repetitions
        tokens = self.encoding.encode(filler)

        if len(tokens) > target_tokens:
            return self.encoding.decode(tokens[:target_tokens])
        return filler

    def build_context_prompt(
        self,
        context: str,
        question: str,
        system_prompt: str | None = None,
        max_context_tokens: int = 1_000_000,
        reserve_tokens: int = 1024,
    ) -> list[dict[str, str]]:
        """Build a chat prompt with context fitting within token budget.

        Args:
            context: The context text (will be truncated if needed).
            question: The question to ask.
            system_prompt: Optional system prompt.
            max_context_tokens: Maximum total tokens allowed.
            reserve_tokens: Tokens to reserve for output.

        Returns:
            List of message dicts ready for API call.
        """
        messages: list[dict[str, str]] = []

        # Calculate available tokens for context
        system_tokens = 0
        if system_prompt:
            system_tokens = self.count_message_tokens("system", system_prompt) + 3

        question_tokens = self.count_message_tokens("user", question)
        available_for_context = (
            max_context_tokens - system_tokens - question_tokens - reserve_tokens
        )

        # Truncate context if needed
        context_tokens = self.count_tokens(context)
        if context_tokens > available_for_context:
            context = self.truncate_to_tokens(context, available_for_context)

        # Build messages
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Combine context and question
        full_prompt = f"Context:\n{context}\n\nQuestion: {question}"
        messages.append({"role": "user", "content": full_prompt})

        return messages

    def split_into_chunks(self, text: str, chunk_size: int, overlap: int = 0) -> list[str]:
        """Split text into token-sized chunks.

        Args:
            text: The text to split.
            chunk_size: Target tokens per chunk.
            overlap: Number of tokens to overlap between chunks.

        Returns:
            List of text chunks.
        """
        tokens = self.encoding.encode(text)
        chunks = []

        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunks.append(self.encoding.decode(chunk_tokens))

            # Move start position, accounting for overlap
            start = end - overlap if overlap > 0 else end

        return chunks

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        price_per_1m_input: float,
        price_per_1m_output: float,
    ) -> float:
        """Estimate API call cost based on token counts.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            price_per_1m_input: Price per 1M input tokens.
            price_per_1m_output: Price per 1M output tokens.

        Returns:
            Estimated cost in USD.
        """
        input_cost = (input_tokens / 1_000_000) * price_per_1m_input
        output_cost = (output_tokens / 1_000_000) * price_per_1m_output
        return input_cost + output_cost


# Global tokenizer instance
tokenizer = Tokenizer()


def get_tokenizer(encoding_name: str = DEFAULT_ENCODING) -> Tokenizer:
    """Get or create a tokenizer instance.

    Args:
        encoding_name: The encoding to use.

    Returns:
        Tokenizer instance.
    """
    return Tokenizer(encoding_name)
