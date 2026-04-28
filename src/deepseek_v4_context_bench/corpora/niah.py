"""NIAH (Needle In A Haystack) corpus generator.

Generates synthetic texts with hidden "needles" that models must find
to test their ability to retrieve information from long contexts.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Final

# Filler text for creating haystack
HAYSTACK_SENTENCES: Final[list[str]] = [
    "The quick brown fox jumps over the lazy dog.",
    "In 1492, Christopher Columbus sailed the ocean blue.",
    "The capital of France is Paris, known for its iconic Eiffel Tower.",
    "Photosynthesis is the process by which plants convert sunlight into energy.",
    "The Great Wall of China stretches over 13,000 miles across northern China.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "Shakespeare wrote 37 plays and 154 sonnets during his lifetime.",
    "The human body contains 206 bones and over 600 muscles.",
    "Python is a high-level programming language known for its readability.",
    "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
    "Mount Everest stands at 8,849 meters above sea level.",
    "The Amazon rainforest produces about 20% of the world's oxygen.",
    "DNA stands for deoxyribonucleic acid and carries genetic information.",
    "The Mona Lisa was painted by Leonardo da Vinci in the early 16th century.",
    "Jupiter is the largest planet in our solar system.",
]

# Different needle formats to test various retrieval patterns
NEEDLE_TEMPLATES: Final[list[str]] = [
    "The secret code is: {code}",
    "Remember this code: {code}",
    "Important: {code}",
    "The answer is {code}",
    "Code: {code}",
]

# Questions to ask about needles
NEEDLE_QUESTIONS: Final[list[str]] = [
    "What is the secret code mentioned in the text?",
    "What code should I remember from the context?",
    "What important information was provided?",
    "What is the answer given in the text?",
    "What code was specified?",
]


@dataclass
class NIAHConfig:
    """Configuration for NIAH corpus generation."""

    seed: int = 42
    needle_count: int = 1
    haystack_sentences: int = 1000
    needle_position: str = "random"  # "start", "middle", "end", "random"
    code_length: int = 8


@dataclass
class NIAHSample:
    """A single NIAH sample."""

    text: str
    needles: list[dict[str, str]]
    question: str
    expected_answer: str
    needle_positions: list[int]


class NIAHCorpus:
    """Needle In A Haystack corpus generator.

    Generates texts with hidden information (needles) that models
    must retrieve from long contexts.
    """

    def __init__(self, config: NIAHConfig | None = None) -> None:
        """Initialize the NIAH corpus generator.

        Args:
            config: Configuration for generation. Uses defaults if None.
        """
        self.config = config or NIAHConfig()
        self.rng = random.Random(self.config.seed)

    def _generate_code(self, index: int) -> str:
        """Generate a deterministic secret code.

        Args:
            index: Index for deterministic generation.

        Returns:
            A unique code string.
        """
        # Use hash for deterministic but seemingly random codes
        hash_input = f"{self.config.seed}_{index}"
        hash_bytes = hashlib.sha256(hash_input.encode()).hexdigest()
        return hash_bytes[:self.config.code_length].upper()

    def _generate_needle(self, index: int) -> tuple[str, str, str]:
        """Generate a needle with its question and answer.

        Args:
            index: Index for deterministic generation.

        Returns:
            Tuple of (needle_text, question, answer).
        """
        code = self._generate_code(index)
        template = NEEDLE_TEMPLATES[index % len(NEEDLE_TEMPLATES)]
        question = NEEDLE_QUESTIONS[index % len(NEEDLE_QUESTIONS)]
        needle_text = template.format(code=code)
        return needle_text, question, code

    def _generate_haystack(self, num_sentences: int) -> list[str]:
        """Generate haystack sentences.

        Args:
            num_sentences: Number of sentences to generate.

        Returns:
            List of sentences.
        """
        sentences = []
        for i in range(num_sentences):
            sentence = HAYSTACK_SENTENCES[i % len(HAYSTACK_SENTENCES)]
            # Add some variation to avoid exact repetition
            if i >= len(HAYSTACK_SENTENCES):
                sentence = f"{sentence} (paragraph {i // len(HAYSTACK_SENTENCES) + 1})"
            sentences.append(sentence)
        return sentences

    def _insert_needles(
        self,
        haystack: list[str],
        needles: list[tuple[str, str, str]],
    ) -> tuple[list[str], list[int]]:
        """Insert needles into haystack at specified positions.

        Args:
            haystack: List of haystack sentences.
            needles: List of (needle_text, question, answer) tuples.

        Returns:
            Tuple of (modified_haystack, needle_positions).
        """
        positions = []
        result = haystack.copy()

        for i, (needle_text, _, _) in enumerate(needles):
            if self.config.needle_position == "start":
                pos = i
            elif self.config.needle_position == "end":
                pos = len(result) - len(needles) + i
            elif self.config.needle_position == "middle":
                mid = len(result) // 2
                pos = mid - len(needles) // 2 + i
            else:  # random
                pos = self.rng.randint(0, len(result))

            # Ensure position is valid
            pos = max(0, min(pos, len(result)))
            positions.append(pos)
            result.insert(pos, needle_text)

        return result, positions

    def generate(self, count: int = 1) -> list[NIAHSample]:
        """Generate NIAH samples.

        Args:
            count: Number of samples to generate.

        Returns:
            List of NIAHSample objects.
        """
        samples = []

        for sample_idx in range(count):
            # Generate needles
            needles = []
            for i in range(self.config.needle_count):
                needle_text, question, answer = self._generate_needle(
                    sample_idx * self.config.needle_count + i
                )
                needles.append((needle_text, question, answer))

            # Generate haystack
            haystack = self._generate_haystack(self.config.haystack_sentences)

            # Insert needles
            text_parts, positions = self._insert_needles(haystack, needles)
            text = " ".join(text_parts)

            # Create sample
            needle_dicts = [
                {"text": nt, "question": q, "answer": a}
                for nt, q, a in needles
            ]

            # Use the first needle's question
            sample = NIAHSample(
                text=text,
                needles=needle_dicts,
                question=needles[0][1],
                expected_answer=needles[0][2],
                needle_positions=positions,
            )
            samples.append(sample)

        return samples

    def generate_single(self) -> NIAHSample:
        """Generate a single NIAH sample.

        Returns:
            A single NIAHSample.
        """
        samples = self.generate(1)
        return samples[0]
