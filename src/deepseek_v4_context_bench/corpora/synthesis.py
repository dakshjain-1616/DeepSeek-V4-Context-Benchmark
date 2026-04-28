"""Synthetic data corpus generator.

Generates diverse synthetic texts for comprehensive context testing,
including structured data, narratives, and mixed content types.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Final

# Synthetic data templates
NARRATIVE_TEMPLATES: Final[list[str]] = [
    "Once upon a time in {location}, there lived a {adjective} {noun}.",
    "The {adjective} {noun} traveled to {location} seeking adventure.",
    "In the year {year}, scientists discovered a {adjective} {noun}.",
    "A {adjective} {noun} was found in {location} by local residents.",
    "The story of the {adjective} {noun} began in {location}.",
]

DIALOGUE_TEMPLATES: Final[list[str]] = [
    '"Hello," said {person_a}. "How are you today?"',
    '"I am looking for {item}," {person_a} explained.',
    '"Have you seen {item}?" asked {person_b}.',
    '"{person_a} replied, "Yes, I found it in {location}."',
    '"That is {adjective}," said {person_b}.',
]

# Vocabulary for generation
VOCABULARY: Final[dict[str, list[str]]] = {
    "adjective": [
        "mysterious", "ancient", "brilliant", "curious", "enormous",
        "tiny", "colorful", "dark", "bright", "silent",
    ],
    "noun": [
        "artifact", "creature", "machine", "book", "city",
        "forest", "ocean", "mountain", "star", "crystal",
    ],
    "location": [
        "Tokyo", "Paris", "New York", "London", "Sydney",
        "Cairo", "Mumbai", "Rio", "Berlin", "Toronto",
    ],
    "person": [
        "Alice", "Bob", "Charlie", "Diana", "Eve",
        "Frank", "Grace", "Henry", "Ivy", "Jack",
    ],
    "item": [
        "the key", "the map", "the document", "the artifact",
        "the message", "the code", "the formula", "the secret",
    ],
}

# Structured data schemas
DATA_SCHEMAS: Final[list[dict[str, str]]] = [
    {"name": "string", "age": "integer", "city": "string"},
    {"product": "string", "price": "float", "quantity": "integer"},
    {"title": "string", "author": "string", "year": "integer"},
    {"company": "string", "revenue": "float", "employees": "integer"},
]

# Question templates for synthetic data
QUESTION_TEMPLATES: Final[list[tuple[str, str]]] = [
    ("What is the main subject of the text?", "subject"),
    ("Where does the story take place?", "location"),
    ("Who is the main character?", "character"),
    ("What year is mentioned?", "year"),
    ("What item is being sought?", "item"),
]


@dataclass
class SynthesisConfig:
    """Configuration for synthetic corpus generation."""

    seed: int = 42
    content_type: str = "mixed"  # "narrative", "dialogue", "structured", "mixed"
    total_tokens_target: int = 10000
    paragraphs: int = 50
    include_structured_data: bool = True


@dataclass
class SynthesisSample:
    """A single synthetic data sample."""

    content: str
    content_type: str
    question: str
    expected_answer: str
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)


class SynthesisCorpus:
    """Synthetic data corpus generator.

    Generates diverse synthetic texts including narratives,
    dialogues, and structured data for comprehensive testing.
    """

    def __init__(self, config: SynthesisConfig | None = None) -> None:
        """Initialize the synthesis corpus generator.

        Args:
            config: Configuration for generation. Uses defaults if None.
        """
        self.config = config or SynthesisConfig()
        self.rng = random.Random(self.config.seed)

    def _get_random_word(self, category: str) -> str:
        """Get a random word from vocabulary.

        Args:
            category: Word category (adjective, noun, etc.).

        Returns:
            Random word from the category.
        """
        words = VOCABULARY.get(category, ["unknown"])
        return self.rng.choice(words)

    def _generate_narrative_paragraph(self, index: int) -> str:
        """Generate a narrative paragraph.

        Args:
            index: Paragraph index for variation.

        Returns:
            Narrative paragraph text.
        """
        template = NARRATIVE_TEMPLATES[index % len(NARRATIVE_TEMPLATES)]
        year = 1900 + (index % 124)  # Years 1900-2023

        paragraph = template.format(
            location=self._get_random_word("location"),
            adjective=self._get_random_word("adjective"),
            noun=self._get_random_word("noun"),
            year=year,
        )

        # Add more sentences
        additional = self.rng.randint(2, 5)
        for _ in range(additional):
            sentence = (
                f"The {self._get_random_word('adjective')} {self._get_random_word('noun')} "
                f"was discovered by {self._get_random_word('person')} in {year}."
            )
            paragraph += " " + sentence

        return paragraph

    def _generate_dialogue(self, index: int) -> str:
        """Generate a dialogue segment.

        Args:
            index: Dialogue index for variation.

        Returns:
            Dialogue text.
        """
        person_a = self._get_random_word("person")
        person_b = self._get_random_word("person")
        while person_b == person_a:
            person_b = self._get_random_word("person")

        lines = []
        for i in range(self.rng.randint(3, 6)):
            template = DIALOGUE_TEMPLATES[i % len(DIALOGUE_TEMPLATES)]
            line = template.format(
                person_a=person_a,
                person_b=person_b,
                item=self._get_random_word("item"),
                location=self._get_random_word("location"),
                adjective=self._get_random_word("adjective"),
            )
            lines.append(line)

        return "\n".join(lines)

    def _generate_structured_data(self, index: int) -> str:
        """Generate structured data (JSON-like).

        Args:
            index: Data index for variation.

        Returns:
            Structured data as formatted string.
        """
        schema = DATA_SCHEMAS[index % len(DATA_SCHEMAS)]
        records = []

        for i in range(self.rng.randint(3, 8)):
            record: dict[str, Any] = {}
            for schema_field, field_type in schema.items():
                if field_type == "string":
                    if schema_field == "name":
                        record[schema_field] = self._get_random_word("person")
                    elif schema_field in ["city", "location"]:
                        record[schema_field] = self._get_random_word("location")
                    else:
                        record[schema_field] = f"{schema_field}_{i}"
                elif field_type == "integer":
                    if schema_field == "age":
                        record[schema_field] = self.rng.randint(18, 80)
                    elif schema_field == "year":
                        record[schema_field] = self.rng.randint(1900, 2024)
                    else:
                        record[schema_field] = self.rng.randint(1, 1000)
                elif field_type == "float":
                    record[schema_field] = round(self.rng.uniform(10.0, 1000.0), 2)
            records.append(record)

        return json.dumps(records, indent=2)

    def _generate_mixed_content(self, index: int) -> str:
        """Generate mixed content type.

        Args:
            index: Content index.

        Returns:
            Mixed content text.
        """
        content_types = ["narrative", "dialogue"]
        if self.config.include_structured_data:
            content_types.append("structured")

        selected = self.rng.choice(content_types)

        if selected == "narrative":
            return self._generate_narrative_paragraph(index)
        elif selected == "dialogue":
            return self._generate_dialogue(index)
        else:
            return self._generate_structured_data(index)

    def _generate_content(self, content_type: str, paragraphs: int) -> tuple[str, dict[str, Any]]:
        """Generate content based on type.

        Args:
            content_type: Type of content to generate.
            paragraphs: Number of paragraphs/sections.

        Returns:
            Tuple of (content, metadata).
        """
        sections = []
        metadata: dict[str, Any] = {
            "content_type": content_type,
            "paragraphs": paragraphs,
            "entities": [],
        }

        for i in range(paragraphs):
            if content_type == "narrative":
                section = self._generate_narrative_paragraph(i)
                metadata["entities"].append({
                    "type": "narrative",
                    "location": self._get_random_word("location"),
                    "subject": self._get_random_word("noun"),
                })
            elif content_type == "dialogue":
                section = self._generate_dialogue(i)
                metadata["entities"].append({
                    "type": "dialogue",
                    "participants": [
                        self._get_random_word("person"),
                        self._get_random_word("person"),
                    ],
                })
            elif content_type == "structured":
                section = self._generate_structured_data(i)
                metadata["entities"].append({
                    "type": "structured",
                    "records": self.rng.randint(3, 8),
                })
            else:  # mixed
                section = self._generate_mixed_content(i)
                metadata["entities"].append({"type": "mixed"})

            sections.append(section)

        return "\n\n".join(sections), metadata

    def _make_marker_token(self, sample_id: int) -> str:
        """Build a unique 8-char hex token for a planted-fact marker.

        Args:
            sample_id: Sample index, used as deterministic salt.

        Returns:
            Uppercase 8-char hex string (e.g. "A1B2C3D4").
        """
        rng = random.Random(self.config.seed + 1000 + sample_id)
        return "".join(rng.choice("0123456789ABCDEF") for _ in range(8))

    def _plant_marker(self, content: str, marker: str) -> str:
        """Insert a planted-fact sentence somewhere in the middle of content.

        Args:
            content: Generated synthetic content.
            marker: Unique answer token to embed.

        Returns:
            Content with the planted-fact sentence inserted.
        """
        sections = content.split("\n\n")
        if not sections:
            return f"The reference identifier is {marker}.\n\n{content}"
        midpoint = len(sections) // 2
        sections.insert(
            midpoint,
            f"The reference identifier for this document is {marker}.",
        )
        return "\n\n".join(sections)

    def generate(self, count: int = 1) -> list[SynthesisSample]:
        """Generate synthetic samples.

        Each sample has a unique 8-char hex marker planted mid-document and
        a deterministic question asking for it. Scoring with the
        ``ContainsMatchScorer`` credits any response that includes the marker
        anywhere in its text.

        Args:
            count: Number of samples to generate.

        Returns:
            List of SynthesisSample objects.
        """
        samples = []

        for i in range(count):
            # Determine content type
            content_type = self.config.content_type
            if content_type == "mixed":
                types = ["narrative", "dialogue", "structured"]
                content_type = types[i % len(types)]

            # Generate content
            content, metadata = self._generate_content(
                content_type, self.config.paragraphs
            )

            # Plant a unique marker and ask for it
            marker = self._make_marker_token(i)
            content = self._plant_marker(content, marker)
            metadata["planted_marker"] = marker

            question = (
                "What is the reference identifier for this document? "
                "Reply with the identifier only, no extra words."
            )
            answer = marker

            sample = SynthesisSample(
                content=content,
                content_type=content_type,
                question=question,
                expected_answer=answer,
                metadata=metadata,
            )
            samples.append(sample)

        return samples

    def generate_single(self) -> SynthesisSample:
        """Generate a single synthetic sample.

        Returns:
            A single SynthesisSample.
        """
        samples = self.generate(1)
        return samples[0]
