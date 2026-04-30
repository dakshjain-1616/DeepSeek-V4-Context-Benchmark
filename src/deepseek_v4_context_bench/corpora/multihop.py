"""Multi-hop reasoning corpus generator.

Generates texts requiring multiple steps of reasoning to answer,
testing models' ability to connect information across long contexts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Final

# Entity types for multi-hop questions
ENTITIES: Final[dict[str, list[str]]] = {
    "person": [
        "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
        "Ivy", "Jack", "Kate", "Liam", "Maria", "Noah", "Olivia", "Peter",
    ],
    "location": [
        "New York", "London", "Tokyo", "Paris", "Sydney", "Berlin", "Moscow",
        "Beijing", "Dubai", "Singapore", "Toronto", "Mumbai", "Cairo", "Rio",
    ],
    "organization": [
        "TechCorp", "GlobalSystems", "InnovateLabs", "FutureWorks", "DataStream",
        "CloudNine", "QuantumLeap", "NexusGroup", "SynergyInc", "PrimeTech",
    ],
    "product": [
        "AlphaPhone", "BetaBook", "GammaWatch", "DeltaPad", "EpsilonCam",
        "ZetaDrive", "EtaCloud", "ThetaAI", "IotaChip", "KappaNet",
    ],
}

# Relationship types
RELATIONSHIPS: Final[list[tuple[str, str, str]]] = [
    ("works_at", "{person} works at {organization}", "Where does {person} work?"),
    ("located_in", "{organization} is located in {location}", "Where is {organization} located?"),
    ("born_in", "{person} was born in {location}", "Where was {person} born?"),
    ("created", "{organization} created {product}", "Who created {product}?"),
    (
        "manufactured_in",
        "{product} is manufactured in {location}",
        "Where is {product} manufactured?",
    ),
    ("leads", "{person} leads {organization}", "Who leads {organization}?"),
    ("founded", "{person} founded {organization}", "Who founded {organization}?"),
    ("studied_at", "{person} studied at {organization}", "Where did {person} study?"),
]

# Filler facts for padding
FILLER_FACTS: Final[list[str]] = [
    "The weather was pleasant that day.",
    "It was a Tuesday morning.",
    "The meeting lasted for two hours.",
    "Everyone was excited about the news.",
    "The project was completed ahead of schedule.",
    "There were over 100 attendees.",
    "The presentation was well-received.",
    "Coffee was served during the break.",
    "The venue was beautifully decorated.",
    "Several important decisions were made.",
]


@dataclass
class MultiHopConfig:
    """Configuration for multi-hop corpus generation."""

    seed: int = 42
    num_hops: int = 2  # Number of reasoning hops required
    context_facts: int = 50  # Total facts in context
    num_questions: int = 10  # Number of questions to generate


@dataclass
class MultiHopSample:
    """A single multi-hop reasoning sample."""

    context: str
    question: str
    answer: str
    reasoning_chain: list[dict[str, str]]
    hop_count: int


class MultiHopCorpus:
    """Multi-hop reasoning corpus generator.

    Generates contexts with interconnected facts requiring
    multiple reasoning steps to answer questions.
    """

    def __init__(self, config: MultiHopConfig | None = None) -> None:
        """Initialize the multi-hop corpus generator.

        Args:
            config: Configuration for generation. Uses defaults if None.
        """
        self.config = config or MultiHopConfig()
        self.rng = random.Random(self.config.seed)

    def _get_entity(self, entity_type: str, index: int) -> str:
        """Get a deterministic entity.

        Args:
            entity_type: Type of entity (person, location, etc.).
            index: Index for deterministic selection.

        Returns:
            Entity name.
        """
        entities = ENTITIES.get(entity_type, [])
        if not entities:
            return f"Entity{index}"
        return entities[index % len(entities)]

    def _generate_fact(self, fact_id: int) -> tuple[str, str, str, str]:
        """Generate a single fact.

        Args:
            fact_id: ID for deterministic generation.

        Returns:
            Tuple of (fact_text, entity_a, entity_b, relationship_type).
        """
        rel_idx = fact_id % len(RELATIONSHIPS)
        rel_type, template, _ = RELATIONSHIPS[rel_idx]

        # (entity_a_type, entity_b_type) per relationship — drives both entity
        # selection and the template's named placeholders. Without this lookup
        # a template with two distinct placeholders (e.g. "{person} works at
        # {organization}") would bind both placeholders to the same entity.
        type_map: dict[str, tuple[str, str]] = {
            "works_at":        ("person", "organization"),
            "located_in":      ("organization", "location"),
            "born_in":         ("person", "location"),
            "created":         ("organization", "product"),
            "manufactured_in": ("product", "location"),
            "leads":           ("person", "organization"),
            "founded":         ("person", "organization"),
            "studied_at":      ("person", "organization"),
        }
        a_type, b_type = type_map.get(rel_type, ("person", "organization"))

        # Use the corpus RNG rather than a deterministic index formula.
        # The formula `entities[(fact_id + 100) % len]` creates a fixed parity
        # split: for len-10 types (org, product) the offset is zero, so
        # entity_b indices are always even/odd and entity_a indices for the
        # *next* relationship type are always odd/even — permanently disjoint.
        # That makes the only valid 3-hop path (person→org→product→location)
        # structurally impossible because no product can appear as both the
        # target of "created" and the source of "manufactured_in".
        entity_a = self.rng.choice(ENTITIES[a_type])
        entity_b = self.rng.choice(ENTITIES[b_type])

        fact_text = template.format(**{a_type: entity_a, b_type: entity_b})
        return fact_text, entity_a, entity_b, rel_type

    def _generate_context(self) -> list[tuple[str, str, str, str]]:
        """Generate a context with interconnected facts.

        Each ``(entity_a, relationship)`` pair appears at most once so that
        questions of the form "what X does ENTITY_A do?" have a unique answer
        in the context. Without this dedup the small entity pool causes
        collisions (e.g. multiple distinct "Frank leads ORG" facts), making
        every question genuinely ambiguous and the benchmark useless.

        Returns:
            List of (fact_text, entity_a, entity_b, rel_type) tuples.
        """
        facts: list[tuple[str, str, str, str]] = []
        seen: set[tuple[str, str]] = set()
        i = 0
        max_attempts = self.config.context_facts * 50
        while len(facts) < self.config.context_facts and i < max_attempts:
            fact = self._generate_fact(i)
            i += 1
            key = (fact[1], fact[3])  # (entity_a, rel_type)
            if key in seen:
                continue
            seen.add(key)
            facts.append(fact)
        return facts

    def _find_reasoning_chain(
        self,
        facts: list[tuple[str, str, str, str]],
        num_hops: int,
    ) -> list[dict[str, str]] | None:
        """Find a chain of reasoning through the facts.

        Args:
            facts: List of facts.
            num_hops: Number of hops required.

        Returns:
            List of reasoning steps or None if no chain found.
        """
        if num_hops < 1 or not facts:
            return None

        # Build a *forward-only* entity graph: edges go from ent_a to ent_b in
        # the direction of the natural-language fact. Allowing reverse edges
        # makes the DFS find chains the question phrasing can't faithfully
        # describe (e.g. hopping from "TechCorp" backwards through
        # "Olivia founded TechCorp" yields a step the rendered question would
        # word as "the organization TechCorp founded", which isn't true).
        entity_to_facts: dict[str, list[tuple[int, str, str]]] = {}
        for idx, (_fact_text, ent_a, ent_b, rel_type) in enumerate(facts):
            entity_to_facts.setdefault(ent_a, []).append((idx, ent_b, rel_type))
            entity_to_facts.setdefault(ent_b, [])  # ensure key exists for endpoint lookups

        # Try to find a chain. Shuffle starting entities with the corpus rng so
        # successive calls within a single generate() invocation can return
        # different chains, producing diverse samples instead of N copies of
        # the first chain found.
        candidates = list(entity_to_facts.keys())
        self.rng.shuffle(candidates)
        for start_entity in candidates:
            chain = self._dfs_chain(
                start_entity, entity_to_facts, facts, num_hops, [], set()
            )
            if chain:
                return chain

        return None

    def _dfs_chain(
        self,
        current: str,
        entity_to_facts: dict[str, list[tuple[int, str, str]]],
        facts: list[tuple[str, str, str, str]],
        target_hops: int,
        current_chain: list[dict[str, str]],
        visited: set[int],
    ) -> list[dict[str, str]] | None:
        """DFS to find a reasoning chain.

        Args:
            current: Current entity.
            entity_to_facts: Mapping of entities to facts.
            facts: List of all facts.
            target_hops: Target number of hops.
            current_chain: Current chain of reasoning.
            visited: Set of visited fact indices.

        Returns:
            Reasoning chain if found, None otherwise.
        """
        if len(current_chain) >= target_hops:
            return current_chain

        if current not in entity_to_facts:
            return None

        for fact_idx, next_entity, rel_type in entity_to_facts[current]:
            if fact_idx in visited:
                continue

            fact_text, _ent_a, _ent_b, _ = facts[fact_idx]
            # Record the *traversal* direction (current -> next_entity), not the
            # underlying fact's literal direction. Without this, the chain's
            # final to_entity points back at where we started for any fact whose
            # ent_b == current, breaking the answer the question asks for.
            step = {
                "fact": fact_text,
                "from_entity": current,
                "to_entity": next_entity,
                "relationship": rel_type,
            }

            new_visited = visited | {fact_idx}
            new_chain = current_chain + [step]

            result = self._dfs_chain(
                next_entity, entity_to_facts, facts, target_hops, new_chain, new_visited
            )
            if result:
                return result

        return None

    def _generate_question(
        self,
        chain: list[dict[str, str]],
    ) -> tuple[str, str]:
        """Generate a question from a reasoning chain.

        Args:
            chain: List of reasoning steps.

        Returns:
            Tuple of (question, answer).
        """
        if not chain:
            return "What is the answer?", "Unknown"

        # Start entity is the first step's from_entity
        start_entity = chain[0]["from_entity"]
        # End entity is the last step's to_entity
        end_entity = chain[-1]["to_entity"]

        # Build a deterministic question that names every relationship in the
        # chain, so the model can resolve the multi-hop traversal exactly. The
        # final answer is the last hop's target entity.
        rel_phrases = {
            "works_at":        "the organization X works at",
            "located_in":      "the location X is located in",
            "born_in":         "the location X was born in",
            "created":         "the product X created",
            "manufactured_in": "the location X is manufactured in",
            "leads":           "the organization X leads",
            "founded":         "the organization X founded",
            "studied_at":      "the organization X studied at",
        }

        if len(chain) == 1:
            phrase = rel_phrases.get(chain[0]["relationship"], "the entity related to X")
            question = (
                f"Using only the facts in the context, name "
                f"{phrase.replace('X', start_entity)}. "
                f"Reply with the name only, no extra words."
            )
        else:
            steps = []
            cur = "X"
            for step in chain:
                phrase = rel_phrases.get(step["relationship"], "the entity related to X")
                steps.append(phrase.replace("X", cur))
                cur = "that result"
            question = (
                f"Using only the facts in the context, starting from "
                f"{start_entity}: find " + ", then find ".join(steps)
                + ". Reply with the final name only, no extra words."
            )

        answer = end_entity
        return question, answer

    def generate(self, count: int | None = None) -> list[MultiHopSample]:
        """Generate multi-hop reasoning samples.

        Args:
            count: Number of samples to generate. Uses config if None.

        Returns:
            List of MultiHopSample objects.
        """
        if count is None:
            count = self.config.num_questions

        samples: list[MultiHopSample] = []
        attempts = 0
        max_attempts = count * 10

        while len(samples) < count and attempts < max_attempts:
            attempts += 1

            # Generate context
            facts = self._generate_context()

            # Find reasoning chain
            chain = self._find_reasoning_chain(facts, self.config.num_hops)
            if not chain:
                continue

            # Generate question and answer
            question, answer = self._generate_question(chain)

            # Build context text
            # Shuffle facts to make it harder
            shuffled_facts = facts.copy()
            self.rng.shuffle(shuffled_facts)
            context_text = " ".join([f[0] for f in shuffled_facts])

            sample = MultiHopSample(
                context=context_text,
                question=question,
                answer=answer,
                reasoning_chain=chain,
                hop_count=len(chain),
            )
            samples.append(sample)

        return samples

    def generate_single(self) -> MultiHopSample:
        """Generate a single multi-hop sample.

        Returns:
            A single MultiHopSample.
        """
        samples = self.generate(1)
        if not samples:
            # Return a default sample if generation fails
            return MultiHopSample(
                context="No valid context generated.",
                question="What is the answer?",
                answer="Unknown",
                reasoning_chain=[],
                hop_count=0,
            )
        return samples[0]
