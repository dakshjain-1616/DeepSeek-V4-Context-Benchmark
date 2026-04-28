"""Codebase analysis corpus generator.

Generates synthetic code repositories with specific patterns that
models must identify, testing code understanding across long contexts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Final

# Programming languages supported
LANGUAGES: Final[list[str]] = ["python", "javascript", "typescript", "java", "cpp", "rust"]

# Code patterns to embed
CODE_PATTERNS: Final[dict[str, list[str]]] = {
    "python": [
        "def calculate_sum(numbers):\n    return sum(numbers)",
        "class DataProcessor:\n    def process(self, data):\n        return data.upper()",
        "def find_max(items):\n    return max(items, key=lambda x: x['value'])",
        "def validate_input(value):\n    if not value:\n        raise ValueError('Invalid input')",
        "async def fetch_data(url):\n    return await http.get(url)",
    ],
    "javascript": [
        "function calculateSum(numbers) {\n    return numbers.reduce((a, b) => a + b, 0);\n}",
        "class DataProcessor {\n    process(data) {\n        return data.toUpperCase();\n    }\n}",
        "const findMax = (items) => Math.max(...items.map(x => x.value));",
        "function validateInput(value) {\n    if (!value) throw new Error('Invalid input');\n}",
        "async function fetchData(url) {\n    return await fetch(url).then(r => r.json());\n}",
    ],
    "typescript": [
        "function calculateSum(numbers: number[]): number {",
        "    return numbers.reduce((a, b) => a + b, 0);",
        "}",
        "interface Processor<T> {",
        "    process(data: T): T;",
        "}",
        "const findMax = <T extends { value: number }>(items: T[]): T => {",
        "    return items.reduce((max, item) => ",
        "        item.value > max.value ? item : max);",
        "};",
    ],
    "java": [
        "public int calculateSum(int[] numbers) {",
        "    return Arrays.stream(numbers).sum();",
        "}",
        "public class DataProcessor {",
        "    public String process(String data) {",
        "        return data.toUpperCase();",
        "    }",
        "}",
        "public Optional<User> findById(String id) {",
        "    return repository.findById(id);",
        "}",
    ],
    "cpp": [
        "int calculateSum(const std::vector<int>& numbers) {",
        "    return std::accumulate(numbers.begin(), numbers.end(), 0);",
        "}",
        "class DataProcessor {",
        "public:",
        "    std::string process(const std::string& data) {",
        "        return toUpper(data);",
        "    }",
        "};",
    ],
    "rust": [
        "fn calculate_sum(numbers: &[i32]) -> i32 {",
        "    numbers.iter().sum()",
        "}",
        "struct DataProcessor;",
        "",
        "impl DataProcessor {",
        "    fn process(&self, data: &str) -> String {",
        "        data.to_uppercase()",
        "    }",
        "}",
    ],
}

# Filler code snippets
FILLER_CODE: Final[dict[str, list[str]]] = {
    "python": [
        "def helper_function():\n    pass",
        "class BaseClass:\n    def __init__(self):\n        self.value = 0",
        "def process_data(data):\n    return data",
        "# This is a comment\n# Another comment",
        "import os\nimport sys",
    ],
    "javascript": [
        "function helper() {\n    return null;\n}",
        "const config = {\n    debug: true\n};",
        "// TODO: implement this\n// FIXME: bug here",
        "export const utils = {};",
    ],
    "typescript": [
        "type Config = {\n    debug: boolean;\n};",
        "interface Base {\n    id: string;\n}",
    ],
    "java": [
        "public class Helper {\n    // utility class\n}",
        "@Component\npublic class Service {\n}",
    ],
    "cpp": [
        "void helper() {\n    // implementation\n}",
        "namespace utils {\n    int value = 0;\n}",
    ],
    "rust": [
        "fn helper() {\n    // implementation\n}",
        "mod utils {\n    pub fn init() {}\n}",
    ],
}

# Questions about code patterns
CODE_QUESTIONS: Final[list[tuple[str, str]]] = [
    ("What function calculates the sum of numbers?", "calculate_sum"),
    ("Which class processes data?", "DataProcessor"),
    ("What method finds the maximum value?", "find_max"),
    ("Which function validates input?", "validate_input"),
    ("What async function fetches data?", "fetch_data"),
]


@dataclass
class CodebaseConfig:
    """Configuration for codebase corpus generation."""

    seed: int = 42
    language: str = "python"
    files_count: int = 20
    lines_per_file: int = 50
    patterns_per_sample: int = 3


@dataclass
class CodebaseSample:
    """A single codebase analysis sample."""

    code: str
    language: str
    question: str
    expected_answer: str
    file_structure: list[str]
    pattern_locations: list[dict[str, Any]]


class CodebaseCorpus:
    """Codebase analysis corpus generator.

    Generates synthetic code repositories with embedded patterns
    for testing code understanding in long contexts.
    """

    def __init__(self, config: CodebaseConfig | None = None) -> None:
        """Initialize the codebase corpus generator.

        Args:
            config: Configuration for generation. Uses defaults if None.
        """
        self.config = config or CodebaseConfig()
        self.rng = random.Random(self.config.seed)

        # Validate language
        if self.config.language not in LANGUAGES:
            self.config.language = "python"

    def _generate_file_name(self, index: int) -> str:
        """Generate a file name.

        Args:
            index: File index.

        Returns:
            File name with appropriate extension.
        """
        extensions = {
            "python": ".py",
            "javascript": ".js",
            "typescript": ".ts",
            "java": ".java",
            "cpp": ".cpp",
            "rust": ".rs",
        }
        ext = extensions.get(self.config.language, ".txt")
        return f"file_{index + 1}{ext}"

    def _generate_filler_code(self, lines: int) -> str:
        """Generate filler code.

        Args:
            lines: Number of lines to generate.

        Returns:
            Filler code string.
        """
        fillers = FILLER_CODE.get(self.config.language, FILLER_CODE["python"])
        result = []
        for i in range(lines):
            filler = fillers[i % len(fillers)]
            result.append(f"# Line {i + 1}\n{filler}")
        return "\n\n".join(result)

    def _embed_patterns(self, code: str, patterns: list[str]) -> tuple[str, list[int]]:
        """Embed patterns into code.

        Args:
            code: Base code.
            patterns: Patterns to embed.

        Returns:
            Tuple of (modified_code, pattern_positions).
        """
        lines = code.split("\n")
        positions = []

        for pattern in patterns:
            # Insert at random position
            pos = self.rng.randint(0, len(lines))
            positions.append(pos)
            pattern_lines = pattern.split("\n")
            for i, line in enumerate(pattern_lines):
                lines.insert(pos + i, line)

        return "\n".join(lines), positions

    def _generate_file_content(
        self,
        file_index: int,
        patterns: list[str] | None = None,
    ) -> tuple[str, list[int]]:
        """Generate content for a single file.

        Args:
            file_index: Index of the file.
            patterns: Optional patterns to embed.

        Returns:
            Tuple of (content, pattern_positions).
        """
        # Generate filler code
        filler = self._generate_filler_code(self.config.lines_per_file)

        # Embed patterns if provided
        if patterns:
            content, positions = self._embed_patterns(filler, patterns)
            return content, positions

        return filler, []

    def generate(self, count: int = 1) -> list[CodebaseSample]:
        """Generate codebase samples.

        Args:
            count: Number of samples to generate.

        Returns:
            List of CodebaseSample objects.
        """
        samples = []
        language_patterns = CODE_PATTERNS.get(self.config.language, CODE_PATTERNS["python"])

        for sample_idx in range(count):
            # Select patterns for this sample
            num_patterns = min(self.config.patterns_per_sample, len(language_patterns))
            selected_patterns = self.rng.sample(language_patterns, num_patterns)

            # Generate files
            files = []
            file_structure = []
            pattern_locations = []

            for file_idx in range(self.config.files_count):
                file_name = self._generate_file_name(file_idx)
                file_structure.append(file_name)

                # Distribute patterns across files
                file_patterns = []
                if file_idx < len(selected_patterns):
                    file_patterns = [selected_patterns[file_idx]]

                content, positions = self._generate_file_content(file_idx, file_patterns)
                files.append(f"// File: {file_name}\n{content}")

                if positions:
                    pattern_locations.append({
                        "file": file_name,
                        "patterns": file_patterns,
                        "positions": positions,
                    })

            # Combine all files
            full_code = "\n\n".join(files)

            # Generate question based on first pattern
            question_idx = sample_idx % len(CODE_QUESTIONS)
            question, answer = CODE_QUESTIONS[question_idx]

            sample = CodebaseSample(
                code=full_code,
                language=self.config.language,
                question=question,
                expected_answer=answer,
                file_structure=file_structure,
                pattern_locations=pattern_locations,
            )
            samples.append(sample)

        return samples

    def generate_single(self) -> CodebaseSample:
        """Generate a single codebase sample.

        Returns:
            A single CodebaseSample.
        """
        samples = self.generate(1)
        return samples[0]
