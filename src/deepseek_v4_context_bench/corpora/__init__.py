"""Corpora generators for DeepSeek V4 Context Benchmark.

This package provides deterministic corpus generators for various
benchmark tasks including NIAH, multi-hop reasoning, codebase analysis,
and synthetic data generation.
"""

from .codebase import CodebaseConfig, CodebaseCorpus, CodebaseSample
from .multihop import MultiHopConfig, MultiHopCorpus, MultiHopSample
from .niah import NIAHConfig, NIAHCorpus, NIAHSample
from .synthesis import SynthesisConfig, SynthesisCorpus, SynthesisSample

__all__ = [
    "NIAHCorpus",
    "NIAHConfig",
    "NIAHSample",
    "MultiHopCorpus",
    "MultiHopConfig",
    "MultiHopSample",
    "CodebaseCorpus",
    "CodebaseConfig",
    "CodebaseSample",
    "SynthesisCorpus",
    "SynthesisConfig",
    "SynthesisSample",
]
