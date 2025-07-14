"""
Distributed Agentic Cognitive Grammar Network

Phase 1: Cognitive Primitives & Foundational Hypergraph Encoding

This module implements a distributed cognitive grammar network that bridges
symbolic reasoning and neural computation through RWKV-based tensor operations.

Architecture Overview:
- Scheme cognitive grammar microservices for agentic grammar ↔ AtomSpace translation
- Tensor fragment architecture with hypergraph encoding
- Prime factorization-based tensor shape registry
- ECAN-inspired attention allocation foundation
- Modular design for neural-symbolic synthesis
"""

__version__ = "0.1.0"
__author__ = "RWKV Cognitive Grammar Network Team"

# Core modules
from .core import (
    CognitiveGrammarNetwork,
    TensorFragment,
    HypergraphNode,
    AtomSpaceAdapter,
    TensorSignature,
    ModalityType
)

from .adapters import (
    SchemeAdapter,
    BiDirectionalTranslator
)

from .tensor_registry import (
    TensorShapeRegistry,
    PrimeFactorizationMapper
)

from .verification import (
    PatternTransformationTest,
    HypergraphVerifier
)

__all__ = [
    "CognitiveGrammarNetwork",
    "TensorFragment", 
    "HypergraphNode",
    "AtomSpaceAdapter",
    "TensorSignature",
    "ModalityType",
    "SchemeAdapter",
    "BiDirectionalTranslator",
    "TensorShapeRegistry",
    "PrimeFactorizationMapper",
    "PatternTransformationTest",
    "HypergraphVerifier"
]