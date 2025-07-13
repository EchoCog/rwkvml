# Distributed Agentic Cognitive Grammar Network Documentation

This documentation suite provides comprehensive guidance for the Cognitive Grammar Network framework.

## Documentation Sections

- [**Architecture Guide**](architecture.md) - Comprehensive system architecture and design
- [**API Reference**](api.md) - Complete API documentation and method signatures  
- [**Examples & Tutorials**](examples.md) - Practical usage examples and tutorials

## Quick Start

```python
from cognitive_grammar import CognitiveGrammarNetwork, ModalityType

# Create cognitive network
network = CognitiveGrammarNetwork()

# Create agentic fragment  
fragment = network.create_agentic_fragment(
    agent_type="my_agent",
    modality=ModalityType.LINGUISTIC
)

# Register fragment
fragment_id = network.register_fragment(fragment)
print(f"Created fragment: {fragment_id}")
```

## Framework Overview

The Cognitive Grammar Network bridges symbolic reasoning and neural computation through:

- **Hypergraph Encoding**: Agents and states as hypergraph nodes with tensor representations
- **Prime Factorization**: Tensor shapes mapped via semantic complexity factorization
- **Bidirectional Translation**: Scheme ↔ Tensor ↔ AtomSpace consistency  
- **Pattern Transformations**: Verified transformations preserving semantic content
- **Distributed Processing**: Scalable cognitive mesh architecture

## Phase 1 Implementation Status

✅ **Cognitive Primitives & Foundational Hypergraph Encoding**
- Scheme cognitive grammar microservices
- Tensor fragment architecture with hypergraph encoding
- Prime factorization tensor shape registry
- Comprehensive verification framework
- Living documentation and visualization

## Next: Phase 2 - ECAN Attention Allocation

🚧 **Upcoming Implementation**
- ECAN-inspired resource allocators
- Dynamic mesh integration  
- Real-world verification with live data

Generated: 2025-07-13 23:56:44
Network Status: 3 fragments, 3 nodes
