"""
Visualization and Documentation Module

This module generates Mermaid flowcharts, architectural diagrams,
and living documentation for the cognitive grammar network.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json

from ..core import TensorFragment, HypergraphNode, CognitiveGrammarNetwork, TensorSignature
from ..tensor_registry import TensorShapeRegistry, KernelTensorShape


@dataclass
class FlowchartNode:
    """Flowchart node representation"""
    node_id: str
    label: str
    node_type: str  # start, end, operation, decision, etc.
    properties: Dict[str, Any]


@dataclass
class FlowchartEdge:
    """Flowchart edge representation"""
    source: str
    target: str
    label: Optional[str] = None
    edge_type: str = "arrow"


class MermaidFlowchartGenerator:
    """
    Mermaid flowchart generator for cognitive grammar network visualization
    
    Creates comprehensive flowcharts for hypergraph pathways,
    tensor transformations, and network architecture.
    """
    
    def __init__(self):
        self.flowchart_nodes = {}
        self.flowchart_edges = []
        
    def create_hypergraph_flowchart(self, fragment: TensorFragment) -> str:
        """Create Mermaid flowchart for hypergraph structure"""
        mermaid_code = ["graph TD"]
        
        # Add nodes
        for i, node in enumerate(fragment.nodes):
            node_id = f"N{i}"
            node_label = f"{node.node_type}\\n{node.node_id[:8]}"
            tensor_shape = node.tensor_signature.shape
            node_label += f"\\nShape: {tensor_shape}"
            
            mermaid_code.append(f"    {node_id}[{node_label}]")
        
        # Add edges for hypergraph links
        node_to_id = {node.node_id: f"N{i}" for i, node in enumerate(fragment.nodes)}
        
        for i, node in enumerate(fragment.nodes):
            source_id = f"N{i}"
            
            for link in node.links:
                target_node_id = link['target'].node_id
                if target_node_id in node_to_id:
                    target_id = node_to_id[target_node_id]
                    link_label = f"{link['type']}\\nweight: {link['weight']:.2f}"
                    mermaid_code.append(f"    {source_id} -->|{link_label}| {target_id}")
        
        # Add styling
        mermaid_code.extend([
            "    classDef nodeClass fill:#e1f5fe,stroke:#01579b,stroke-width:2px",
            "    classDef linkClass stroke:#ff5722,stroke-width:2px"
        ])
        
        for i in range(len(fragment.nodes)):
            mermaid_code.append(f"    class N{i} nodeClass")
        
        return "\n".join(mermaid_code)
    
    def create_tensor_transformation_flowchart(self, 
                                             transformation_name: str,
                                             input_shapes: List[Tuple[int, ...]],
                                             output_shapes: List[Tuple[int, ...]],
                                             transformation_type: str = "linear") -> str:
        """Create flowchart for tensor transformation pathways"""
        mermaid_code = ["graph LR"]
        
        # Input tensors
        for i, shape in enumerate(input_shapes):
            input_id = f"IN{i}"
            shape_str = "×".join(map(str, shape))
            mermaid_code.append(f"    {input_id}[Input Tensor {i}\\nShape: {shape_str}]")
        
        # Transformation operation
        transform_id = "TRANSFORM"
        mermaid_code.append(f"    {transform_id}[{transformation_name}\\nType: {transformation_type}]")
        
        # Output tensors
        for i, shape in enumerate(output_shapes):
            output_id = f"OUT{i}"
            shape_str = "×".join(map(str, shape))
            mermaid_code.append(f"    {output_id}[Output Tensor {i}\\nShape: {shape_str}]")
        
        # Connect inputs to transformation
        for i in range(len(input_shapes)):
            mermaid_code.append(f"    IN{i} --> {transform_id}")
        
        # Connect transformation to outputs
        for i in range(len(output_shapes)):
            mermaid_code.append(f"    {transform_id} --> OUT{i}")
        
        # Add styling
        mermaid_code.extend([
            "    classDef inputClass fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px",
            "    classDef transformClass fill:#fff3e0,stroke:#ef6c00,stroke-width:2px",
            "    classDef outputClass fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px"
        ])
        
        # Apply styles
        for i in range(len(input_shapes)):
            mermaid_code.append(f"    class IN{i} inputClass")
        
        mermaid_code.append(f"    class {transform_id} transformClass")
        
        for i in range(len(output_shapes)):
            mermaid_code.append(f"    class OUT{i} outputClass")
        
        return "\n".join(mermaid_code)
    
    def create_network_architecture_flowchart(self, network: CognitiveGrammarNetwork) -> str:
        """Create comprehensive network architecture flowchart"""
        mermaid_code = ["graph TB"]
        
        # Network overview
        mermaid_code.append("    subgraph CognitiveGrammarNetwork")
        
        # Fragments
        mermaid_code.append("        subgraph Fragments")
        for i, (frag_id, fragment) in enumerate(network.fragments.items()):
            frag_node_id = f"FRAG{i}"
            node_count = len(fragment.nodes)
            mermaid_code.append(f"            {frag_node_id}[Fragment {i}\\nNodes: {node_count}\\nID: {frag_id[:8]}]")
        mermaid_code.append("        end")
        
        # Pattern library
        mermaid_code.append("        subgraph PatternLibrary")
        mermaid_code.append(f"            PATTERNS[Pattern Library\\nPatterns: {len(network.pattern_library)}]")
        mermaid_code.append("        end")
        
        # AtomSpace adapter
        if network.atomspace_adapter:
            mermaid_code.append("        subgraph AtomSpaceIntegration")
            mermaid_code.append("            ATOMSPACE[AtomSpace Adapter\\nBidirectional Sync]")
            mermaid_code.append("        end")
        
        mermaid_code.append("    end")
        
        # Add connections
        for i in range(len(network.fragments)):
            mermaid_code.append(f"    FRAG{i} -.-> PATTERNS")
            if network.atomspace_adapter:
                mermaid_code.append(f"    FRAG{i} <--> ATOMSPACE")
        
        # Add styling
        mermaid_code.extend([
            "    classDef fragClass fill:#e8f5e8,stroke:#4caf50,stroke-width:2px",
            "    classDef patternClass fill:#fff8e1,stroke:#ff9800,stroke-width:2px",
            "    classDef atomClass fill:#e3f2fd,stroke:#2196f3,stroke-width:2px"
        ])
        
        for i in range(len(network.fragments)):
            mermaid_code.append(f"    class FRAG{i} fragClass")
        
        mermaid_code.append("    class PATTERNS patternClass")
        if network.atomspace_adapter:
            mermaid_code.append("    class ATOMSPACE atomClass")
        
        return "\n".join(mermaid_code)
    
    def create_tensor_registry_flowchart(self, registry: TensorShapeRegistry) -> str:
        """Create flowchart for tensor shape registry"""
        mermaid_code = ["graph TD"]
        
        # Registry overview
        mermaid_code.append("    subgraph TensorShapeRegistry")
        
        # Kernel shapes grouped by complexity
        complexity_groups = {}
        for kernel_name, kernel_shape in registry.kernel_shapes.items():
            complexity = int(kernel_shape.complexity_score)
            if complexity not in complexity_groups:
                complexity_groups[complexity] = []
            complexity_groups[complexity].append((kernel_name, kernel_shape))
        
        # Create nodes for complexity groups
        for complexity, kernels in complexity_groups.items():
            group_id = f"COMP{complexity}"
            mermaid_code.append(f"        subgraph Complexity_{complexity}")
            
            for i, (kernel_name, kernel_shape) in enumerate(kernels):
                kernel_id = f"{group_id}_K{i}"
                shape_str = "×".join(map(str, kernel_shape.tensor_signature.shape))
                params = kernel_shape.total_parameters
                mermaid_code.append(f"            {kernel_id}[{kernel_name}\\nShape: {shape_str}\\nParams: {params}]")
            
            mermaid_code.append("        end")
        
        mermaid_code.append("    end")
        
        # Add prime factorization mapper
        mermaid_code.append("    PRIMEMAPPER[Prime Factorization Mapper\\nSemantic → Prime Mapping]")
        mermaid_code.append("    PRIMEMAPPER --> TensorShapeRegistry")
        
        # Add optimization engine
        mermaid_code.append("    OPTIMIZER[Shape Optimizer\\nCompatibility Analysis]")
        mermaid_code.append("    TensorShapeRegistry --> OPTIMIZER")
        
        # Add styling
        mermaid_code.extend([
            "    classDef kernelClass fill:#f1f8e9,stroke:#689f38,stroke-width:2px",
            "    classDef mapperClass fill:#fce4ec,stroke:#c2185b,stroke-width:2px",
            "    classDef optimizerClass fill:#e0f2f1,stroke:#00695c,stroke-width:2px"
        ])
        
        # Apply styles to all kernels
        for complexity, kernels in complexity_groups.items():
            group_id = f"COMP{complexity}"
            for i in range(len(kernels)):
                kernel_id = f"{group_id}_K{i}"
                mermaid_code.append(f"    class {kernel_id} kernelClass")
        
        mermaid_code.append("    class PRIMEMAPPER mapperClass")
        mermaid_code.append("    class OPTIMIZER optimizerClass")
        
        return "\n".join(mermaid_code)
    
    def create_phase_implementation_flowchart(self) -> str:
        """Create flowchart for the 6-phase implementation pathway"""
        mermaid_code = ["graph TD"]
        
        # Phase 1: Cognitive Primitives
        mermaid_code.extend([
            "    subgraph Phase1[Phase 1: Cognitive Primitives]",
            "        P1A[Scheme Cognitive Grammar\\nMicroservices]",
            "        P1B[Tensor Fragment\\nArchitecture]", 
            "        P1C[Verification &\\nVisualization]",
            "        P1A --> P1B",
            "        P1B --> P1C",
            "    end"
        ])
        
        # Phase 2: ECAN Attention
        mermaid_code.extend([
            "    subgraph Phase2[Phase 2: ECAN Attention Allocation]",
            "        P2A[Kernel & Scheduler\\nDesign]",
            "        P2B[Dynamic Mesh\\nIntegration]",
            "        P2C[Real-World\\nVerification]",
            "        P2A --> P2B",
            "        P2B --> P2C",
            "    end"
        ])
        
        # Phase 3: Neural-Symbolic Synthesis
        mermaid_code.extend([
            "    subgraph Phase3[Phase 3: Neural-Symbolic Synthesis]",
            "        P3A[Custom ggml\\nKernels]",
            "        P3B[Tensor\\nBenchmarking]",
            "        P3C[End-to-End\\nVerification]",
            "        P3A --> P3B",
            "        P3B --> P3C",
            "    end"
        ])
        
        # Phase 4: Distributed Mesh API
        mermaid_code.extend([
            "    subgraph Phase4[Phase 4: Distributed Cognitive Mesh]",
            "        P4A[API & Endpoint\\nEngineering]",
            "        P4B[Embodiment\\nBindings]",
            "        P4C[Integration\\nVerification]",
            "        P4A --> P4B",
            "        P4B --> P4C",
            "    end"
        ])
        
        # Phase 5: Meta-Cognition
        mermaid_code.extend([
            "    subgraph Phase5[Phase 5: Recursive Meta-Cognition]",
            "        P5A[Meta-Cognitive\\nPathways]",
            "        P5B[Adaptive\\nOptimization]",
            "        P5C[Recursive\\nVerification]",
            "        P5A --> P5B",
            "        P5B --> P5C",
            "    end"
        ])
        
        # Phase 6: Unification
        mermaid_code.extend([
            "    subgraph Phase6[Phase 6: Cognitive Unification]",
            "        P6A[Deep Testing\\nProtocols]",
            "        P6B[Recursive\\nDocumentation]",
            "        P6C[Cognitive\\nUnification]",
            "        P6A --> P6B",
            "        P6B --> P6C",
            "    end"
        ])
        
        # Phase connections
        mermaid_code.extend([
            "    Phase1 --> Phase2",
            "    Phase2 --> Phase3", 
            "    Phase3 --> Phase4",
            "    Phase4 --> Phase5",
            "    Phase5 --> Phase6"
        ])
        
        # Add styling for phases
        mermaid_code.extend([
            "    classDef phase1Class fill:#e8f5e8,stroke:#4caf50,stroke-width:3px",
            "    classDef phase2Class fill:#fff3e0,stroke:#ff9800,stroke-width:3px",
            "    classDef phase3Class fill:#e3f2fd,stroke:#2196f3,stroke-width:3px",
            "    classDef phase4Class fill:#f3e5f5,stroke:#9c27b0,stroke-width:3px",
            "    classDef phase5Class fill:#fce4ec,stroke:#e91e63,stroke-width:3px",
            "    classDef phase6Class fill:#fff8e1,stroke:#ffc107,stroke-width:3px"
        ])
        
        # Apply phase styling
        for phase_num in range(1, 7):
            for sub in ['A', 'B', 'C']:
                mermaid_code.append(f"    class P{phase_num}{sub} phase{phase_num}Class")
        
        return "\n".join(mermaid_code)


class DocumentationGenerator:
    """
    Living documentation generator for the cognitive grammar network
    
    Creates comprehensive, auto-updating documentation with
    architectural diagrams, API references, and example usage.
    """
    
    def __init__(self, output_dir: str = "docs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.mermaid_generator = MermaidFlowchartGenerator()
        
    def generate_architecture_documentation(self, 
                                          network: CognitiveGrammarNetwork,
                                          registry: TensorShapeRegistry) -> str:
        """Generate comprehensive architecture documentation"""
        
        # Generate flowcharts
        network_flowchart = self.mermaid_generator.create_network_architecture_flowchart(network)
        registry_flowchart = self.mermaid_generator.create_tensor_registry_flowchart(registry)
        phase_flowchart = self.mermaid_generator.create_phase_implementation_flowchart()
        
        # Create documentation
        doc_content = f"""# Distributed Agentic Cognitive Grammar Network Architecture

## Overview

The Distributed Agentic Cognitive Grammar Network is a novel architecture that bridges
symbolic reasoning and neural computation through RWKV-based tensor operations.

## Network Architecture

```mermaid
{network_flowchart}
```

## Tensor Shape Registry

```mermaid
{registry_flowchart}
```

## Implementation Phases

```mermaid
{phase_flowchart}
```

## Current Network Status

- **Total Fragments**: {len(network.fragments)}
- **Total Nodes**: {sum(len(f.nodes) for f in network.fragments.values())}
- **Pattern Library Size**: {len(network.pattern_library)}
- **Device**: {network.device}
- **AtomSpace Integration**: {network.atomspace_adapter is not None}

## Registry Statistics

{self._format_registry_summary(registry)}

## API Reference

### Core Classes

#### CognitiveGrammarNetwork
Main orchestrator for the distributed cognitive grammar network.

```python
network = CognitiveGrammarNetwork(device='cpu')
fragment = network.create_agentic_fragment('test_agent')
```

#### TensorFragment
Encodes agents/states as hypergraph nodes & links with tensor representations.

```python
fragment = TensorFragment(device='cpu')
node = HypergraphNode(node_type='cognitive_entity')
fragment.add_node(node)
```

#### TensorShapeRegistry
Manages tensor shapes and their prime factorization mappings.

```python
registry = TensorShapeRegistry()
kernel_shape = KernelTensorShape(...)
registry.register_kernel_shape(kernel_shape)
```

## Usage Examples

### Basic Network Creation

```python
from cognitive_grammar import CognitiveGrammarNetwork, ModalityType

# Create network
network = CognitiveGrammarNetwork()

# Create agentic fragment
fragment = network.create_agentic_fragment(
    agent_type="linguistic_agent",
    modality=ModalityType.LINGUISTIC,
    depth=4,
    context=8
)

# Register fragment
fragment_id = network.register_fragment(fragment)
```

### Scheme Grammar Processing

```python
from cognitive_grammar.adapters import SchemeAdapter

# Create adapter
adapter = SchemeAdapter()

# Parse agentic grammar
grammar_text = "(action move (agent robot) (target location))"
fragments = adapter.parse_agentic_grammar(grammar_text)
```

### Tensor Shape Management

```python
from cognitive_grammar.tensor_registry import TensorShapeRegistry, PrimeFactorizationMapper

# Create registry
registry = TensorShapeRegistry()

# Create mapper
mapper = PrimeFactorizationMapper()

# Suggest kernel shapes
suggestions = mapper.suggest_kernel_shapes(
    function_complexity=3,
    input_modalities=[ModalityType.LINGUISTIC],
    output_requirements={{'context_size': 16}}
)
```

## Verification and Testing

The framework includes comprehensive verification:

- Pattern transformation tests
- Hypergraph integrity verification  
- Bidirectional translation consistency
- Tensor-hypergraph correspondence

```python
from cognitive_grammar.verification import ComprehensiveVerificationSuite

# Create verification suite
verifier = ComprehensiveVerificationSuite()
verifier.setup_default_tests()

# Run verification
report = verifier.run_full_verification_suite(network)
print(f"Success rate: {{report.success_rate:.2%}}")
```

## Next Steps: Phase 2 Implementation

1. **ECAN Attention Allocation**: Implement resource allocators
2. **Dynamic Mesh Integration**: Benchmark attention allocation
3. **Real-World Verification**: Schedule real tasks with live data

## Contributing

This is Phase 1 of a 6-phase implementation. Each phase builds upon the 
foundational architecture established here.

Generated on: {self._get_timestamp()}
"""
        
        # Save documentation
        doc_path = self.output_dir / "architecture.md"
        with open(doc_path, 'w') as f:
            f.write(doc_content)
        
        return doc_content
    
    def _format_registry_summary(self, registry: TensorShapeRegistry) -> str:
        """Format registry summary for documentation"""
        summary = registry.get_registry_summary()
        
        return f"""
- **Total Kernels**: {summary.get('total_kernels', 0)}
- **Total Parameters**: {summary.get('total_parameters', 0):,}
- **Average Complexity**: {summary.get('average_complexity', 0):.2f}
- **Modality Distribution**: {summary.get('modality_distribution', {})}
"""
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def generate_api_documentation(self, network: CognitiveGrammarNetwork) -> str:
        """Generate API documentation"""
        
        api_doc = """# Cognitive Grammar Network API Documentation

## Core API

### CognitiveGrammarNetwork

The main orchestrator class for the distributed cognitive grammar network.

#### Constructor

```python
CognitiveGrammarNetwork(device='cpu', dtype=torch.float32)
```

**Parameters:**
- `device` (str): Computing device ('cpu' or 'cuda')
- `dtype` (torch.dtype): Default tensor data type

#### Methods

##### create_agentic_fragment()

Creates a new agentic tensor fragment.

```python
create_agentic_fragment(
    agent_type: str,
    modality: ModalityType = ModalityType.AGENTIC,
    depth: int = 4,
    context: int = 8
) -> TensorFragment
```

##### register_fragment()

Registers a tensor fragment in the network.

```python
register_fragment(fragment: TensorFragment) -> str
```

##### apply_transformation_pattern()

Applies a stored transformation pattern to a fragment.

```python
apply_transformation_pattern(
    fragment_id: str,
    pattern_name: str
) -> TensorFragment
```

### TensorFragment

Encodes agents/states as hypergraph nodes & links.

#### Constructor

```python
TensorFragment(
    fragment_id: str = None,
    nodes: List[HypergraphNode] = None,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32
)
```

#### Methods

##### add_node()

Adds a hypergraph node to the fragment.

```python
add_node(node: HypergraphNode) -> HypergraphNode
```

##### pattern_transformation()

Applies pattern transformation to the fragment.

```python
pattern_transformation(transformation_matrix: torch.Tensor) -> TensorFragment
```

### HypergraphNode

Represents an agent, state, or cognitive entity.

#### Constructor

```python
HypergraphNode(
    node_id: str = None,
    node_type: str = "cognitive_entity",
    tensor_signature: TensorSignature = None,
    semantic_content: Dict[str, Any] = None
)
```

#### Methods

##### create_tensor_state()

Creates tensor state from signature.

```python
create_tensor_state(device='cpu', dtype=torch.float32) -> torch.Tensor
```

##### add_link()

Adds hypergraph link to another node.

```python
add_link(
    target_node: HypergraphNode,
    link_type: str,
    weight: float = 1.0
)
```

## Adapter API

### SchemeAdapter

Scheme adapter for agentic grammar microservices.

#### Methods

##### parse_agentic_grammar()

Parses agentic grammar text into tensor fragments.

```python
parse_agentic_grammar(grammar_text: str) -> List[TensorFragment]
```

##### add_grammar_rule()

Adds a cognitive grammar rule.

```python
add_grammar_rule(
    rule_name: str,
    pattern: str,
    transformation: str
)
```

### BiDirectionalTranslator

Bidirectional translator between cognitive grammar and tensor representations.

#### Methods

##### scheme_to_tensor_fragment()

Converts Scheme expression to tensor fragment.

```python
scheme_to_tensor_fragment(scheme_expr: SchemeExpression) -> TensorFragment
```

##### verify_bidirectional_consistency()

Verifies bidirectional translation preserves semantics.

```python
verify_bidirectional_consistency(scheme_expr: SchemeExpression) -> bool
```

## Registry API

### TensorShapeRegistry

Registry for cognitive kernel tensor shapes.

#### Methods

##### register_kernel_shape()

Registers a new kernel tensor shape.

```python
register_kernel_shape(kernel_shape: KernelTensorShape) -> str
```

##### find_compatible_kernels()

Finds kernels compatible with the given kernel.

```python
find_compatible_kernels(kernel_name: str) -> List[KernelTensorShape]
```

### PrimeFactorizationMapper

Prime factorization mapper for tensor dimensions.

#### Methods

##### create_tensor_signature_from_semantics()

Creates tensor signature from semantic specifications.

```python
create_tensor_signature_from_semantics(
    semantic_specs: Dict[SemanticDimension, int]
) -> TensorSignature
```

## Verification API

### ComprehensiveVerificationSuite

Comprehensive verification suite for the cognitive grammar network.

#### Methods

##### run_full_verification_suite()

Runs complete verification suite.

```python
run_full_verification_suite(
    network: CognitiveGrammarNetwork,
    additional_fragments: List[TensorFragment] = None
) -> VerificationReport
```

## Error Handling

All API methods include proper error handling and validation:

```python
try:
    fragment = network.create_agentic_fragment("test_agent")
except ValueError as e:
    print(f"Invalid agent type: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Threading and Performance

The framework is designed for efficient computation:

- Tensor operations use PyTorch's optimized backends
- CUDA support for GPU acceleration
- Batch processing for multiple fragments
- Memory-efficient tensor storage

## Integration Examples

### With RWKV-v7

```python
# Integration with existing RWKV model
import torch
from cognitive_grammar import CognitiveGrammarNetwork

# Create network using same device as RWKV
device = 'cuda' if torch.cuda.is_available() else 'cpu'
network = CognitiveGrammarNetwork(device=device)

# Create fragments that can interface with RWKV tensors
fragment = network.create_agentic_fragment(
    agent_type="rwkv_interface",
    depth=24,  # Match RWKV layer depth
    context=4096  # Match RWKV context length
)
```

### With External Systems

```python
# REST API integration example
from flask import Flask, request, jsonify

app = Flask(__name__)
network = CognitiveGrammarNetwork()

@app.route('/api/cognitive_grammar/parse', methods=['POST'])
def parse_grammar():
    grammar_text = request.json['grammar']
    adapter = SchemeAdapter()
    fragments = adapter.parse_agentic_grammar(grammar_text)
    
    return jsonify({
        'fragments_created': len(fragments),
        'fragment_ids': [f.fragment_id for f in fragments]
    })
```
"""
        
        # Save API documentation
        api_path = self.output_dir / "api.md"
        with open(api_path, 'w') as f:
            f.write(api_doc)
        
        return api_doc
    
    def generate_examples_documentation(self) -> str:
        """Generate examples and tutorials documentation"""
        
        examples_doc = """# Cognitive Grammar Network Examples

## Basic Usage Examples

### Example 1: Creating Your First Cognitive Network

```python
from cognitive_grammar import CognitiveGrammarNetwork, ModalityType

# Initialize the network
network = CognitiveGrammarNetwork(device='cpu')

# Create an agentic fragment for natural language processing
language_fragment = network.create_agentic_fragment(
    agent_type="language_processor",
    modality=ModalityType.LINGUISTIC,
    depth=4,
    context=16
)

# Register the fragment
fragment_id = network.register_fragment(language_fragment)
print(f"Created fragment: {fragment_id}")

# Get network summary
summary = network.get_network_summary()
print(f"Network has {summary['total_fragments']} fragments")
```

### Example 2: Scheme Grammar Processing

```python
from cognitive_grammar.adapters import SchemeAdapter, SchemeExpression

# Create adapter
adapter = SchemeAdapter()

# Define cognitive grammar in Scheme
grammar_examples = [
    "(action move (agent robot) (target kitchen))",
    "(cognitive-state (agent self) (emotion happy) (confidence 0.8))", 
    "(relation (subject Alice) (predicate teaches) (object Mathematics))",
    "(planning (goal complete-task) (steps (step1 analyze) (step2 execute)))"
]

# Process each grammar expression
for grammar in grammar_examples:
    print(f"\\nProcessing: {grammar}")
    
    # Parse into tensor fragments
    fragments = adapter.parse_agentic_grammar(grammar)
    
    for fragment in fragments:
        print(f"  Fragment: {fragment.fragment_id[:8]}")
        print(f"  Nodes: {len(fragment.nodes)}")
        
        # Test bidirectional consistency
        if fragment.nodes:
            from cognitive_grammar.adapters import BiDirectionalTranslator
            translator = BiDirectionalTranslator()
            
            # Convert back to Scheme
            reconstructed = translator.tensor_fragment_to_scheme(fragment)
            print(f"  Reconstructed: {reconstructed.expression}")
```

### Example 3: Tensor Shape Registry Usage

```python
from cognitive_grammar.tensor_registry import (
    TensorShapeRegistry, 
    PrimeFactorizationMapper,
    SemanticDimension
)

# Create registry and mapper
registry = TensorShapeRegistry()
mapper = PrimeFactorizationMapper()

# Define semantic requirements for a memory kernel
memory_semantics = {
    SemanticDimension.MODALITY: 2,  # Text and visual
    SemanticDimension.DEPTH: 5,     # Deep processing
    SemanticDimension.CONTEXT: 8,   # Large context window
    SemanticDimension.SALIENCE: 3,  # Medium attention levels
    SemanticDimension.AUTONOMY_INDEX: 2  # Semi-autonomous
}

# Create tensor signature
signature = mapper.create_tensor_signature_from_semantics(memory_semantics)
print(f"Generated signature: {signature.shape}")
print(f"Prime factorization: {signature.prime_factors}")

# Create and register kernel shape
from cognitive_grammar.tensor_registry import KernelTensorShape

kernel_shape = KernelTensorShape(
    kernel_name="memory_kernel",
    tensor_signature=signature,
    semantic_dimensions=memory_semantics,
    metadata={"purpose": "episodic_memory", "version": "1.0"}
)

registry.register_kernel_shape(kernel_shape)

# Find compatible kernels
compatible = registry.find_compatible_kernels("memory_kernel")
print(f"Compatible kernels: {len(compatible)}")

# Get optimization suggestions
suggestions = registry.suggest_optimizations("memory_kernel")
for suggestion in suggestions:
    print(f"Optimization: {suggestion['type']} - {suggestion['message']}")
```

### Example 4: Pattern Transformation

```python
import torch
from cognitive_grammar import CognitiveGrammarNetwork, TensorFragment

# Create network and fragment
network = CognitiveGrammarNetwork()
fragment = network.create_agentic_fragment("transformer_agent", depth=3, context=6)

# Define a rotation transformation pattern
rotation_angle = torch.pi / 4  # 45 degrees
if fragment.nodes and fragment.nodes[0].tensor_state is not None:
    tensor_size = fragment.nodes[0].tensor_state.numel()
    
    # Create rotation matrix (simplified 2D rotation for demonstration)
    rotation_matrix = torch.eye(tensor_size)
    
    # Add transformation to pattern library
    network.add_transformation_pattern("rotation_45", rotation_matrix)
    
    # Apply transformation
    transformed_fragment = network.apply_transformation_pattern(
        fragment.fragment_id, 
        "rotation_45"
    )
    
    print(f"Original fragment: {fragment.fragment_id[:8]}")
    print(f"Transformed fragment: {transformed_fragment.fragment_id[:8]}")
```

### Example 5: Comprehensive Verification

```python
from cognitive_grammar.verification import ComprehensiveVerificationSuite

# Create verification suite
verifier = ComprehensiveVerificationSuite()

# Set up default tests
verifier.setup_default_tests()

# Add custom verification rule
def check_memory_efficiency(fragment):
    \"\"\"Custom rule to check memory efficiency\"\"\"
    total_params = sum(
        node.tensor_state.numel() 
        for node in fragment.nodes 
        if node.tensor_state is not None
    )
    
    # Consider efficient if less than 100K parameters
    efficient = total_params < 100000
    
    return {
        'passed': efficient,
        'details': {
            'total_parameters': total_params,
            'efficiency_threshold': 100000
        }
    }

verifier.hypergraph_verifier.add_verification_rule(
    "memory_efficiency", 
    check_memory_efficiency
)

# Run full verification
report = verifier.run_full_verification_suite(network)

print(f"Verification Results:")
print(f"  Total tests: {report.total_tests}")
print(f"  Passed: {report.passed_tests}")
print(f"  Failed: {report.failed_tests}")
print(f"  Success rate: {report.success_rate:.2%}")
print(f"  Execution time: {report.execution_time:.2f}s")

# Generate detailed report
verifier.generate_verification_report_file(report, "verification_results.json")
```

### Example 6: Visualization and Documentation

```python
from cognitive_grammar.visualization import (
    MermaidFlowchartGenerator,
    DocumentationGenerator
)

# Create visualization generator
mermaid_gen = MermaidFlowchartGenerator()

# Generate network architecture flowchart
network_flowchart = mermaid_gen.create_network_architecture_flowchart(network)
print("Network Architecture Flowchart:")
print(network_flowchart)

# Generate hypergraph flowchart for a fragment
if network.fragments:
    fragment = list(network.fragments.values())[0]
    hypergraph_flowchart = mermaid_gen.create_hypergraph_flowchart(fragment)
    print("\\nHypergraph Structure Flowchart:")
    print(hypergraph_flowchart)

# Generate documentation
doc_gen = DocumentationGenerator(output_dir="generated_docs")
architecture_doc = doc_gen.generate_architecture_documentation(network, registry)

print("\\nDocumentation generated in 'generated_docs' directory")
```

## Advanced Usage Patterns

### Integration with RWKV Models

```python
# Example: Creating cognitive layer for RWKV
class CognitiveRWKVLayer:
    def __init__(self, rwkv_layer, cognitive_network):
        self.rwkv_layer = rwkv_layer
        self.cognitive_network = cognitive_network
        
        # Create cognitive fragment matching RWKV dimensions
        self.cognitive_fragment = cognitive_network.create_agentic_fragment(
            agent_type="rwkv_cognitive_layer",
            depth=rwkv_layer.n_layer if hasattr(rwkv_layer, 'n_layer') else 4,
            context=getattr(rwkv_layer, 'ctx_len', 1024)
        )
    
    def forward(self, x, state=None):
        # Standard RWKV forward pass
        rwkv_output, rwkv_state = self.rwkv_layer(x, state)
        
        # Cognitive processing
        # (This would integrate tensor operations from cognitive fragment)
        
        return rwkv_output, rwkv_state

# Usage
# rwkv_model = load_rwkv_model()
# cognitive_network = CognitiveGrammarNetwork()
# cognitive_layer = CognitiveRWKVLayer(rwkv_model.layers[0], cognitive_network)
```

### Distributed Processing

```python
# Example: Distributed cognitive processing
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

def process_fragment(fragment_data):
    \"\"\"Process a fragment in separate process\"\"\"
    network = CognitiveGrammarNetwork()
    
    # Recreate fragment from serialized data
    # (Implementation details would depend on serialization method)
    
    # Process fragment
    result = perform_cognitive_processing(fragment)
    
    return result

# Distribute fragments across processes
def distributed_cognitive_processing(fragments):
    with ProcessPoolExecutor() as executor:
        # Submit fragments for processing
        futures = [
            executor.submit(process_fragment, fragment.serialize())
            for fragment in fragments
        ]
        
        # Collect results
        results = [future.result() for future in futures]
    
    return results
```

### Real-time Cognitive Grammar Processing

```python
# Example: Real-time processing pipeline
import asyncio
from asyncio import Queue

class CognitiveGrammarProcessor:
    def __init__(self):
        self.network = CognitiveGrammarNetwork()
        self.adapter = SchemeAdapter()
        self.input_queue = Queue()
        self.output_queue = Queue()
    
    async def process_stream(self):
        \"\"\"Process cognitive grammar in real-time\"\"\"
        while True:
            try:
                # Get input from queue
                grammar_input = await asyncio.wait_for(
                    self.input_queue.get(), 
                    timeout=1.0
                )
                
                # Process grammar
                fragments = self.adapter.parse_agentic_grammar(grammar_input)
                
                # Apply cognitive transformations
                for fragment in fragments:
                    self.network.register_fragment(fragment)
                    
                    # Apply pattern transformations
                    if "attention" in self.network.pattern_library:
                        transformed = self.network.apply_transformation_pattern(
                            fragment.fragment_id, 
                            "attention"
                        )
                        
                        # Put result in output queue
                        await self.output_queue.put({
                            'original_fragment': fragment.fragment_id,
                            'transformed_fragment': transformed.fragment_id,
                            'processing_time': time.time()
                        })
                
            except asyncio.TimeoutError:
                # Continue processing if no input
                continue
            except Exception as e:
                print(f"Processing error: {e}")

# Usage
# processor = CognitiveGrammarProcessor()
# asyncio.run(processor.process_stream())
```

## Performance Optimization Tips

1. **Use GPU acceleration when available:**
   ```python
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   network = CognitiveGrammarNetwork(device=device)
   ```

2. **Batch process multiple fragments:**
   ```python
   # Process multiple fragments together for better efficiency
   fragments = [fragment1, fragment2, fragment3]
   collective_tensors = [f.create_collective_tensor() for f in fragments]
   batched_tensor = torch.stack(collective_tensors)
   ```

3. **Optimize tensor shapes for your use case:**
   ```python
   # Use prime factorization mapper to find optimal shapes
   optimal_shapes = mapper.suggest_kernel_shapes(
       function_complexity=your_complexity,
       input_modalities=your_modalities,
       output_requirements=your_requirements
   )
   ```

4. **Cache frequently used patterns:**
   ```python
   # Pre-compute and cache transformation matrices
   network.add_transformation_pattern("common_transform", precomputed_matrix)
   ```

These examples demonstrate the flexibility and power of the Cognitive Grammar Network
framework. Each example builds upon the foundational concepts while showing practical
applications for various use cases.
"""
        
        # Save examples documentation
        examples_path = self.output_dir / "examples.md"
        with open(examples_path, 'w') as f:
            f.write(examples_doc)
        
        return examples_doc
    
    def generate_complete_documentation_suite(self, 
                                            network: CognitiveGrammarNetwork,
                                            registry: TensorShapeRegistry) -> Dict[str, str]:
        """Generate complete documentation suite"""
        
        docs = {}
        
        # Generate all documentation types
        docs['architecture'] = self.generate_architecture_documentation(network, registry)
        docs['api'] = self.generate_api_documentation(network)
        docs['examples'] = self.generate_examples_documentation()
        
        # Create index file
        index_content = f"""# Distributed Agentic Cognitive Grammar Network Documentation

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
print(f"Created fragment: {{fragment_id}}")
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

Generated: {self._get_timestamp()}
Network Status: {len(network.fragments)} fragments, {sum(len(f.nodes) for f in network.fragments.values())} nodes
"""
        
        # Save index
        index_path = self.output_dir / "README.md"
        with open(index_path, 'w') as f:
            f.write(index_content)
        
        docs['index'] = index_content
        
        print(f"Complete documentation suite generated in {self.output_dir}")
        return docs