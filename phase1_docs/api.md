# Cognitive Grammar Network API Documentation

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
