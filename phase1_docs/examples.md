# Cognitive Grammar Network Examples

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
    print(f"\nProcessing: {grammar}")
    
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
    """Custom rule to check memory efficiency"""
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
    print("\nHypergraph Structure Flowchart:")
    print(hypergraph_flowchart)

# Generate documentation
doc_gen = DocumentationGenerator(output_dir="generated_docs")
architecture_doc = doc_gen.generate_architecture_documentation(network, registry)

print("\nDocumentation generated in 'generated_docs' directory")
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
    """Process a fragment in separate process"""
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
        """Process cognitive grammar in real-time"""
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
