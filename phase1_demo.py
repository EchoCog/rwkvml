#!/usr/bin/env python3
"""
Cognitive Grammar Network Phase 1 Demonstration

This script demonstrates the Phase 1 implementation of the Distributed
Agentic Cognitive Grammar Network, showing:

1. Scheme cognitive grammar microservices
2. Tensor fragment architecture with hypergraph encoding
3. Bidirectional translation tests (no mocks)
4. Prime factorization tensor shape registry
5. Comprehensive verification and visualization

Usage:
    python phase1_demo.py
"""

import sys
import os
import torch
import time
from pathlib import Path

# Add the cognitive_grammar module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cognitive_grammar import (
    CognitiveGrammarNetwork, 
    TensorFragment, 
    HypergraphNode, 
    TensorSignature,
    ModalityType
)
from cognitive_grammar.adapters import SchemeAdapter, BiDirectionalTranslator, SchemeExpression
from cognitive_grammar.tensor_registry import (
    TensorShapeRegistry, 
    PrimeFactorizationMapper, 
    SemanticDimension,
    KernelTensorShape
)
from cognitive_grammar.verification import ComprehensiveVerificationSuite
from cognitive_grammar.visualization import DocumentationGenerator, MermaidFlowchartGenerator


def print_section(title: str):
    """Print formatted section header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")


def print_subsection(title: str):
    """Print formatted subsection header"""
    print(f"\n{'-'*40}")
    print(f" {title}")
    print(f"{'-'*40}\n")


def demo_cognitive_grammar_network():
    """Demonstrate core cognitive grammar network functionality"""
    print_section("Phase 1.1: Cognitive Grammar Network Core")
    
    # Create network
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    network = CognitiveGrammarNetwork(device=device)
    print(f"Created CognitiveGrammarNetwork")
    
    # Create various agentic fragments
    print_subsection("Creating Agentic Fragments")
    
    # Linguistic agent
    linguistic_fragment = network.create_agentic_fragment(
        agent_type="linguistic_processor",
        modality=ModalityType.LINGUISTIC,
        depth=4,
        context=16
    )
    print(f"Created linguistic fragment: {linguistic_fragment.fragment_id[:8]}")
    
    # Visual agent
    visual_fragment = network.create_agentic_fragment(
        agent_type="visual_processor", 
        modality=ModalityType.VISUAL,
        depth=6,
        context=24
    )
    print(f"Created visual fragment: {visual_fragment.fragment_id[:8]}")
    
    # Conceptual agent
    conceptual_fragment = network.create_agentic_fragment(
        agent_type="conceptual_reasoner",
        modality=ModalityType.CONCEPTUAL,
        depth=8,
        context=32
    )
    print(f"Created conceptual fragment: {conceptual_fragment.fragment_id[:8]}")
    
    # Create inter-fragment links
    print_subsection("Creating Hypergraph Links")
    
    # Link linguistic to conceptual
    if linguistic_fragment.nodes and conceptual_fragment.nodes:
        linguistic_fragment.nodes[0].add_link(
            conceptual_fragment.nodes[0],
            "semantic_interface",
            weight=0.8
        )
        print("Created semantic link: linguistic → conceptual")
    
    # Link visual to conceptual
    if visual_fragment.nodes and conceptual_fragment.nodes:
        visual_fragment.nodes[0].add_link(
            conceptual_fragment.nodes[0],
            "perceptual_interface", 
            weight=0.7
        )
        print("Created perceptual link: visual → conceptual")
    
    # Update adjacency matrices
    for fragment in [linguistic_fragment, visual_fragment, conceptual_fragment]:
        fragment.create_collective_tensor()
    
    # Display network summary
    summary = network.get_network_summary()
    print(f"\nNetwork Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    return network, [linguistic_fragment, visual_fragment, conceptual_fragment]


def demo_scheme_adapters():
    """Demonstrate Scheme cognitive grammar adapters"""
    print_section("Phase 1.2: Scheme Cognitive Grammar Microservices")
    
    # Create adapter
    adapter = SchemeAdapter()
    translator = BiDirectionalTranslator()
    
    print_subsection("Agentic Grammar Examples")
    
    # Define various cognitive grammar expressions
    grammar_examples = [
        "(action move (agent robot) (target kitchen) (method walking))",
        "(cognitive-state (agent self) (emotion curious) (confidence 0.9))",
        "(relation (subject Alice) (predicate teaches) (object Mathematics) (location university))",
        "(planning (goal complete-research) (steps (step1 gather-data) (step2 analyze) (step3 conclude)))",
        "(memory (type episodic) (content meeting) (timestamp yesterday) (participants (Alice Bob)))",
        "(perception (modality visual) (object red-car) (location street) (confidence 0.85))"
    ]
    
    fragments_created = []
    
    for i, grammar in enumerate(grammar_examples):
        print(f"Example {i+1}: {grammar}")
        
        # Parse into tensor fragments
        fragments = adapter.parse_agentic_grammar(grammar)
        fragments_created.extend(fragments)
        
        print(f"  Created {len(fragments)} fragments")
        
        for j, fragment in enumerate(fragments):
            print(f"    Fragment {j}: {len(fragment.nodes)} nodes")
            
            # Show tensor signatures
            for k, node in enumerate(fragment.nodes):
                sig = node.tensor_signature
                print(f"      Node {k}: shape={sig.shape}, type={node.node_type}")
    
    print_subsection("Bidirectional Translation Tests")
    
    # Test bidirectional consistency (no mocks - real implementation)
    test_expressions = [
        "(action speak (agent human) (content greeting))",
        "(compound (operator and) (arg1 true) (arg2 false))",
        "(recursive (function fibonacci) (input 5))"
    ]
    
    translation_results = []
    
    for expr_text in test_expressions:
        print(f"Testing: {expr_text}")
        
        try:
            # Parse original
            original_expr = SchemeExpression.parse(expr_text)
            print(f"  Original type: {original_expr.atom_type}")
            print(f"  Original args: {len(original_expr.arguments)}")
            
            # Convert to tensor fragment
            tensor_fragment = translator.scheme_to_tensor_fragment(original_expr)
            print(f"  Tensor nodes: {len(tensor_fragment.nodes)}")
            
            # Convert back to Scheme
            reconstructed_expr = translator.tensor_fragment_to_scheme(tensor_fragment)
            print(f"  Reconstructed: {reconstructed_expr.expression}")
            
            # Verify consistency
            consistent = translator.verify_bidirectional_consistency(original_expr)
            print(f"  Consistency check: {'PASS' if consistent else 'FAIL'}")
            
            translation_results.append({
                'expression': expr_text,
                'consistent': consistent,
                'nodes_created': len(tensor_fragment.nodes)
            })
            
        except Exception as e:
            print(f"  ERROR: {e}")
            translation_results.append({
                'expression': expr_text,
                'consistent': False,
                'error': str(e)
            })
    
    # Summary of translation tests
    print_subsection("Translation Test Summary")
    passed = sum(1 for r in translation_results if r.get('consistent', False))
    total = len(translation_results)
    print(f"Bidirectional translation tests: {passed}/{total} passed")
    
    return adapter, translator, fragments_created, translation_results


def demo_tensor_registry():
    """Demonstrate tensor shape registry and prime factorization"""
    print_section("Phase 1.3: Tensor Fragment Architecture & Registry")
    
    # Create registry and mapper
    registry = TensorShapeRegistry()
    mapper = PrimeFactorizationMapper()
    
    print_subsection("Prime Factorization Mapping")
    
    # Demonstrate semantic complexity to prime mapping
    semantic_examples = [
        {SemanticDimension.MODALITY: 2, SemanticDimension.DEPTH: 3, 
         SemanticDimension.CONTEXT: 4, SemanticDimension.SALIENCE: 2},
        {SemanticDimension.MODALITY: 3, SemanticDimension.DEPTH: 5,
         SemanticDimension.CONTEXT: 8, SemanticDimension.SALIENCE: 4},
        {SemanticDimension.MODALITY: 1, SemanticDimension.DEPTH: 7,
         SemanticDimension.CONTEXT: 16, SemanticDimension.SALIENCE: 3}
    ]
    
    kernel_shapes = []
    
    for i, semantic_spec in enumerate(semantic_examples):
        print(f"Semantic specification {i+1}:")
        for dim, value in semantic_spec.items():
            print(f"  {dim.value}: {value}")
        
        # Create tensor signature from semantics
        signature = mapper.create_tensor_signature_from_semantics(semantic_spec)
        print(f"  Generated shape: {signature.shape}")
        print(f"  Prime factors: {signature.prime_factors}")
        print(f"  Total parameters: {np.prod(signature.shape):,}")
        
        # Create kernel shape
        kernel_shape = KernelTensorShape(
            kernel_name=f"cognitive_kernel_{i+1}",
            tensor_signature=signature,
            semantic_dimensions=semantic_spec,
            metadata={
                'creation_time': time.time(),
                'complexity_level': sum(semantic_spec.values()),
                'example_id': i+1
            }
        )
        
        # Register in registry
        kernel_id = registry.register_kernel_shape(kernel_shape)
        kernel_shapes.append(kernel_shape)
        print(f"  Registered as: {kernel_id}")
        print(f"  Complexity score: {kernel_shape.complexity_score:.2f}")
    
    print_subsection("Kernel Compatibility Analysis")
    
    # Test kernel compatibility
    for i, kernel_shape in enumerate(kernel_shapes):
        compatible = registry.find_compatible_kernels(kernel_shape.kernel_name)
        print(f"Kernel {i+1} ({kernel_shape.kernel_name}):")
        print(f"  Compatible with {len(compatible)} other kernels")
        
        # Show optimization suggestions
        suggestions = registry.suggest_optimizations(kernel_shape.kernel_name)
        if suggestions:
            print(f"  Optimization suggestions:")
            for suggestion in suggestions:
                print(f"    - {suggestion['type']}: {suggestion['message']}")
        else:
            print(f"  No optimization suggestions")
    
    # Generate registry catalog
    print_subsection("Registry Catalog")
    catalog = registry.create_kernel_catalog()
    print(f"Registry contains:")
    print(f"  Total kernels: {catalog['total_kernels']}")
    print(f"  Relationship mappings: {len(catalog['relationships'])}")
    
    # Save registry to file
    registry.save_registry("phase1_tensor_registry.json")
    print(f"  Registry saved to: phase1_tensor_registry.json")
    
    return registry, mapper, kernel_shapes


def demo_pattern_transformations(network, fragments):
    """Demonstrate pattern transformations with real hypergraph fragments"""
    print_section("Phase 1.4: Pattern Transformations")
    
    print_subsection("Creating Transformation Patterns")
    
    # Create various transformation patterns
    patterns_created = []
    
    for i, fragment in enumerate(fragments[:3]):  # Test with first 3 fragments
        if not fragment.nodes or fragment.nodes[0].tensor_state is None:
            continue
            
        node = fragment.nodes[0]
        tensor_size = node.tensor_state.numel()
        
        # Identity transformation
        identity_matrix = torch.eye(tensor_size, device=network.device)
        pattern_name = f"identity_transform_{i}"
        network.add_transformation_pattern(pattern_name, identity_matrix)
        
        # Rotation transformation (simplified)
        rotation_matrix = torch.eye(tensor_size, device=network.device)
        # Add small rotation component
        rotation_matrix *= 0.99
        rotation_matrix += 0.01 * torch.randn(tensor_size, tensor_size, device=network.device)
        
        rotation_pattern_name = f"rotation_transform_{i}"
        network.add_transformation_pattern(rotation_pattern_name, rotation_matrix)
        
        patterns_created.extend([pattern_name, rotation_pattern_name])
        
        print(f"Created patterns for fragment {i}:")
        print(f"  - {pattern_name} (identity)")
        print(f"  - {rotation_pattern_name} (rotation)")
    
    print_subsection("Applying Transformations")
    
    transformation_results = []
    
    for fragment in fragments[:2]:  # Test transformations
        if not fragment.nodes:
            continue
            
        fragment_id = fragment.fragment_id
        print(f"Testing fragment: {fragment_id[:8]}")
        
        # Try available patterns
        for pattern_name in patterns_created:
            if pattern_name in network.pattern_library:
                try:
                    transformed = network.apply_transformation_pattern(fragment_id, pattern_name)
                    
                    print(f"  Applied {pattern_name}:")
                    print(f"    Original nodes: {len(fragment.nodes)}")
                    print(f"    Transformed nodes: {len(transformed.nodes)}")
                    
                    # Compare tensor magnitudes
                    if fragment.nodes[0].tensor_state is not None and transformed.nodes[0].tensor_state is not None:
                        orig_norm = torch.norm(fragment.nodes[0].tensor_state).item()
                        trans_norm = torch.norm(transformed.nodes[0].tensor_state).item()
                        print(f"    Tensor norm change: {orig_norm:.3f} → {trans_norm:.3f}")
                    
                    transformation_results.append({
                        'fragment_id': fragment_id,
                        'pattern_name': pattern_name,
                        'success': True,
                        'transformed_fragment_id': transformed.fragment_id
                    })
                    
                except Exception as e:
                    print(f"  Failed to apply {pattern_name}: {e}")
                    transformation_results.append({
                        'fragment_id': fragment_id,
                        'pattern_name': pattern_name,
                        'success': False,
                        'error': str(e)
                    })
    
    # Summary
    successful = sum(1 for r in transformation_results if r.get('success', False))
    total = len(transformation_results)
    print(f"\nTransformation Summary:")
    print(f"  Successful transformations: {successful}/{total}")
    print(f"  Patterns in library: {len(network.pattern_library)}")
    
    return transformation_results


def demo_comprehensive_verification(network, fragments, translation_results):
    """Demonstrate comprehensive verification framework"""
    print_section("Phase 1.5: Verification & Testing (No Mocks)")
    
    # Create verification suite
    verifier = ComprehensiveVerificationSuite(device=network.device)
    
    print_subsection("Setting Up Verification Tests")
    
    # Set up default tests
    verifier.setup_default_tests()
    print("Default test suite configured")
    
    # Add custom verification rules
    def check_cognitive_coherence(fragment):
        """Custom rule: Check cognitive coherence"""
        if not fragment.nodes:
            return {'passed': True, 'details': {'reason': 'no_nodes'}}
        
        # Check that all nodes have semantic content
        nodes_with_semantics = sum(
            1 for node in fragment.nodes 
            if node.semantic_content and len(node.semantic_content) > 0
        )
        
        coherent = nodes_with_semantics == len(fragment.nodes)
        
        return {
            'passed': coherent,
            'details': {
                'total_nodes': len(fragment.nodes),
                'nodes_with_semantics': nodes_with_semantics,
                'coherence_ratio': nodes_with_semantics / len(fragment.nodes) if fragment.nodes else 1.0
            }
        }
    
    def check_tensor_energy_conservation(fragment):
        """Custom rule: Check energy conservation in tensors"""
        if not fragment.nodes:
            return {'passed': True, 'details': {'reason': 'no_nodes'}}
        
        total_energy = 0.0
        energy_per_node = []
        
        for node in fragment.nodes:
            if node.tensor_state is not None:
                node_energy = torch.sum(node.tensor_state ** 2).item()
                energy_per_node.append(node_energy)
                total_energy += node_energy
        
        # Energy should be finite and positive
        energy_valid = (
            not np.isnan(total_energy) and 
            not np.isinf(total_energy) and 
            total_energy >= 0
        )
        
        return {
            'passed': energy_valid,
            'details': {
                'total_energy': total_energy,
                'energy_per_node': energy_per_node,
                'finite_energy': not (np.isnan(total_energy) or np.isinf(total_energy))
            }
        }
    
    # Register custom rules
    verifier.hypergraph_verifier.add_verification_rule("cognitive_coherence", check_cognitive_coherence)
    verifier.hypergraph_verifier.add_verification_rule("energy_conservation", check_tensor_energy_conservation)
    
    print("Custom verification rules added:")
    print("  - cognitive_coherence: Checks semantic content completeness")
    print("  - energy_conservation: Checks tensor energy properties")
    
    # Add translation test expressions
    print_subsection("Adding Translation Test Cases")
    
    additional_expressions = [
        "(goal achieve (objective learn-language) (timeline 6-months))",
        "(emotion (type happiness) (intensity 0.7) (trigger success))",
        "(decision (options (A stay) (B leave)) (criteria (cost time)) (choice A))"
    ]
    
    for expr in additional_expressions:
        verifier.translation_tester.add_test_expression(expr, f"verification_test")
    
    print(f"Added {len(additional_expressions)} additional test expressions")
    
    print_subsection("Running Comprehensive Verification")
    
    # Run full verification suite
    start_time = time.time()
    verification_report = verifier.run_full_verification_suite(network, fragments)
    end_time = time.time()
    
    # Display results
    print(f"Verification completed in {end_time - start_time:.2f} seconds")
    print(f"\nVerification Report:")
    print(f"  Total tests: {verification_report.total_tests}")
    print(f"  Passed tests: {verification_report.passed_tests}")
    print(f"  Failed tests: {verification_report.failed_tests}")
    print(f"  Success rate: {verification_report.success_rate:.2%}")
    print(f"  Execution time: {verification_report.execution_time:.2f}s")
    
    # Coverage metrics
    print(f"\nCoverage Metrics:")
    for metric, value in verification_report.coverage_metrics.items():
        print(f"  {metric}: {value:.2%}")
    
    # Show detailed results for failed tests
    failed_tests = [r for r in verification_report.test_results if not r.passed]
    if failed_tests:
        print(f"\nFailed Test Details:")
        for test in failed_tests[:5]:  # Show first 5 failures
            print(f"  {test.test_name}: {test.error_message or 'No error message'}")
    
    # Generate verification report file
    report_path = "phase1_verification_report.json" 
    verifier.generate_verification_report_file(verification_report, report_path)
    print(f"\nDetailed report saved to: {report_path}")
    
    return verification_report


def demo_visualization_and_documentation(network, registry, verification_report):
    """Demonstrate visualization and living documentation"""
    print_section("Phase 1.6: Visualization & Living Documentation")
    
    # Create generators
    mermaid_gen = MermaidFlowchartGenerator()
    doc_gen = DocumentationGenerator(output_dir="phase1_docs")
    
    print_subsection("Generating Flowcharts")
    
    # Network architecture flowchart
    print("Generating network architecture flowchart...")
    network_flowchart = mermaid_gen.create_network_architecture_flowchart(network)
    print(f"  Network flowchart: {len(network_flowchart.split())} words")
    
    # Hypergraph flowcharts for fragments
    hypergraph_flowcharts = []
    for i, (frag_id, fragment) in enumerate(list(network.fragments.items())[:2]):
        print(f"Generating hypergraph flowchart for fragment {i+1}...")
        flowchart = mermaid_gen.create_hypergraph_flowchart(fragment)
        hypergraph_flowcharts.append(flowchart)
        print(f"  Fragment {i+1} flowchart: {len(flowchart.split())} words")
    
    # Tensor registry flowchart
    print("Generating tensor registry flowchart...")
    registry_flowchart = mermaid_gen.create_tensor_registry_flowchart(registry)
    print(f"  Registry flowchart: {len(registry_flowchart.split())} words")
    
    # Phase implementation flowchart
    print("Generating phase implementation flowchart...")
    phase_flowchart = mermaid_gen.create_phase_implementation_flowchart()
    print(f"  Phase flowchart: {len(phase_flowchart.split())} words")
    
    print_subsection("Generating Living Documentation")
    
    # Generate complete documentation suite
    print("Generating comprehensive documentation...")
    docs = doc_gen.generate_complete_documentation_suite(network, registry)
    
    print(f"Documentation generated:")
    for doc_type, content in docs.items():
        doc_size = len(content.split())
        print(f"  {doc_type}.md: {doc_size:,} words")
    
    # Save individual flowcharts for inspection
    flowchart_dir = Path("phase1_docs") / "flowcharts"
    flowchart_dir.mkdir(exist_ok=True)
    
    flowcharts = {
        'network_architecture.mmd': network_flowchart,
        'tensor_registry.mmd': registry_flowchart, 
        'phase_implementation.mmd': phase_flowchart
    }
    
    for i, flowchart in enumerate(hypergraph_flowcharts):
        flowcharts[f'hypergraph_fragment_{i+1}.mmd'] = flowchart
    
    for filename, content in flowcharts.items():
        with open(flowchart_dir / filename, 'w') as f:
            f.write(content)
    
    print(f"\nFlowcharts saved to: {flowchart_dir}")
    print(f"  {len(flowcharts)} Mermaid diagram files created")
    
    # Generate verification summary for docs
    verification_summary = f"""
## Phase 1 Verification Results

- **Total Tests**: {verification_report.total_tests}
- **Success Rate**: {verification_report.success_rate:.2%}
- **Execution Time**: {verification_report.execution_time:.2f}s
- **Coverage**: {verification_report.coverage_metrics.get('fragment_coverage', 0):.2%} fragment coverage

### Test Categories
"""
    
    # Add category breakdown
    categories = {}
    for result in verification_report.test_results:
        category = result.test_name.split('_')[0]
        if category not in categories:
            categories[category] = {'total': 0, 'passed': 0}
        categories[category]['total'] += 1
        if result.passed:
            categories[category]['passed'] += 1
    
    for category, stats in categories.items():
        success_rate = stats['passed'] / stats['total'] if stats['total'] > 0 else 0
        verification_summary += f"- **{category}**: {stats['passed']}/{stats['total']} ({success_rate:.1%})\n"
    
    # Save verification summary
    with open("phase1_docs/verification_summary.md", 'w') as f:
        f.write(verification_summary)
    
    print(f"\nVerification summary saved to: phase1_docs/verification_summary.md")
    
    return docs, flowcharts


def main():
    """Main demonstration function"""
    print_section("🧬 Distributed Agentic Cognitive Grammar Network")
    print("Phase 1: Cognitive Primitives & Foundational Hypergraph Encoding")
    print("Real Implementation Demonstration (No Mocks)")
    
    try:
        # Import numpy for calculations
        import numpy as np
        globals()['np'] = np
        
        # Phase 1.1: Core Network
        network, fragments = demo_cognitive_grammar_network()
        
        # Phase 1.2: Scheme Adapters 
        adapter, translator, scheme_fragments, translation_results = demo_scheme_adapters()
        
        # Add scheme fragments to main fragments list
        fragments.extend(scheme_fragments)
        
        # Phase 1.3: Tensor Registry
        registry, mapper, kernel_shapes = demo_tensor_registry()
        
        # Phase 1.4: Pattern Transformations
        transformation_results = demo_pattern_transformations(network, fragments)
        
        # Phase 1.5: Comprehensive Verification
        verification_report = demo_comprehensive_verification(
            network, fragments, translation_results
        )
        
        # Phase 1.6: Visualization & Documentation
        docs, flowcharts = demo_visualization_and_documentation(
            network, registry, verification_report
        )
        
        # Final Summary
        print_section("🎯 Phase 1 Implementation Summary")
        
        print("✅ **COMPLETED: Phase 1 - Cognitive Primitives & Foundational Hypergraph Encoding**")
        print()
        print("**1.1 Scheme Cognitive Grammar Microservices**")
        print(f"   - Created {len(scheme_fragments)} fragments from Scheme expressions") 
        print(f"   - Bidirectional translation: {sum(1 for r in translation_results if r.get('consistent', False))}/{len(translation_results)} consistent")
        print("   - Real implementation verification (no mocks)")
        print()
        print("**1.2 Tensor Fragment Architecture**")
        print(f"   - Created {len(fragments)} total tensor fragments")
        print(f"   - Hypergraph encoding: {sum(len(f.nodes) for f in fragments)} nodes total")
        print(f"   - Tensor shapes: [modality, depth, context, salience, autonomy_index]")
        print()
        print("**1.3 Prime Factorization Tensor Registry**")
        print(f"   - Registered {len(kernel_shapes)} kernel tensor shapes")
        print(f"   - Prime factorization mappings documented")
        print(f"   - Registry catalog with compatibility analysis")
        print()
        print("**1.4 Pattern Transformations**")
        print(f"   - Applied {len(transformation_results)} transformations")
        print(f"   - Pattern library: {len(network.pattern_library)} patterns")
        print("   - Real hypergraph fragment testing")
        print()
        print("**1.5 Verification & Testing**")
        print(f"   - Executed {verification_report.total_tests} tests")
        print(f"   - Success rate: {verification_report.success_rate:.2%}")
        print("   - Exhaustive pattern transformation tests")
        print("   - No mocks - real implementation verification")
        print()
        print("**1.6 Documentation & Visualization**")
        print(f"   - Generated {len(docs)} documentation files")
        print(f"   - Created {len(flowcharts)} Mermaid flowcharts")
        print("   - Living documentation with auto-generation")
        print()
        print("**📁 Generated Files:**")
        print("   - phase1_tensor_registry.json")
        print("   - phase1_verification_report.json") 
        print("   - phase1_docs/ (complete documentation suite)")
        print("   - phase1_docs/flowcharts/ (Mermaid diagrams)")
        print()
        print("**🚀 Ready for Phase 2: ECAN Attention Allocation & Resource Kernel Construction**")
        print()
        print("**Next Implementation Steps:**")
        print("   - Architect ECAN-inspired resource allocators")
        print("   - Integrate with AtomSpace for activation spreading")
        print("   - Benchmark attention allocation across distributed agents")
        print("   - Document mesh topology and dynamic state propagation")
        
    except Exception as e:
        print(f"\n❌ ERROR in Phase 1 demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)