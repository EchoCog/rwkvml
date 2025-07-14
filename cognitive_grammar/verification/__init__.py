"""
Verification and Testing Framework

This module provides pattern transformation tests, hypergraph verification,
and exhaustive testing protocols for the cognitive grammar network.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import time
import json
from pathlib import Path
import unittest
from abc import ABC, abstractmethod

from ..core import TensorFragment, HypergraphNode, CognitiveGrammarNetwork, TensorSignature
from ..adapters import SchemeExpression, BiDirectionalTranslator, SchemeAdapter
from ..tensor_registry import TensorShapeRegistry, KernelTensorShape


@dataclass
class TestResult:
    """Test result container"""
    test_name: str
    passed: bool
    execution_time: float
    details: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class VerificationReport:
    """Comprehensive verification report"""
    total_tests: int
    passed_tests: int
    failed_tests: int
    execution_time: float
    test_results: List[TestResult]
    coverage_metrics: Dict[str, float]
    
    @property
    def success_rate(self) -> float:
        return self.passed_tests / self.total_tests if self.total_tests > 0 else 0.0


class PatternTransformationTest:
    """
    Pattern transformation test framework
    
    Provides exhaustive testing of pattern transformations with
    real implementation verification (no mocks as requested).
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.test_patterns = {}
        self.test_results = []
        
    def register_test_pattern(self, 
                            pattern_name: str, 
                            input_fragment: TensorFragment,
                            transformation_matrix: torch.Tensor,
                            expected_properties: Dict[str, Any]):
        """Register a test pattern for verification"""
        self.test_patterns[pattern_name] = {
            'input_fragment': input_fragment,
            'transformation_matrix': transformation_matrix.to(device=self.device),
            'expected_properties': expected_properties
        }
    
    def test_pattern_preservation(self, pattern_name: str) -> TestResult:
        """Test that pattern transformation preserves expected properties"""
        start_time = time.time()
        
        if pattern_name not in self.test_patterns:
            return TestResult(
                test_name=f"pattern_preservation_{pattern_name}",
                passed=False,
                execution_time=0.0,
                details={},
                error_message=f"Pattern '{pattern_name}' not found"
            )
        
        try:
            pattern = self.test_patterns[pattern_name]
            input_fragment = pattern['input_fragment']
            transformation_matrix = pattern['transformation_matrix']
            expected_props = pattern['expected_properties']
            
            # Apply transformation
            transformed_fragment = input_fragment.pattern_transformation(transformation_matrix)
            
            # Verify properties
            verification_results = {}
            
            # Test 1: Node count preservation
            verification_results['node_count_preserved'] = (
                len(transformed_fragment.nodes) == len(input_fragment.nodes)
            )
            
            # Test 2: Tensor shape consistency
            shape_consistent = True
            for orig_node, trans_node in zip(input_fragment.nodes, transformed_fragment.nodes):
                if orig_node.tensor_state is not None and trans_node.tensor_state is not None:
                    if orig_node.tensor_state.shape != trans_node.tensor_state.shape:
                        shape_consistent = False
                        break
            verification_results['tensor_shape_consistent'] = shape_consistent
            
            # Test 3: Semantic content preservation
            semantic_preserved = True
            for orig_node, trans_node in zip(input_fragment.nodes, transformed_fragment.nodes):
                if orig_node.semantic_content != trans_node.semantic_content:
                    semantic_preserved = False
                    break
            verification_results['semantic_content_preserved'] = semantic_preserved
            
            # Test 4: Expected properties
            for prop_name, expected_value in expected_props.items():
                if prop_name == 'max_tensor_norm':
                    max_norm = max(
                        torch.norm(node.tensor_state).item() 
                        for node in transformed_fragment.nodes 
                        if node.tensor_state is not None
                    )
                    verification_results[prop_name] = abs(max_norm - expected_value) < 0.1
                elif prop_name == 'energy_conservation':
                    orig_energy = sum(
                        torch.sum(node.tensor_state ** 2).item()
                        for node in input_fragment.nodes
                        if node.tensor_state is not None
                    )
                    trans_energy = sum(
                        torch.sum(node.tensor_state ** 2).item()
                        for node in transformed_fragment.nodes
                        if node.tensor_state is not None
                    )
                    verification_results[prop_name] = abs(orig_energy - trans_energy) < 0.1
            
            # Overall test result
            all_passed = all(verification_results.values())
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name=f"pattern_preservation_{pattern_name}",
                passed=all_passed,
                execution_time=execution_time,
                details=verification_results
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name=f"pattern_preservation_{pattern_name}",
                passed=False,
                execution_time=execution_time,
                details={},
                error_message=str(e)
            )
    
    def test_transformation_invertibility(self, pattern_name: str) -> TestResult:
        """Test that transformation is invertible when expected"""
        start_time = time.time()
        
        try:
            pattern = self.test_patterns[pattern_name]
            input_fragment = pattern['input_fragment']
            transformation_matrix = pattern['transformation_matrix']
            
            # Apply forward transformation
            transformed_fragment = input_fragment.pattern_transformation(transformation_matrix)
            
            # Try to compute inverse transformation
            try:
                inverse_matrix = torch.pinverse(transformation_matrix)
                reconstructed_fragment = transformed_fragment.pattern_transformation(inverse_matrix)
                
                # Compare original and reconstructed
                reconstruction_error = 0.0
                node_pairs = zip(input_fragment.nodes, reconstructed_fragment.nodes)
                
                for orig_node, recon_node in node_pairs:
                    if orig_node.tensor_state is not None and recon_node.tensor_state is not None:
                        error = torch.norm(orig_node.tensor_state - recon_node.tensor_state).item()
                        reconstruction_error += error
                
                # Test passes if reconstruction error is small
                passed = reconstruction_error < 0.1
                
                execution_time = time.time() - start_time
                
                return TestResult(
                    test_name=f"transformation_invertibility_{pattern_name}",
                    passed=passed,
                    execution_time=execution_time,
                    details={
                        'reconstruction_error': reconstruction_error,
                        'inverse_computed': True
                    }
                )
                
            except Exception as inv_e:
                execution_time = time.time() - start_time
                
                return TestResult(
                    test_name=f"transformation_invertibility_{pattern_name}",
                    passed=False,
                    execution_time=execution_time,
                    details={'inverse_computed': False},
                    error_message=f"Could not compute inverse: {inv_e}"
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name=f"transformation_invertibility_{pattern_name}",
                passed=False,
                execution_time=execution_time,
                details={},
                error_message=str(e)
            )
    
    def run_all_pattern_tests(self) -> List[TestResult]:
        """Run all registered pattern tests"""
        results = []
        
        for pattern_name in self.test_patterns.keys():
            # Test pattern preservation
            preservation_result = self.test_pattern_preservation(pattern_name)
            results.append(preservation_result)
            
            # Test invertibility
            invertibility_result = self.test_transformation_invertibility(pattern_name)
            results.append(invertibility_result)
        
        self.test_results.extend(results)
        return results


class HypergraphVerifier:
    """
    Hypergraph verification and validation framework
    
    Verifies hypergraph structure integrity, semantic consistency,
    and tensor-hypergraph correspondence.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.verification_rules = {}
        
    def add_verification_rule(self, rule_name: str, rule_function: Callable):
        """Add a custom verification rule"""
        self.verification_rules[rule_name] = rule_function
    
    def verify_hypergraph_integrity(self, fragment: TensorFragment) -> TestResult:
        """Verify hypergraph structural integrity"""
        start_time = time.time()
        
        try:
            verification_details = {}
            
            # Test 1: All nodes have valid IDs
            node_ids = [node.node_id for node in fragment.nodes]
            unique_ids = set(node_ids)
            verification_details['unique_node_ids'] = len(node_ids) == len(unique_ids)
            
            # Test 2: All links reference valid nodes
            valid_links = True
            for node in fragment.nodes:
                for link in node.links:
                    target_id = link['target'].node_id
                    if target_id not in node_ids:
                        valid_links = False
                        break
                if not valid_links:
                    break
            verification_details['valid_link_references'] = valid_links
            
            # Test 3: Adjacency matrix consistency
            if fragment.adjacency_matrix is not None:
                expected_size = len(fragment.nodes)
                matrix_size = fragment.adjacency_matrix.shape
                verification_details['adjacency_matrix_size_correct'] = (
                    matrix_size == (expected_size, expected_size)
                )
            else:
                verification_details['adjacency_matrix_size_correct'] = True
            
            # Test 4: Tensor state validity
            tensor_states_valid = True
            for node in fragment.nodes:
                if node.tensor_state is not None:
                    if torch.isnan(node.tensor_state).any() or torch.isinf(node.tensor_state).any():
                        tensor_states_valid = False
                        break
            verification_details['tensor_states_valid'] = tensor_states_valid
            
            # Test 5: Semantic content consistency
            semantic_consistent = True
            for node in fragment.nodes:
                if 'node_type' in node.semantic_content:
                    if node.semantic_content['node_type'] != node.node_type:
                        semantic_consistent = False
                        break
            verification_details['semantic_content_consistent'] = semantic_consistent
            
            all_passed = all(verification_details.values())
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name="hypergraph_integrity",
                passed=all_passed,
                execution_time=execution_time,
                details=verification_details
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="hypergraph_integrity",
                passed=False,
                execution_time=execution_time,
                details={},
                error_message=str(e)
            )
    
    def verify_tensor_hypergraph_correspondence(self, fragment: TensorFragment) -> TestResult:
        """Verify correspondence between tensor and hypergraph representations"""
        start_time = time.time()
        
        try:
            verification_details = {}
            
            # Test 1: Node count matches tensor dimensions
            if fragment.collective_tensor is not None:
                expected_nodes = fragment.collective_tensor.shape[0]
                actual_nodes = len(fragment.nodes)
                verification_details['node_tensor_count_match'] = (expected_nodes == actual_nodes)
            else:
                verification_details['node_tensor_count_match'] = True
            
            # Test 2: Tensor signatures match actual tensor shapes
            signature_shape_match = True
            for node in fragment.nodes:
                if node.tensor_state is not None:
                    expected_shape = node.tensor_signature.shape
                    actual_shape = node.tensor_state.shape
                    if expected_shape != actual_shape:
                        signature_shape_match = False
                        break
            verification_details['signature_shape_match'] = signature_shape_match
            
            # Test 3: Link weights reflected in adjacency matrix
            if fragment.adjacency_matrix is not None:
                matrix_link_consistency = True
                node_to_idx = {node.node_id: i for i, node in enumerate(fragment.nodes)}
                
                for i, node in enumerate(fragment.nodes):
                    for link in node.links:
                        target_id = link['target'].node_id
                        if target_id in node_to_idx:
                            j = node_to_idx[target_id]
                            expected_weight = link['weight']
                            actual_weight = fragment.adjacency_matrix[i, j].item()
                            if abs(expected_weight - actual_weight) > 1e-6:
                                matrix_link_consistency = False
                                break
                    if not matrix_link_consistency:
                        break
                
                verification_details['matrix_link_consistency'] = matrix_link_consistency
            else:
                verification_details['matrix_link_consistency'] = True
            
            all_passed = all(verification_details.values())
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name="tensor_hypergraph_correspondence",
                passed=all_passed,
                execution_time=execution_time,
                details=verification_details
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="tensor_hypergraph_correspondence",
                passed=False,
                execution_time=execution_time,
                details={},
                error_message=str(e)
            )
    
    def run_custom_verification_rules(self, fragment: TensorFragment) -> List[TestResult]:
        """Run all custom verification rules"""
        results = []
        
        for rule_name, rule_function in self.verification_rules.items():
            start_time = time.time()
            
            try:
                rule_result = rule_function(fragment)
                execution_time = time.time() - start_time
                
                if isinstance(rule_result, bool):
                    results.append(TestResult(
                        test_name=f"custom_rule_{rule_name}",
                        passed=rule_result,
                        execution_time=execution_time,
                        details={}
                    ))
                elif isinstance(rule_result, dict):
                    passed = rule_result.get('passed', False)
                    details = rule_result.get('details', {})
                    results.append(TestResult(
                        test_name=f"custom_rule_{rule_name}",
                        passed=passed,
                        execution_time=execution_time,
                        details=details
                    ))
                
            except Exception as e:
                execution_time = time.time() - start_time
                results.append(TestResult(
                    test_name=f"custom_rule_{rule_name}",
                    passed=False,
                    execution_time=execution_time,
                    details={},
                    error_message=str(e)
                ))
        
        return results


class BiDirectionalTranslationTest:
    """
    Test framework for bidirectional translation between representations
    
    Verifies Scheme ↔ TensorFragment ↔ AtomSpace translation consistency.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.translator = BiDirectionalTranslator(device)
        self.test_expressions = []
        
    def add_test_expression(self, scheme_text: str, description: str = ""):
        """Add a Scheme expression for testing"""
        self.test_expressions.append({
            'scheme_text': scheme_text,
            'description': description
        })
    
    def test_scheme_to_tensor_roundtrip(self, scheme_text: str) -> TestResult:
        """Test Scheme → Tensor → Scheme roundtrip"""
        start_time = time.time()
        
        try:
            # Parse original expression
            original_expr = SchemeExpression.parse(scheme_text)
            
            # Convert to tensor fragment
            tensor_fragment = self.translator.scheme_to_tensor_fragment(original_expr)
            
            # Convert back to Scheme
            reconstructed_expr = self.translator.tensor_fragment_to_scheme(tensor_fragment)
            
            # Verify consistency
            consistency_check = self.translator.verify_bidirectional_consistency(original_expr)
            
            # Additional checks
            verification_details = {
                'bidirectional_consistency': consistency_check,
                'original_atom_type': original_expr.atom_type,
                'reconstructed_atom_type': reconstructed_expr.atom_type,
                'original_arg_count': len(original_expr.arguments),
                'reconstructed_arg_count': len(reconstructed_expr.arguments),
                'tensor_nodes_created': len(tensor_fragment.nodes)
            }
            
            # Check structural similarity
            structural_match = (
                original_expr.atom_type == reconstructed_expr.atom_type and
                len(original_expr.arguments) == len(reconstructed_expr.arguments)
            )
            verification_details['structural_match'] = structural_match
            
            all_passed = consistency_check and structural_match
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name=f"scheme_tensor_roundtrip",
                passed=all_passed,
                execution_time=execution_time,
                details=verification_details
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name=f"scheme_tensor_roundtrip",
                passed=False,
                execution_time=execution_time,
                details={},
                error_message=str(e)
            )
    
    def test_all_expressions(self) -> List[TestResult]:
        """Test all registered expressions"""
        results = []
        
        for expr_data in self.test_expressions:
            result = self.test_scheme_to_tensor_roundtrip(expr_data['scheme_text'])
            result.test_name += f"_{expr_data['description']}"
            results.append(result)
        
        return results


class ComprehensiveVerificationSuite:
    """
    Comprehensive verification suite for the entire cognitive grammar network
    
    Orchestrates all verification components and generates detailed reports.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.pattern_tester = PatternTransformationTest(device)
        self.hypergraph_verifier = HypergraphVerifier(device)
        self.translation_tester = BiDirectionalTranslationTest(device)
        
    def setup_default_tests(self):
        """Set up default test suite"""
        # Add default pattern tests
        self._create_default_pattern_tests()
        
        # Add default translation tests
        self._create_default_translation_tests()
        
        # Add default verification rules
        self._create_default_verification_rules()
    
    def _create_default_pattern_tests(self):
        """Create default pattern transformation tests"""
        # Create simple test fragment
        from ..core import ModalityType
        
        # Create a simple cognitive network for testing
        network = CognitiveGrammarNetwork(device=self.device)
        test_fragment = network.create_agentic_fragment(
            agent_type="test_agent",
            modality=ModalityType.LINGUISTIC,
            depth=2,
            context=4
        )
        
        # Create identity transformation for testing
        if test_fragment.nodes:
            node = test_fragment.nodes[0]
            if node.tensor_state is not None:
                flat_size = node.tensor_state.numel()
                identity_matrix = torch.eye(flat_size, device=self.device)
                
                self.pattern_tester.register_test_pattern(
                    "identity_transform",
                    test_fragment,
                    identity_matrix,
                    {
                        'max_tensor_norm': torch.norm(node.tensor_state).item(),
                        'energy_conservation': True
                    }
                )
    
    def _create_default_translation_tests(self):
        """Create default translation tests"""
        default_expressions = [
            "(action move (agent robot) (target location))",
            "(cognitive-state (agent self) (state thinking))",
            "(relation (subject A) (predicate loves) (object B))",
            "(compound (operator and) (arg1 true) (arg2 false))"
        ]
        
        for i, expr in enumerate(default_expressions):
            self.translation_tester.add_test_expression(expr, f"default_test_{i}")
    
    def _create_default_verification_rules(self):
        """Create default verification rules"""
        
        def check_node_connectivity(fragment: TensorFragment) -> Dict[str, Any]:
            """Check that nodes have reasonable connectivity"""
            if not fragment.nodes:
                return {'passed': True, 'details': {'reason': 'no_nodes'}}
            
            total_links = sum(len(node.links) for node in fragment.nodes)
            avg_connectivity = total_links / len(fragment.nodes)
            
            # Reasonable connectivity: not completely isolated, not over-connected
            reasonable = 0 <= avg_connectivity <= len(fragment.nodes)
            
            return {
                'passed': reasonable,
                'details': {
                    'average_connectivity': avg_connectivity,
                    'total_links': total_links,
                    'total_nodes': len(fragment.nodes)
                }
            }
        
        def check_tensor_numerical_stability(fragment: TensorFragment) -> Dict[str, Any]:
            """Check that tensors are numerically stable"""
            for node in fragment.nodes:
                if node.tensor_state is not None:
                    if torch.isnan(node.tensor_state).any():
                        return {'passed': False, 'details': {'reason': 'nan_values'}}
                    if torch.isinf(node.tensor_state).any():
                        return {'passed': False, 'details': {'reason': 'inf_values'}}
                    if torch.norm(node.tensor_state).item() > 1e6:
                        return {'passed': False, 'details': {'reason': 'large_magnitude'}}
            
            return {'passed': True, 'details': {'reason': 'numerically_stable'}}
        
        self.hypergraph_verifier.add_verification_rule("node_connectivity", check_node_connectivity)
        self.hypergraph_verifier.add_verification_rule("numerical_stability", check_tensor_numerical_stability)
    
    def run_full_verification_suite(self, 
                                  network: CognitiveGrammarNetwork,
                                  additional_fragments: List[TensorFragment] = None) -> VerificationReport:
        """Run complete verification suite"""
        start_time = time.time()
        all_results = []
        
        # Test pattern transformations
        pattern_results = self.pattern_tester.run_all_pattern_tests()
        all_results.extend(pattern_results)
        
        # Test bidirectional translations
        translation_results = self.translation_tester.test_all_expressions()
        all_results.extend(translation_results)
        
        # Test all fragments in the network
        test_fragments = list(network.fragments.values())
        if additional_fragments:
            test_fragments.extend(additional_fragments)
        
        for fragment in test_fragments:
            # Hypergraph integrity tests
            integrity_result = self.hypergraph_verifier.verify_hypergraph_integrity(fragment)
            all_results.append(integrity_result)
            
            # Tensor-hypergraph correspondence tests
            correspondence_result = self.hypergraph_verifier.verify_tensor_hypergraph_correspondence(fragment)
            all_results.append(correspondence_result)
            
            # Custom verification rules
            custom_results = self.hypergraph_verifier.run_custom_verification_rules(fragment)
            all_results.extend(custom_results)
        
        # Calculate metrics
        total_tests = len(all_results)
        passed_tests = sum(1 for result in all_results if result.passed)
        failed_tests = total_tests - passed_tests
        total_execution_time = time.time() - start_time
        
        # Calculate coverage metrics
        coverage_metrics = self._calculate_coverage_metrics(all_results, network)
        
        return VerificationReport(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            execution_time=total_execution_time,
            test_results=all_results,
            coverage_metrics=coverage_metrics
        )
    
    def _calculate_coverage_metrics(self, 
                                  results: List[TestResult],
                                  network: CognitiveGrammarNetwork) -> Dict[str, float]:
        """Calculate test coverage metrics"""
        # Basic coverage calculations
        test_categories = {}
        for result in results:
            category = result.test_name.split('_')[0]  # First part of test name
            if category not in test_categories:
                test_categories[category] = {'total': 0, 'passed': 0}
            test_categories[category]['total'] += 1
            if result.passed:
                test_categories[category]['passed'] += 1
        
        coverage_metrics = {}
        for category, stats in test_categories.items():
            coverage_metrics[f"{category}_coverage"] = stats['passed'] / stats['total']
        
        # Network coverage
        total_fragments = len(network.fragments)
        fragments_tested = min(total_fragments, len([r for r in results if 'hypergraph' in r.test_name]))
        
        if total_fragments > 0:
            coverage_metrics['fragment_coverage'] = fragments_tested / total_fragments
        else:
            coverage_metrics['fragment_coverage'] = 1.0
        
        return coverage_metrics
    
    def generate_verification_report_file(self, 
                                        report: VerificationReport,
                                        output_path: str = "verification_report.json"):
        """Generate detailed verification report file"""
        report_data = {
            'summary': {
                'total_tests': report.total_tests,
                'passed_tests': report.passed_tests,
                'failed_tests': report.failed_tests,
                'success_rate': report.success_rate,
                'execution_time': report.execution_time
            },
            'coverage_metrics': report.coverage_metrics,
            'test_results': []
        }
        
        for result in report.test_results:
            result_data = {
                'test_name': result.test_name,
                'passed': result.passed,
                'execution_time': result.execution_time,
                'details': result.details
            }
            if result.error_message:
                result_data['error_message'] = result.error_message
                
            report_data['test_results'].append(result_data)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            print(f"Verification report saved to {output_path}")
        except Exception as e:
            print(f"Failed to save verification report: {e}")
        
        return report_data