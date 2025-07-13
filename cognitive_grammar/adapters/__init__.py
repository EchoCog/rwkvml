"""
Scheme Cognitive Grammar Adapters

This module implements bidirectional adapters for agentic grammar ↔ AtomSpace
translation and Scheme-based cognitive grammar microservices.
"""

import torch
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
import re
from abc import ABC, abstractmethod

from ..core import TensorFragment, HypergraphNode, AtomSpaceAdapter, TensorSignature


@dataclass
class SchemeExpression:
    """Represents a Scheme S-expression for cognitive grammar"""
    expression: str
    atom_type: str
    arguments: List['SchemeExpression']
    metadata: Dict[str, Any]
    
    @classmethod
    def parse(cls, scheme_str: str) -> 'SchemeExpression':
        """Parse Scheme string into expression tree"""
        # Simplified Scheme parser for cognitive grammar
        scheme_str = scheme_str.strip()
        
        if not scheme_str.startswith('('):
            # Atomic expression
            return cls(
                expression=scheme_str,
                atom_type='atom',
                arguments=[],
                metadata={'value': scheme_str}
            )
        
        # Parse parenthesized expression
        tokens = cls._tokenize(scheme_str)
        return cls._parse_tokens(tokens)
    
    @staticmethod
    def _tokenize(scheme_str: str) -> List[str]:
        """Tokenize Scheme expression"""
        # Simple tokenizer for basic Scheme expressions
        tokens = []
        current_token = ""
        in_string = False
        paren_depth = 0
        
        for char in scheme_str:
            if char == '"' and not in_string:
                in_string = True
                current_token += char
            elif char == '"' and in_string:
                in_string = False
                current_token += char
                tokens.append(current_token)
                current_token = ""
            elif in_string:
                current_token += char
            elif char in '()':
                if current_token.strip():
                    tokens.append(current_token.strip())
                    current_token = ""
                tokens.append(char)
            elif char.isspace():
                if current_token.strip():
                    tokens.append(current_token.strip())
                    current_token = ""
            else:
                current_token += char
        
        if current_token.strip():
            tokens.append(current_token.strip())
        
        return tokens
    
    @classmethod
    def _parse_tokens(cls, tokens: List[str]) -> 'SchemeExpression':
        """Parse tokenized Scheme expression"""
        if not tokens:
            return cls("", "empty", [], {})
        
        if tokens[0] != '(':
            return cls(tokens[0], "atom", [], {"value": tokens[0]})
        
        # Find matching closing paren
        paren_count = 0
        end_idx = 0
        for i, token in enumerate(tokens):
            if token == '(':
                paren_count += 1
            elif token == ')':
                paren_count -= 1
                if paren_count == 0:
                    end_idx = i
                    break
        
        inner_tokens = tokens[1:end_idx]
        if not inner_tokens:
            return cls("()", "empty_list", [], {})
        
        # First token is the operator/function
        operator = inner_tokens[0]
        
        # Parse arguments
        args = []
        i = 1
        while i < len(inner_tokens):
            if inner_tokens[i] == '(':
                # Find matching closing paren for this argument
                paren_count = 1
                j = i + 1
                while j < len(inner_tokens) and paren_count > 0:
                    if inner_tokens[j] == '(':
                        paren_count += 1
                    elif inner_tokens[j] == ')':
                        paren_count -= 1
                    j += 1
                
                # Parse this sub-expression
                arg_tokens = inner_tokens[i:j]
                args.append(cls._parse_tokens(arg_tokens))
                i = j
            else:
                # Simple atom argument
                args.append(cls(inner_tokens[i], "atom", [], {"value": inner_tokens[i]}))
                i += 1
        
        return cls(
            expression=' '.join(tokens[:end_idx+1]),
            atom_type=operator,
            arguments=args,
            metadata={"operator": operator}
        )
    
    def to_atomspace_format(self) -> Dict[str, Any]:
        """Convert to AtomSpace-compatible format"""
        return {
            "type": self.atom_type,
            "name": self.expression,
            "arguments": [arg.to_atomspace_format() for arg in self.arguments],
            "metadata": self.metadata
        }


class BiDirectionalTranslator:
    """
    Bidirectional translator between cognitive grammar and tensor representations
    
    Handles conversion between Scheme expressions, hypergraph structures,
    and tensor fragments with semantic preservation.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.grammar_vocabulary = {}  # Map grammar elements to indices
        self.tensor_patterns = {}     # Map patterns to transformations
        
    def scheme_to_tensor_fragment(self, scheme_expr: SchemeExpression) -> TensorFragment:
        """Convert Scheme expression to tensor fragment"""
        fragment = TensorFragment(device=self.device)
        
        # Create root node from main expression
        root_node = self._scheme_expr_to_node(scheme_expr)
        fragment.add_node(root_node)
        
        # Add argument nodes and create links
        for i, arg in enumerate(scheme_expr.arguments):
            arg_node = self._scheme_expr_to_node(arg)
            fragment.add_node(arg_node)
            
            # Create link from root to argument
            root_node.add_link(arg_node, f"argument_{i}", weight=1.0 / (i + 1))
        
        fragment.create_collective_tensor()
        return fragment
    
    def _scheme_expr_to_node(self, expr: SchemeExpression) -> HypergraphNode:
        """Convert single Scheme expression to hypergraph node"""
        # Determine tensor signature based on expression complexity
        depth = self._calculate_expression_depth(expr)
        context = len(expr.arguments) + 1
        
        signature = TensorSignature(
            modality=hash(expr.atom_type) % 8 + 1,
            depth=depth,
            context=context,
            salience=4,
            autonomy_index=2
        )
        
        node = HypergraphNode(
            node_type=f"scheme_{expr.atom_type}",
            tensor_signature=signature,
            semantic_content={
                "scheme_expression": expr.expression,
                "atom_type": expr.atom_type,
                "metadata": expr.metadata
            }
        )
        
        return node
    
    def _calculate_expression_depth(self, expr: SchemeExpression) -> int:
        """Calculate depth of Scheme expression"""
        if not expr.arguments:
            return 1
        return 1 + max(self._calculate_expression_depth(arg) for arg in expr.arguments)
    
    def tensor_fragment_to_scheme(self, fragment: TensorFragment) -> SchemeExpression:
        """Convert tensor fragment back to Scheme expression"""
        if not fragment.nodes:
            return SchemeExpression("()", "empty", [], {})
        
        # Find root node (typically first or most connected)
        root_node = fragment.nodes[0]
        
        # Convert to Scheme expression
        return self._node_to_scheme_expr(root_node, fragment)
    
    def _node_to_scheme_expr(self, node: HypergraphNode, fragment: TensorFragment) -> SchemeExpression:
        """Convert hypergraph node back to Scheme expression"""
        semantic_content = node.semantic_content
        
        if "scheme_expression" in semantic_content:
            # Try to reconstruct from stored expression
            return SchemeExpression.parse(semantic_content["scheme_expression"])
        
        # Reconstruct from node structure
        atom_type = semantic_content.get("atom_type", node.node_type)
        
        # Find linked nodes as arguments
        arguments = []
        for link in node.links:
            target_node = link['target']
            arg_expr = self._node_to_scheme_expr(target_node, fragment)
            arguments.append(arg_expr)
        
        if arguments:
            expr_str = f"({atom_type} {' '.join(arg.expression for arg in arguments)})"
        else:
            expr_str = atom_type
        
        return SchemeExpression(
            expression=expr_str,
            atom_type=atom_type,
            arguments=arguments,
            metadata=semantic_content.get("metadata", {})
        )
    
    def verify_bidirectional_consistency(self, scheme_expr: SchemeExpression) -> bool:
        """Verify that bidirectional translation preserves semantics"""
        # Convert to tensor fragment and back
        fragment = self.scheme_to_tensor_fragment(scheme_expr)
        reconstructed = self.tensor_fragment_to_scheme(fragment)
        
        # Check semantic preservation
        return self._expressions_semantically_equivalent(scheme_expr, reconstructed)
    
    def _expressions_semantically_equivalent(self, 
                                           expr1: SchemeExpression, 
                                           expr2: SchemeExpression) -> bool:
        """Check if two expressions are semantically equivalent"""
        # Compare structure and content
        if expr1.atom_type != expr2.atom_type:
            return False
        
        if len(expr1.arguments) != len(expr2.arguments):
            return False
        
        # Recursively check arguments
        for arg1, arg2 in zip(expr1.arguments, expr2.arguments):
            if not self._expressions_semantically_equivalent(arg1, arg2):
                return False
        
        return True


class SchemeAdapter:
    """
    Scheme adapter for agentic grammar microservices
    
    Provides high-level interface for Scheme-based cognitive grammar
    processing with bidirectional AtomSpace integration.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.translator = BiDirectionalTranslator(device)
        self.active_fragments = {}
        self.grammar_rules = {}
        
    def parse_agentic_grammar(self, grammar_text: str) -> List[TensorFragment]:
        """Parse agentic grammar text into tensor fragments"""
        fragments = []
        
        # Split into individual expressions
        expressions = self._extract_scheme_expressions(grammar_text)
        
        for expr_text in expressions:
            try:
                scheme_expr = SchemeExpression.parse(expr_text)
                fragment = self.translator.scheme_to_tensor_fragment(scheme_expr)
                fragments.append(fragment)
            except Exception as e:
                print(f"Warning: Failed to parse expression '{expr_text}': {e}")
        
        return fragments
    
    def _extract_scheme_expressions(self, text: str) -> List[str]:
        """Extract individual Scheme expressions from text"""
        expressions = []
        current_expr = ""
        paren_depth = 0
        in_expression = False
        
        for char in text:
            if char == '(' and not in_expression:
                in_expression = True
                current_expr = char
                paren_depth = 1
            elif char == '(' and in_expression:
                current_expr += char
                paren_depth += 1
            elif char == ')' and in_expression:
                current_expr += char
                paren_depth -= 1
                if paren_depth == 0:
                    expressions.append(current_expr.strip())
                    current_expr = ""
                    in_expression = False
            elif in_expression:
                current_expr += char
        
        return expressions
    
    def create_microservice_interface(self, service_name: str) -> Dict[str, Any]:
        """Create microservice interface for agentic grammar processing"""
        return {
            "service_name": service_name,
            "endpoints": {
                "parse": f"/cognitive_grammar/{service_name}/parse",
                "transform": f"/cognitive_grammar/{service_name}/transform",
                "verify": f"/cognitive_grammar/{service_name}/verify"
            },
            "capabilities": [
                "scheme_parsing",
                "tensor_conversion",
                "hypergraph_encoding",
                "bidirectional_translation"
            ],
            "tensor_device": self.device,
            "active_fragments": len(self.active_fragments)
        }
    
    def add_grammar_rule(self, rule_name: str, pattern: str, transformation: str):
        """Add a cognitive grammar rule"""
        self.grammar_rules[rule_name] = {
            "pattern": pattern,
            "transformation": transformation,
            "compiled": False
        }
    
    def apply_grammar_rule(self, fragment: TensorFragment, rule_name: str) -> TensorFragment:
        """Apply a grammar rule to transform a tensor fragment"""
        if rule_name not in self.grammar_rules:
            raise ValueError(f"Grammar rule '{rule_name}' not found")
        
        rule = self.grammar_rules[rule_name]
        
        # Convert fragment to Scheme for pattern matching
        scheme_expr = self.translator.tensor_fragment_to_scheme(fragment)
        
        # Apply transformation (simplified pattern matching)
        if rule["pattern"] in scheme_expr.expression:
            # Create simple transformation
            transformed_expr_text = scheme_expr.expression.replace(
                rule["pattern"], rule["transformation"]
            )
            transformed_expr = SchemeExpression.parse(transformed_expr_text)
            return self.translator.scheme_to_tensor_fragment(transformed_expr)
        
        return fragment  # No transformation applied
    
    def get_adapter_status(self) -> Dict[str, Any]:
        """Get current adapter status and metrics"""
        return {
            "device": self.device,
            "active_fragments": len(self.active_fragments),
            "grammar_rules": len(self.grammar_rules),
            "translator_vocabulay_size": len(self.translator.grammar_vocabulary),
            "tensor_patterns": len(self.translator.tensor_patterns)
        }


class RealAtomSpaceAdapter(AtomSpaceAdapter):
    """
    Real AtomSpace adapter implementation for hypergraph integration
    
    Note: This is a basic implementation. In production, this would
    interface with the actual OpenCog AtomSpace API.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.atomspace_data = {}  # Simulated AtomSpace storage
        self.atom_counter = 0
        
    def to_atomspace(self, tensor_fragment: TensorFragment) -> Dict[str, Any]:
        """Convert tensor fragment to AtomSpace representation"""
        atoms = []
        links = []
        
        # Convert each node to an Atom
        node_map = {}
        for node in tensor_fragment.nodes:
            atom_id = f"atom_{self.atom_counter}"
            self.atom_counter += 1
            
            atom = {
                "id": atom_id,
                "type": "ConceptNode",  # OpenCog atom type
                "name": f"cognitive_entity_{node.node_id}",
                "tensor_signature": node.tensor_signature.__dict__,
                "semantic_content": node.semantic_content,
                "strength": 1.0,  # Truth value strength
                "confidence": 0.9   # Truth value confidence
            }
            
            atoms.append(atom)
            node_map[node.node_id] = atom_id
            
        # Convert hypergraph links to AtomSpace Links
        for node in tensor_fragment.nodes:
            source_id = node_map[node.node_id]
            
            for link in node.links:
                target_id = node_map[link['target'].node_id]
                
                link_atom = {
                    "id": f"link_{self.atom_counter}",
                    "type": "InheritanceLink",  # OpenCog link type
                    "source": source_id,
                    "target": target_id,
                    "weight": link['weight'],
                    "link_type": link['type']
                }
                
                links.append(link_atom)
                self.atom_counter += 1
        
        atomspace_repr = {
            "fragment_id": tensor_fragment.fragment_id,
            "atoms": atoms,
            "links": links,
            "timestamp": torch.tensor([0.0])  # Placeholder timestamp
        }
        
        # Store in simulated AtomSpace
        self.atomspace_data[tensor_fragment.fragment_id] = atomspace_repr
        
        return atomspace_repr
    
    def from_atomspace(self, atomspace_data: Dict[str, Any]) -> TensorFragment:
        """Convert AtomSpace representation to tensor fragment"""
        fragment = TensorFragment(device=self.device)
        atom_to_node = {}
        
        # Recreate nodes from atoms
        for atom in atomspace_data.get("atoms", []):
            # Reconstruct tensor signature
            sig_data = atom["tensor_signature"]
            signature = TensorSignature(**sig_data)
            
            # Create hypergraph node
            node = HypergraphNode(
                node_type=atom["type"],
                tensor_signature=signature,
                semantic_content=atom["semantic_content"]
            )
            
            fragment.add_node(node)
            atom_to_node[atom["id"]] = node
        
        # Recreate links
        for link in atomspace_data.get("links", []):
            source_node = atom_to_node[link["source"]]
            target_node = atom_to_node[link["target"]]
            
            source_node.add_link(
                target_node,
                link["link_type"],
                link["weight"]
            )
        
        return fragment
    
    def sync_bidirectional(self, tensor_fragment: TensorFragment) -> TensorFragment:
        """Synchronize bidirectional updates between representations"""
        # Convert to AtomSpace
        atomspace_repr = self.to_atomspace(tensor_fragment)
        
        # Simulate AtomSpace processing (e.g., inference, attention allocation)
        # In real implementation, this would involve OpenCog inference engines
        
        # Apply mock updates (placeholder for real AtomSpace operations)
        for atom in atomspace_repr["atoms"]:
            atom["strength"] = min(1.0, atom["strength"] * 1.01)  # Increase strength
        
        # Convert back to tensor fragment
        updated_fragment = self.from_atomspace(atomspace_repr)
        
        return updated_fragment
    
    def query_atomspace(self, query_pattern: str) -> List[Dict[str, Any]]:
        """Query the AtomSpace with a pattern"""
        # Simplified pattern matching
        results = []
        
        for fragment_id, data in self.atomspace_data.items():
            for atom in data["atoms"]:
                if query_pattern.lower() in atom["name"].lower():
                    results.append(atom)
        
        return results
    
    def get_atomspace_statistics(self) -> Dict[str, Any]:
        """Get AtomSpace statistics"""
        total_atoms = sum(len(data["atoms"]) for data in self.atomspace_data.values())
        total_links = sum(len(data["links"]) for data in self.atomspace_data.values())
        
        return {
            "total_fragments": len(self.atomspace_data),
            "total_atoms": total_atoms,
            "total_links": total_links,
            "device": self.device
        }