"""
Core components of the Distributed Agentic Cognitive Grammar Network

This module contains the fundamental classes and interfaces for cognitive
grammar processing, tensor fragment management, and hypergraph encoding.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import uuid
from abc import ABC, abstractmethod


class ModalityType(Enum):
    """Types of cognitive modalities in the network"""
    LINGUISTIC = "linguistic"
    VISUAL = "visual"
    AUDITORY = "auditory"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    CONCEPTUAL = "conceptual"
    AGENTIC = "agentic"


@dataclass
class TensorSignature:
    """
    Tensor signature with prime factorization mapping
    
    Shape: [modality, depth, context, salience, autonomy_index]
    Each dimension represents semantic complexity and functional depth
    """
    modality: int
    depth: int
    context: int
    salience: int
    autonomy_index: int
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return (self.modality, self.depth, self.context, self.salience, self.autonomy_index)
    
    @property
    def prime_factors(self) -> Dict[str, List[int]]:
        """Prime factorization of each dimension for unique decomposition"""
        def factorize(n):
            factors = []
            d = 2
            while d * d <= n:
                while n % d == 0:
                    factors.append(d)
                    n //= d
                d += 1
            if n > 1:
                factors.append(n)
            return factors
        
        return {
            'modality': factorize(self.modality),
            'depth': factorize(self.depth),
            'context': factorize(self.context),
            'salience': factorize(self.salience),
            'autonomy_index': factorize(self.autonomy_index)
        }


class HypergraphNode:
    """
    Hypergraph node representing an agent, state, or cognitive entity
    
    Encodes cognitive grammar elements as hypergraph fragments with
    tensor representations and semantic relationships.
    """
    
    def __init__(self, 
                 node_id: str = None,
                 node_type: str = "cognitive_entity",
                 tensor_signature: TensorSignature = None,
                 semantic_content: Dict[str, Any] = None):
        self.node_id = node_id or str(uuid.uuid4())
        self.node_type = node_type
        self.tensor_signature = tensor_signature or TensorSignature(4, 4, 8, 4, 2)
        self.semantic_content = semantic_content or {}
        self.links = []  # Hypergraph links to other nodes
        self.tensor_state = None  # Actual tensor representation
        
    def create_tensor_state(self, device='cpu', dtype=torch.float32) -> torch.Tensor:
        """Create tensor state from signature"""
        self.tensor_state = torch.zeros(self.tensor_signature.shape, 
                                      device=device, dtype=dtype)
        return self.tensor_state
    
    def add_link(self, target_node: 'HypergraphNode', link_type: str, weight: float = 1.0):
        """Add hypergraph link to another node"""
        link = {
            'target': target_node,
            'type': link_type,
            'weight': weight,
            'id': str(uuid.uuid4())
        }
        self.links.append(link)
        return link
    
    def __repr__(self):
        return f"HypergraphNode(id={self.node_id[:8]}, type={self.node_type}, shape={self.tensor_signature.shape})"


class TensorFragment:
    """
    Tensor fragment for encoding agents/states as hypergraph nodes & links
    
    Manages tensor operations on cognitive grammar elements with
    semantic-preserving transformations.
    """
    
    def __init__(self, 
                 fragment_id: str = None,
                 nodes: List[HypergraphNode] = None,
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float32):
        self.fragment_id = fragment_id or str(uuid.uuid4())
        self.nodes = nodes or []
        self.device = device
        self.dtype = dtype
        self.adjacency_matrix = None
        self.collective_tensor = None
        
    def add_node(self, node: HypergraphNode) -> HypergraphNode:
        """Add a hypergraph node to the fragment"""
        self.nodes.append(node)
        node.create_tensor_state(self.device, self.dtype)
        self._update_adjacency_matrix()
        return node
    
    def _update_adjacency_matrix(self):
        """Update adjacency matrix based on node links"""
        n = len(self.nodes)
        if n == 0:
            return
            
        self.adjacency_matrix = torch.zeros((n, n), device=self.device, dtype=self.dtype)
        
        node_to_idx = {node.node_id: i for i, node in enumerate(self.nodes)}
        
        for i, node in enumerate(self.nodes):
            for link in node.links:
                target_id = link['target'].node_id
                if target_id in node_to_idx:
                    j = node_to_idx[target_id]
                    self.adjacency_matrix[i, j] = link['weight']
    
    def create_collective_tensor(self) -> torch.Tensor:
        """Create collective tensor representation of all nodes"""
        if not self.nodes:
            return torch.tensor([], device=self.device, dtype=self.dtype)
        
        # Get all node tensors
        node_tensors = [node.tensor_state for node in self.nodes if node.tensor_state is not None]
        if not node_tensors:
            return torch.tensor([], device=self.device, dtype=self.dtype)
        
        # Handle different tensor shapes by flattening and padding
        if len(node_tensors) == 1:
            self.collective_tensor = node_tensors[0].unsqueeze(0)
        else:
            # Flatten all tensors
            flattened_tensors = [tensor.flatten() for tensor in node_tensors]
            
            # Find maximum size
            max_size = max(tensor.numel() for tensor in flattened_tensors)
            
            # Pad all tensors to same size
            padded_tensors = []
            for tensor in flattened_tensors:
                current_size = tensor.numel()
                if current_size < max_size:
                    padding = torch.zeros(max_size - current_size, device=self.device, dtype=self.dtype)
                    padded_tensor = torch.cat([tensor, padding])
                else:
                    padded_tensor = tensor
                padded_tensors.append(padded_tensor)
            
            # Stack padded tensors
            self.collective_tensor = torch.stack(padded_tensors, dim=0)
        
        return self.collective_tensor
    
    def pattern_transformation(self, transformation_matrix: torch.Tensor) -> 'TensorFragment':
        """Apply pattern transformation to the fragment"""
        new_fragment = TensorFragment(device=self.device, dtype=self.dtype)
        
        for node in self.nodes:
            # Create transformed node
            new_node = HypergraphNode(
                node_type=node.node_type,
                tensor_signature=node.tensor_signature,
                semantic_content=node.semantic_content.copy()
            )
            
            # Apply transformation to tensor state
            if node.tensor_state is not None:
                # Flatten, transform, reshape
                flat_tensor = node.tensor_state.flatten()
                if transformation_matrix.shape[1] == flat_tensor.shape[0]:
                    transformed_flat = transformation_matrix @ flat_tensor
                    new_node.tensor_state = transformed_flat.reshape(node.tensor_signature.shape)
                else:
                    new_node.tensor_state = node.tensor_state.clone()
            
            new_fragment.add_node(new_node)
        
        return new_fragment
    
    def __repr__(self):
        return f"TensorFragment(id={self.fragment_id[:8]}, nodes={len(self.nodes)})"


class AtomSpaceAdapter(ABC):
    """
    Abstract adapter for AtomSpace hypergraph integration
    
    Provides interface for bidirectional translation between
    cognitive grammar elements and AtomSpace representations.
    """
    
    @abstractmethod
    def to_atomspace(self, tensor_fragment: TensorFragment) -> Dict[str, Any]:
        """Convert tensor fragment to AtomSpace representation"""
        pass
    
    @abstractmethod
    def from_atomspace(self, atomspace_data: Dict[str, Any]) -> TensorFragment:
        """Convert AtomSpace representation to tensor fragment"""
        pass
    
    @abstractmethod
    def sync_bidirectional(self, tensor_fragment: TensorFragment) -> TensorFragment:
        """Synchronize bidirectional updates between representations"""
        pass


class CognitiveGrammarNetwork:
    """
    Main network orchestrator for distributed agentic cognitive grammar
    
    Manages tensor fragments, hypergraph encoding, and pattern transformations
    across the distributed cognitive mesh.
    """
    
    def __init__(self, 
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float32):
        self.device = device
        self.dtype = dtype
        self.fragments = {}  # Fragment registry
        self.atomspace_adapter = None
        self.pattern_library = {}  # Library of transformation patterns
        
    def register_fragment(self, fragment: TensorFragment) -> str:
        """Register a tensor fragment in the network"""
        self.fragments[fragment.fragment_id] = fragment
        return fragment.fragment_id
    
    def create_agentic_fragment(self, 
                              agent_type: str,
                              modality: ModalityType = ModalityType.AGENTIC,
                              depth: int = 4,
                              context: int = 8) -> TensorFragment:
        """Create a new agentic tensor fragment"""
        # Create signature with prime factorization mapping
        signature = TensorSignature(
            modality=modality.value.__hash__() % 16 + 1,  # Map modality to prime
            depth=depth,
            context=context,
            salience=4,  # Default salience
            autonomy_index=2  # Default autonomy
        )
        
        # Create agent node
        agent_node = HypergraphNode(
            node_type=agent_type,
            tensor_signature=signature,
            semantic_content={'agent_type': agent_type, 'modality': modality.value}
        )
        
        # Create fragment
        fragment = TensorFragment(device=self.device, dtype=self.dtype)
        fragment.add_node(agent_node)
        
        # Register fragment
        self.register_fragment(fragment)
        
        return fragment
    
    def apply_transformation_pattern(self, 
                                   fragment_id: str,
                                   pattern_name: str) -> TensorFragment:
        """Apply a stored transformation pattern to a fragment"""
        if fragment_id not in self.fragments:
            raise ValueError(f"Fragment {fragment_id} not found")
        
        if pattern_name not in self.pattern_library:
            raise ValueError(f"Pattern {pattern_name} not found")
        
        fragment = self.fragments[fragment_id]
        transformation_matrix = self.pattern_library[pattern_name]
        
        return fragment.pattern_transformation(transformation_matrix)
    
    def add_transformation_pattern(self, name: str, matrix: torch.Tensor):
        """Add a transformation pattern to the library"""
        self.pattern_library[name] = matrix.to(device=self.device, dtype=self.dtype)
    
    def set_atomspace_adapter(self, adapter: AtomSpaceAdapter):
        """Set the AtomSpace adapter for hypergraph integration"""
        self.atomspace_adapter = adapter
    
    def get_network_summary(self) -> Dict[str, Any]:
        """Get summary of the cognitive grammar network state"""
        return {
            'total_fragments': len(self.fragments),
            'total_nodes': sum(len(f.nodes) for f in self.fragments.values()),
            'pattern_library_size': len(self.pattern_library),
            'device': self.device,
            'dtype': str(self.dtype),
            'has_atomspace_adapter': self.atomspace_adapter is not None
        }
    
    def __repr__(self):
        summary = self.get_network_summary()
        return f"CognitiveGrammarNetwork(fragments={summary['total_fragments']}, nodes={summary['total_nodes']})"