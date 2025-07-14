"""
Tensor Shape Registry and Prime Factorization Mapping

This module manages tensor shapes, their prime factorization mappings,
and provides a registry for cognitive kernel tensor signatures.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
from pathlib import Path

from ..core import TensorSignature, ModalityType


class SemanticDimension(Enum):
    """Semantic dimensions for tensor shape assignment"""
    MODALITY = "modality"          # Type of cognitive modality
    DEPTH = "depth"                # Processing depth/complexity
    CONTEXT = "context"            # Context window size
    SALIENCE = "salience"          # Attention/importance level
    AUTONOMY_INDEX = "autonomy_index"  # Agent autonomy level
    TEMPORAL = "temporal"          # Temporal dynamics
    SPATIAL = "spatial"            # Spatial relationships
    SEMANTIC = "semantic"          # Semantic content complexity


@dataclass
class KernelTensorShape:
    """
    Tensor shape specification for cognitive kernels
    
    Each kernel's tensor shape represents the product of its
    independent semantic dimensions with prime factorization.
    """
    kernel_name: str
    tensor_signature: TensorSignature
    semantic_dimensions: Dict[SemanticDimension, int]
    prime_factorization: Dict[str, List[int]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate prime factorization after initialization"""
        self.prime_factorization = self.tensor_signature.prime_factors
        
    @property
    def total_parameters(self) -> int:
        """Calculate total parameters in the tensor"""
        shape = self.tensor_signature.shape
        return int(np.prod(shape))
    
    @property
    def complexity_score(self) -> float:
        """Calculate complexity score based on prime factors"""
        # More prime factors indicate higher complexity
        total_factors = sum(len(factors) for factors in self.prime_factorization.values())
        unique_primes = len(set().union(*self.prime_factorization.values()))
        
        return total_factors * 0.7 + unique_primes * 0.3
    
    def is_compatible_with(self, other: 'KernelTensorShape') -> bool:
        """Check if two kernel shapes are compatible for operations"""
        # Check if dimensions align for matrix operations
        self_shape = self.tensor_signature.shape
        other_shape = other.tensor_signature.shape
        
        # Compatible if last dimension of self matches first of other
        if len(self_shape) >= 2 and len(other_shape) >= 2:
            return self_shape[-1] == other_shape[0]
        
        return self_shape == other_shape


class PrimeFactorizationMapper:
    """
    Prime factorization mapper for tensor dimensions
    
    Maps semantic complexity to prime factorizations for
    unique decomposition and efficient computation.
    """
    
    def __init__(self):
        self.prime_cache = {}  # Cache for computed primes
        self.factorization_cache = {}  # Cache for factorizations
        self._generate_prime_cache(1000)  # Pre-compute primes up to 1000
    
    def _generate_prime_cache(self, max_num: int):
        """Generate cache of prime numbers up to max_num"""
        def sieve_of_eratosthenes(n):
            primes = [True] * (n + 1)
            primes[0] = primes[1] = False
            
            for i in range(2, int(n**0.5) + 1):
                if primes[i]:
                    for j in range(i*i, n + 1, i):
                        primes[j] = False
            
            return [i for i in range(2, n + 1) if primes[i]]
        
        primes = sieve_of_eratosthenes(max_num)
        self.prime_cache = {i: p for i, p in enumerate(primes)}
    
    def get_nth_prime(self, n: int) -> int:
        """Get the nth prime number"""
        if n in self.prime_cache:
            return self.prime_cache[n]
        
        # Generate more primes if needed
        current_max = max(self.prime_cache.values()) if self.prime_cache else 2
        self._generate_prime_cache(current_max * 2)
        
        return self.prime_cache.get(n, 2)  # Default to 2 if not found
    
    def factorize(self, n: int) -> List[int]:
        """Prime factorization of n"""
        if n in self.factorization_cache:
            return self.factorization_cache[n]
        
        factors = []
        d = 2
        original_n = n
        
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        
        if n > 1:
            factors.append(n)
        
        self.factorization_cache[original_n] = factors
        return factors
    
    def map_semantic_complexity_to_prime(self, 
                                       semantic_dim: SemanticDimension,
                                       complexity_level: int) -> int:
        """Map semantic complexity to appropriate prime number"""
        # Different semantic dimensions use different prime ranges
        prime_base = {
            SemanticDimension.MODALITY: 0,      # Primes starting from 2
            SemanticDimension.DEPTH: 10,       # Primes starting from 29
            SemanticDimension.CONTEXT: 20,     # Primes starting from 71
            SemanticDimension.SALIENCE: 30,    # Primes starting from 127
            SemanticDimension.AUTONOMY_INDEX: 40,  # Primes starting from 179
            SemanticDimension.TEMPORAL: 50,    # Primes starting from 229
            SemanticDimension.SPATIAL: 60,     # Primes starting from 283
            SemanticDimension.SEMANTIC: 70     # Primes starting from 349
        }
        
        base_idx = prime_base.get(semantic_dim, 0)
        prime_idx = base_idx + complexity_level
        
        return self.get_nth_prime(prime_idx)
    
    def create_tensor_signature_from_semantics(self,
                                             semantic_specs: Dict[SemanticDimension, int]) -> TensorSignature:
        """Create tensor signature from semantic specifications"""
        # Map semantic dimensions to tensor signature fields
        modality = semantic_specs.get(SemanticDimension.MODALITY, 4)
        depth = semantic_specs.get(SemanticDimension.DEPTH, 4)
        context = semantic_specs.get(SemanticDimension.CONTEXT, 8)
        salience = semantic_specs.get(SemanticDimension.SALIENCE, 4)
        autonomy = semantic_specs.get(SemanticDimension.AUTONOMY_INDEX, 2)
        
        # Convert to prime-based dimensions
        modality_prime = self.map_semantic_complexity_to_prime(SemanticDimension.MODALITY, modality)
        depth_prime = self.map_semantic_complexity_to_prime(SemanticDimension.DEPTH, depth)
        context_prime = self.map_semantic_complexity_to_prime(SemanticDimension.CONTEXT, context)
        salience_prime = self.map_semantic_complexity_to_prime(SemanticDimension.SALIENCE, salience)
        autonomy_prime = self.map_semantic_complexity_to_prime(SemanticDimension.AUTONOMY_INDEX, autonomy)
        
        return TensorSignature(
            modality=min(modality_prime, 32),  # Cap to reasonable size
            depth=min(depth_prime, 32),
            context=min(context_prime, 64),
            salience=min(salience_prime, 16),
            autonomy_index=min(autonomy_prime, 8)
        )
    
    def find_optimal_decomposition(self, target_shape: Tuple[int, ...]) -> Dict[str, List[int]]:
        """Find optimal prime decomposition for tensor operations"""
        decomposition = {}
        
        for i, dim in enumerate(target_shape):
            dim_name = f"dim_{i}"
            factors = self.factorize(dim)
            decomposition[dim_name] = factors
        
        return decomposition
    
    def suggest_kernel_shapes(self, 
                            function_complexity: int,
                            input_modalities: List[ModalityType],
                            output_requirements: Dict[str, int]) -> List[KernelTensorShape]:
        """Suggest optimal kernel shapes for given requirements"""
        suggestions = []
        
        for modality in input_modalities:
            # Create semantic specifications
            semantic_specs = {
                SemanticDimension.MODALITY: len(input_modalities),
                SemanticDimension.DEPTH: function_complexity,
                SemanticDimension.CONTEXT: output_requirements.get('context_size', 8),
                SemanticDimension.SALIENCE: output_requirements.get('attention_levels', 4),
                SemanticDimension.AUTONOMY_INDEX: output_requirements.get('autonomy_level', 2)
            }
            
            # Generate tensor signature
            signature = self.create_tensor_signature_from_semantics(semantic_specs)
            
            # Create kernel shape
            kernel_shape = KernelTensorShape(
                kernel_name=f"{modality.value}_kernel",
                tensor_signature=signature,
                semantic_dimensions=semantic_specs,
                metadata={
                    'input_modalities': [m.value for m in input_modalities],
                    'function_complexity': function_complexity,
                    'output_requirements': output_requirements
                }
            )
            
            suggestions.append(kernel_shape)
        
        return suggestions


class TensorShapeRegistry:
    """
    Registry for cognitive kernel tensor shapes
    
    Manages catalog of tensor shapes, their relationships,
    and optimization patterns for the cognitive grammar network.
    """
    
    def __init__(self, registry_path: Optional[str] = None):
        self.registry_path = Path(registry_path) if registry_path else Path("tensor_registry.json")
        self.kernel_shapes = {}  # Map of kernel_name -> KernelTensorShape
        self.shape_relationships = {}  # Map of compatible shapes
        self.optimization_patterns = {}  # Map of optimization patterns
        self.prime_mapper = PrimeFactorizationMapper()
        self.usage_statistics = {}  # Track usage patterns
        
        # Load existing registry if available
        self.load_registry()
    
    def register_kernel_shape(self, kernel_shape: KernelTensorShape) -> str:
        """Register a new kernel tensor shape"""
        kernel_id = kernel_shape.kernel_name
        
        if kernel_id in self.kernel_shapes:
            print(f"Warning: Overwriting existing kernel shape '{kernel_id}'")
        
        self.kernel_shapes[kernel_id] = kernel_shape
        self._update_relationships(kernel_shape)
        self._update_usage_stats(kernel_id)
        
        return kernel_id
    
    def get_kernel_shape(self, kernel_name: str) -> Optional[KernelTensorShape]:
        """Get kernel shape by name"""
        return self.kernel_shapes.get(kernel_name)
    
    def find_compatible_kernels(self, kernel_name: str) -> List[KernelTensorShape]:
        """Find kernels compatible with the given kernel"""
        target_kernel = self.kernel_shapes.get(kernel_name)
        if not target_kernel:
            return []
        
        compatible = []
        for other_name, other_kernel in self.kernel_shapes.items():
            if other_name != kernel_name and target_kernel.is_compatible_with(other_kernel):
                compatible.append(other_kernel)
        
        return compatible
    
    def _update_relationships(self, kernel_shape: KernelTensorShape):
        """Update compatibility relationships"""
        kernel_name = kernel_shape.kernel_name
        self.shape_relationships[kernel_name] = []
        
        for other_name, other_shape in self.kernel_shapes.items():
            if other_name != kernel_name:
                if kernel_shape.is_compatible_with(other_shape):
                    self.shape_relationships[kernel_name].append(other_name)
    
    def _update_usage_stats(self, kernel_name: str):
        """Update usage statistics"""
        if kernel_name not in self.usage_statistics:
            self.usage_statistics[kernel_name] = {
                'registration_count': 0,
                'access_count': 0,
                'last_accessed': None
            }
        
        self.usage_statistics[kernel_name]['registration_count'] += 1
    
    def suggest_optimizations(self, kernel_name: str) -> List[Dict[str, Any]]:
        """Suggest optimizations for a kernel shape"""
        kernel_shape = self.kernel_shapes.get(kernel_name)
        if not kernel_shape:
            return []
        
        suggestions = []
        
        # Check for overly complex shapes
        if kernel_shape.complexity_score > 10:
            suggestions.append({
                'type': 'complexity_reduction',
                'message': 'Consider reducing tensor complexity',
                'current_complexity': kernel_shape.complexity_score
            })
        
        # Check for inefficient prime factorizations
        shape = kernel_shape.tensor_signature.shape
        if any(dim > 100 for dim in shape):
            suggestions.append({
                'type': 'dimension_optimization',
                'message': 'Large dimensions may impact performance',
                'large_dimensions': [i for i, dim in enumerate(shape) if dim > 100]
            })
        
        # Suggest compatible kernels for composition
        compatible = self.find_compatible_kernels(kernel_name)
        if compatible:
            suggestions.append({
                'type': 'composition_opportunity',
                'message': f'Can compose with {len(compatible)} other kernels',
                'compatible_kernels': [k.kernel_name for k in compatible[:3]]  # Top 3
            })
        
        return suggestions
    
    def create_kernel_catalog(self) -> Dict[str, Any]:
        """Create comprehensive catalog of all registered kernels"""
        catalog = {
            'total_kernels': len(self.kernel_shapes),
            'kernels': {},
            'relationships': self.shape_relationships,
            'statistics': self.usage_statistics,
            'optimization_opportunities': {}
        }
        
        for kernel_name, kernel_shape in self.kernel_shapes.items():
            catalog['kernels'][kernel_name] = {
                'tensor_signature': kernel_shape.tensor_signature.__dict__,
                'semantic_dimensions': {dim.value: val for dim, val in kernel_shape.semantic_dimensions.items()},
                'prime_factorization': kernel_shape.prime_factorization,
                'total_parameters': kernel_shape.total_parameters,
                'complexity_score': kernel_shape.complexity_score,
                'metadata': kernel_shape.metadata
            }
            
            # Add optimization suggestions
            catalog['optimization_opportunities'][kernel_name] = self.suggest_optimizations(kernel_name)
        
        return catalog
    
    def save_registry(self, path: Optional[str] = None):
        """Save registry to file"""
        save_path = Path(path) if path else self.registry_path
        catalog = self.create_kernel_catalog()
        
        try:
            with open(save_path, 'w') as f:
                json.dump(catalog, f, indent=2, default=str)
            print(f"Registry saved to {save_path}")
        except Exception as e:
            print(f"Failed to save registry: {e}")
    
    def load_registry(self, path: Optional[str] = None):
        """Load registry from file"""
        load_path = Path(path) if path else self.registry_path
        
        if not load_path.exists():
            print(f"Registry file {load_path} not found, starting with empty registry")
            return
        
        try:
            with open(load_path, 'r') as f:
                catalog = json.load(f)
            
            # Reconstruct kernel shapes
            for kernel_name, kernel_data in catalog.get('kernels', {}).items():
                # Reconstruct tensor signature
                sig_data = kernel_data['tensor_signature']
                signature = TensorSignature(**sig_data)
                
                # Reconstruct semantic dimensions
                semantic_dims = {}
                for dim_name, value in kernel_data['semantic_dimensions'].items():
                    try:
                        semantic_dims[SemanticDimension(dim_name)] = value
                    except ValueError:
                        pass  # Skip unknown dimensions
                
                # Create kernel shape
                kernel_shape = KernelTensorShape(
                    kernel_name=kernel_name,
                    tensor_signature=signature,
                    semantic_dimensions=semantic_dims,
                    metadata=kernel_data.get('metadata', {})
                )
                
                self.kernel_shapes[kernel_name] = kernel_shape
            
            # Load relationships and statistics
            self.shape_relationships = catalog.get('relationships', {})
            self.usage_statistics = catalog.get('statistics', {})
            
            print(f"Loaded {len(self.kernel_shapes)} kernel shapes from {load_path}")
            
        except Exception as e:
            print(f"Failed to load registry: {e}")
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """Get summary of registry contents"""
        if not self.kernel_shapes:
            return {"total_kernels": 0, "message": "Registry is empty"}
        
        total_params = sum(ks.total_parameters for ks in self.kernel_shapes.values())
        avg_complexity = np.mean([ks.complexity_score for ks in self.kernel_shapes.values()])
        
        modality_distribution = {}
        for kernel_shape in self.kernel_shapes.values():
            for dim, value in kernel_shape.semantic_dimensions.items():
                if dim == SemanticDimension.MODALITY:
                    modality_distribution[value] = modality_distribution.get(value, 0) + 1
        
        return {
            "total_kernels": len(self.kernel_shapes),
            "total_parameters": total_params,
            "average_complexity": avg_complexity,
            "modality_distribution": modality_distribution,
            "most_used_kernels": sorted(
                self.usage_statistics.items(),
                key=lambda x: x[1].get('access_count', 0),
                reverse=True
            )[:5]
        }
    
    def optimize_registry(self) -> Dict[str, Any]:
        """Optimize the entire registry for better performance"""
        optimization_results = {
            'kernels_optimized': 0,
            'relationships_updated': 0,
            'patterns_identified': 0
        }
        
        # Update all relationships
        for kernel_shape in self.kernel_shapes.values():
            self._update_relationships(kernel_shape)
            optimization_results['relationships_updated'] += 1
        
        # Identify common patterns
        complexity_groups = {}
        for kernel_name, kernel_shape in self.kernel_shapes.items():
            complexity = int(kernel_shape.complexity_score)
            if complexity not in complexity_groups:
                complexity_groups[complexity] = []
            complexity_groups[complexity].append(kernel_name)
        
        # Store optimization patterns
        self.optimization_patterns['complexity_groups'] = complexity_groups
        optimization_results['patterns_identified'] = len(complexity_groups)
        
        return optimization_results