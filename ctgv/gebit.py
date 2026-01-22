"""
CTGV System - Gebit Implementation
"""
from typing import Dict, Set, Optional
import math
from .shapes import Shape, GEOMETRIC_RULES
from .vector_field import VectorField

class Gebit:
    """Fundamental CTGV unit with geometry and fields"""

    def __init__(self, shape: Shape, intensity: float = 1.0,
                 label: str = None, dimensions: int = 3):
        self.shape = shape
        self.intensity = intensity
        self.state = 0.0
        self.field = VectorField(dimensions)

        # Weighted connections (topology)
        self.connections: Dict['Gebit', float] = {}
        self.locked_connections: Set['Gebit'] = set()

        # Temporal memory (limited to prevent memory leaks)
        self.history: list = []
        self.activation_count = 0

        # Metadata
        self.label = label or f"{shape.name}_{id(self) % 10000:04d}"
        self.constraint = GEOMETRIC_RULES.get(shape, GEOMETRIC_RULES[Shape.ORIGIN])

        # Interference cache
        self._interference_cache: Dict['Gebit', float] = {}
        self._cache_valid = False

    def connect_to(self, other: 'Gebit', weight: float = 1.0,
                   locked: bool = False) -> bool:
        """Connects with geometric validation"""
        # Validate constraints
        if len(self.connections) >= self.constraint.max_connections:
            return False

        # Check for cycles if not allowed
        if not self.constraint.allow_cycles:
            if self._creates_cycle(other):
                return False

        # Create connection
        self.connections[other] = weight
        if locked:
            self.locked_connections.add(other)

        # Symmetry if required
        if self.constraint.symmetric_required and self.shape == other.shape:
            other.connections[self] = weight

        # Invalidate cache
        self._cache_valid = False
        other._cache_valid = False

        return True

    def _creates_cycle(self, target: 'Gebit') -> bool:
        """Detects cycles via DFS"""
        visited = set()

        def dfs(node):
            if node == target:
                return True
            if node in visited:
                return False
            visited.add(node)
            return any(dfs(n) for n in node.connections)

        return dfs(self)

    def calculate_field_interaction(self, neighbor: 'Gebit') -> float:
        """Field interference with cache"""
        if self._cache_valid and neighbor in self._interference_cache:
            return self._interference_cache[neighbor]

        if neighbor not in self.connections:
            return 0.0

        weight = self.connections[neighbor]
        interference = self.field.interfere(neighbor.field)

        # Shape modulation
        if self.shape == Shape.RESONATOR:
            interference *= (1.0 + 0.1 * math.sin(self.activation_count * 0.5))
        elif self.shape == Shape.INHIBITOR:
            interference = max(0, interference * 0.5)

        result = interference * weight
        self._interference_cache[neighbor] = result
        self._cache_valid = True  # Fixed: Set cache as valid after calculation

        return result

    def invalidate_cache(self):
        """Invalidates interference cache"""
        self._cache_valid = False
        self._interference_cache.clear()

    def add_to_history(self, state: float, max_history: int = 100):
        """Add state to history with size limit"""
        self.history.append(state)
        if len(self.history) > max_history:
            self.history.pop(0)  # Remove oldest