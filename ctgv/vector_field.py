"""
CTGV System - Vector Field Implementation
"""
import numpy as np
import math

class VectorField:
    """Electromagnetic/photonic field representation"""

    __slots__ = ['vector', 'phase', 'coherence', '_norm_cache']

    def __init__(self, dimensions: int = 3):
        self.vector = np.zeros(dimensions, dtype=np.float32)
        self.phase = 0.0
        self.coherence = 1.0
        self._norm_cache = 0.0

    def interfere(self, other: 'VectorField') -> float:
        """Constructive/destructive interference"""
        # Update cache if necessary
        if self._norm_cache == 0.0:
            self._norm_cache = np.linalg.norm(self.vector)

        dot_product = np.dot(self.vector, other.vector)
        phase_diff = abs(self.phase - other.phase)

        # Interference = projection × phase coherence
        return dot_product * math.cos(phase_diff) * self.coherence

    def normalize(self, target_magnitude: float = 1.0):
        """In-place normalization for efficiency"""
        norm = np.linalg.norm(self.vector)
        if norm > 1e-10:
            self.vector *= (target_magnitude / norm)
            self._norm_cache = target_magnitude
        else:
            self._norm_cache = 0.0