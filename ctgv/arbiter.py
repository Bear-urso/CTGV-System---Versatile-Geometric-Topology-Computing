"""
CTGV System - Temporal Binding Arbiter
"""
from typing import List, Dict
from .gebit import Gebit
from .engine import CTGVEngine

class TemporalBindingArbiter:
    """Resolves ambiguities through temporal binding"""

    def __init__(self, engine: CTGVEngine, binding_threshold: float = 0.3,
                 enable_parallel: bool = False):
        self.engine = engine
        self.binding_threshold = binding_threshold
        self.enable_parallel = enable_parallel

    def resolve_ambiguity(self, narratives: List[Gebit],
                         max_cycles: int = 30) -> Dict:
        """
        Resolve competing narratives through temporal binding

        Args:
            narratives: List of competing narrative sources
            max_cycles: Maximum binding cycles

        Returns:
            Resolution results
        """
        if len(narratives) < 2:
            return {
                'dominant_narrative': narratives[0].label if narratives else None,
                'narrative_strengths': {n.label: n.state for n in narratives},
                'final_ambiguity': 0.0
            }

        # Initialize binding process
        binding_cycles = 0
        ambiguity = 1.0

        while binding_cycles < max_cycles and ambiguity > self.binding_threshold:
            # Propagate each narrative
            strengths = {}
            for narrative in narratives:
                result = self.engine.propagate([narrative], reset=False)
                strengths[narrative.label] = result['global_coherence']

            # Calculate ambiguity (normalized variance)
            strength_values = list(strengths.values())
            if len(strength_values) > 1:
                mean_strength = sum(strength_values) / len(strength_values)
                variance = sum((s - mean_strength)**2 for s in strength_values) / len(strength_values)
                ambiguity = variance / (mean_strength**2 + 1e-10) if mean_strength > 0 else 0.0
            else:
                ambiguity = 0.0

            binding_cycles += 1

        # Determine dominant narrative
        dominant = max(strengths.items(), key=lambda x: x[1])[0] if strengths else None

        return {
            'dominant_narrative': dominant,
            'narrative_strengths': strengths,
            'final_ambiguity': ambiguity,
            'binding_cycles': binding_cycles
        }

    def resolve_ambiguity_parallel(self, narratives: List[Gebit],
                                  max_cycles: int = 30) -> Dict:
        """
        Parallel version of ambiguity resolution for large networks
        """
        if len(narratives) < 2:
            return self.resolve_ambiguity(narratives, max_cycles)

        # Enable parallel processing in engine temporarily
        original_parallel = self.engine.enable_parallel
        self.engine.enable_parallel = True

        try:
            result = self.resolve_ambiguity(narratives, max_cycles)
            result['method'] = 'parallel'
            return result
        finally:
            self.engine.enable_parallel = original_parallel