"""
CTGV System - Core Engine
"""
import numpy as np
import math
from typing import List, Dict
from .gebit import Gebit
from .shapes import DECAY

class CTGVEngine:
    """Versatile Geometric Topological Computing Engine"""

    def __init__(self, threshold: float = 0.001, max_iterations: int = 1000,
                 use_superposition: bool = True, enable_adaptation: bool = False,
                 enable_parallel: bool = False, chunk_size: int = 100):
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.use_superposition = use_superposition
        self.enable_adaptation = enable_adaptation
        self.enable_parallel = enable_parallel
        self.chunk_size = chunk_size  # For chunked processing

        # Engine state
        self.current_iteration = 0
        self.global_coherence = 0.0

        # Logs
        self.iteration_log: List[Dict] = []

    def propagate(self, start_nodes: List[Gebit],
                  convergence_threshold: float = 0.0001,
                  reset: bool = True) -> Dict:
        """
        Topological field propagation

        Args:
            start_nodes: List of source gebits
            convergence_threshold: Convergence limit
            reset: If True, resets states before propagating

        Returns:
            Dict with execution metrics
        """
        # Initialization
        if reset:
            self._reset_network(start_nodes)

        active_set = set(start_nodes)
        for node in start_nodes:
            node.state = node.intensity
            node.field.vector = np.ones_like(node.field.vector) * node.intensity

        # Propagation loop
        self.current_iteration = 0
        delta_max = float('inf')

        while (self.current_iteration < self.max_iterations and
               delta_max > convergence_threshold):

            delta_max = 0.0
            next_active = set()

            if self.enable_parallel and len(active_set) > 10:
                # Use parallel propagation for larger networks
                active_list = list(active_set)
                new_states, field_contribs, deltas, next_active_batch = self._parallel_propagate_step(active_list)

                # Update nodes
                for i, node in enumerate(active_list):
                    if deltas[i] > self.threshold:
                        node.state = new_states[i]
                        node.activation_count += 1
                        node.add_to_history(new_states[i])

                        # Update field
                        if np.linalg.norm(field_contribs[i]) > 0:
                            node.field.vector = field_contribs[i]
                            node.field.normalize(self._get_target_norm(node.shape))
                            node.field.phase = (node.field.phase +
                                              0.1 * node.activation_count) % (2 * math.pi)

                delta_max = np.max(deltas) if len(deltas) > 0 else 0.0
                next_active = next_active_batch

            else:
                # Original sequential propagation
                for node in active_set:
                    if node.state < self.threshold:
                        continue

                    # Accumulate contributions
                    scalar_contrib = 0.0
                    field_contrib = np.zeros_like(node.field.vector)

                    for neighbor, weight in node.connections.items():
                        # Scalar contribution
                        neighbor_scalar = node.state * weight

                        # Vector contribution (interference)
                        field_interaction = node.calculate_field_interaction(neighbor)

                        if self.use_superposition:
                            scalar_contrib += neighbor_scalar + field_interaction
                        else:
                            scalar_contrib = max(scalar_contrib,
                                               neighbor_scalar + field_interaction)

                        # Vector field
                        field_contrib += neighbor.field.vector * weight

                    # Apply shape rule
                    new_state = self._apply_shape_rule(node, scalar_contrib)

                    # Update vector field
                    if np.linalg.norm(field_contrib) > 0:
                        node.field.vector = field_contrib
                        node.field.normalize(self._get_target_norm(node.shape))
                        node.field.phase = (node.field.phase +
                                          0.1 * node.activation_count) % (2 * math.pi)

                    # Check for change
                    delta = abs(new_state - node.state)
                    if delta > self.threshold:
                        node.state = new_state
                        node.activation_count += 1
                        node.add_to_history(new_state)  # Use new method

                        # Activate neighbors
                        for neighbor in node.connections:
                            next_active.add(neighbor)

                        delta_max = max(delta_max, delta)

            # Adaptation (if enabled)
            if self.enable_adaptation and self.current_iteration % 10 == 0:
                self._adaptive_rewiring(list(active_set))

            # Logging
            self._log_iteration(active_set, delta_max)

            # Next iteration
            active_set = next_active
            self.current_iteration += 1

        # Final metrics
        self.global_coherence = self._calculate_coherence(start_nodes)

        return {
            'iterations': self.current_iteration,
            'converged': delta_max <= convergence_threshold,
            'global_coherence': self.global_coherence,
            'final_states': {node.label: node.state for node in start_nodes}
        }

    def _parallel_propagate_step(self, active_nodes: List[Gebit]) -> tuple:
        """
        Parallel propagation step using vectorized operations
        Returns (new_states, field_contribs, deltas, next_active)
        """
        if not active_nodes:
            return [], [], [], set()

        n_nodes = len(active_nodes)
        new_states = np.zeros(n_nodes)
        field_contribs = np.zeros((n_nodes, active_nodes[0].field.vector.shape[0]))
        deltas = np.zeros(n_nodes)
        next_active = set()

        # Process in chunks for memory efficiency
        for i in range(0, n_nodes, self.chunk_size):
            chunk_end = min(i + self.chunk_size, n_nodes)
            chunk_nodes = active_nodes[i:chunk_end]

            for j, node in enumerate(chunk_nodes):
                if node.state < self.threshold:
                    continue

                # Vectorized neighbor processing
                neighbors = list(node.connections.keys())
                weights = np.array(list(node.connections.values()))

                if not neighbors:
                    continue

                # Batch field interactions
                neighbor_states = np.array([n.state for n in neighbors])
                neighbor_fields = np.array([n.field.vector for n in neighbors])

                # Scalar contributions
                scalar_contribs = neighbor_states * weights

                # Vector field contributions (batch dot product)
                field_contribs[i+j] = np.sum(neighbor_fields * weights[:, np.newaxis], axis=0)

                # Apply superposition or max
                if self.use_superposition:
                    total_scalar = np.sum(scalar_contribs)
                else:
                    total_scalar = np.max(scalar_contribs) if len(scalar_contribs) > 0 else 0.0

                # Add field interactions (simplified batch)
                field_interactions = np.sum([
                    node.calculate_field_interaction(neighbor) for neighbor in neighbors
                ])
                total_scalar += field_interactions

                # Apply shape rule
                new_states[i+j] = self._apply_shape_rule(node, total_scalar)

                # Calculate delta
                deltas[i+j] = abs(new_states[i+j] - node.state)

                # Add neighbors to next active set
                next_active.update(neighbors)

        return new_states, field_contribs, deltas, next_active

    def _reset_network(self, nodes: List[Gebit]):
        """Reset all nodes in network"""
        visited = set()
        to_visit = list(nodes)

        while to_visit:
            node = to_visit.pop()
            if node in visited:
                continue
            visited.add(node)

            node.state = 0.0
            node.activation_count = 0
            node.history.clear()
            node.invalidate_cache()

            to_visit.extend(node.connections.keys())

    def _apply_shape_rule(self, node: Gebit, contribution: float) -> float:
        """Apply shape-specific transformation"""
        decay = DECAY.get(node.shape, 1.0)
        return contribution * decay

    def _get_target_norm(self, shape) -> float:
        """Get target normalization magnitude for shape"""
        return 1.0  # Could be shape-specific

    def _adaptive_rewiring(self, active_nodes: List[Gebit]):
        """Adaptive network rewiring (placeholder)"""
        pass  # Implementation for future adaptation

    def _log_iteration(self, active_set, delta_max):
        """Log iteration data"""
        self.iteration_log.append({
            'iteration': self.current_iteration,
            'active_nodes': len(active_set),
            'max_delta': delta_max
        })

    def _calculate_coherence(self, nodes: List[Gebit]) -> float:
        """Calculate global network coherence"""
        if not nodes:
            return 0.0

        states = [node.state for node in nodes]
        mean_state = np.mean(states)
        variance = np.var(states)

        # Coherence = 1 / (1 + variance/mean^2) if mean > 0 else 0
        return 1.0 / (1.0 + variance / (mean_state**2 + 1e-10)) if mean_state > 0 else 0.0