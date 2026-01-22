"""
CTGV System - Clarification Mechanism
Módulo para desempate e redução de entropia através de intervenção pontual
"""
import math
import numpy as np
from typing import List, Dict, Optional, Tuple
from .gebit import Gebit
from .engine import CTGVEngine
from .arbiter import TemporalBindingArbiter

# =========================
# CLARIFICATION MECHANISM
# =========================

class ClarificationMechanism:
    """
    Pontual intervention mechanism for tie-breaking and entropy reduction
    Implements a 'clarification question' to resolve ambiguous states
    """

    def __init__(self, engine: CTGVEngine, sensitivity: float = 0.05):
        self.engine = engine
        self.sensitivity = sensitivity  # How small a difference triggers clarification
        self.clarification_history = []

    def check_ambiguous_state(self, nodes: List[Gebit]) -> Optional[Dict]:
        """
        Detects ambiguous states requiring clarification.
        Returns None if no clarification needed, or clarification data.
        """
        if len(nodes) < 2:
            return None

        # Get active states
        active_nodes = [n for n in nodes if n.state > self.engine.threshold]
        if len(active_nodes) < 2:
            return None

        # Calculate state differences
        states = [(n, n.state) for n in active_nodes]
        states.sort(key=lambda x: x[1], reverse=True)

        top_nodes = states[:2]
        strength_diff = abs(top_nodes[0][1] - top_nodes[1][1])

        # Check if difference is below sensitivity threshold
        if strength_diff < self.sensitivity:
            # Calculate entropy of the system
            entropy = self._calculate_state_entropy(active_nodes)

            clarification_data = {
                'type': 'tie_breaking',
                'top_nodes': [(n[0].label, n[1]) for n in top_nodes],
                'strength_difference': strength_diff,
                'system_entropy': entropy,
                'recommended_action': self._suggest_clarification(top_nodes, entropy)
            }

            self.clarification_history.append(clarification_data)
            return clarification_data

        return None

    def apply_clarification(self, nodes: List[Gebit],
                          clarification_data: Dict,
                          bias_factor: float = 0.1) -> Dict:
        """
        Applies a clarification bias to break ties and reduce entropy.

        Args:
            nodes: List of competing nodes
            clarification_data: From check_ambiguous_state()
            bias_factor: Strength of clarification intervention (0-1)

        Returns:
            Dict with clarification results
        """
        if clarification_data['type'] != 'tie_breaking':
            return {'applied': False, 'reason': 'Invalid clarification type'}

        # Get the top competing nodes
        top_labels = [n[0] for n in clarification_data['top_nodes']]
        top_gebits = [n for n in nodes if n.label in top_labels]

        if len(top_gebits) != 2:
            return {'applied': False, 'reason': 'Nodes not found'}

        # Apply clarification bias based on additional criteria
        bias_results = self._apply_multi_criteria_bias(top_gebits, bias_factor)

        # Calculate entropy reduction
        before_entropy = clarification_data['system_entropy']
        after_entropy = self._calculate_state_entropy(nodes)
        entropy_reduction = before_entropy - after_entropy

        result = {
            'applied': True,
            'bias_applied': bias_results,
            'entropy_before': before_entropy,
            'entropy_after': after_entropy,
            'entropy_reduction': entropy_reduction,
            'clarification_question': self._generate_clarification_question(bias_results)
        }

        return result

    def _calculate_state_entropy(self, nodes: List[Gebit]) -> float:
        """Calculates Shannon entropy of node states"""
        states = [n.state for n in nodes if n.state > 0]
        if not states:
            return 0.0

        # Normalize states to create probability distribution
        total = sum(states)
        if total == 0:
            return 0.0

        probabilities = [s / total for s in states]

        # Calculate Shannon entropy
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize to [0,1]
        max_entropy = math.log2(len(probabilities)) if len(probabilities) > 1 else 1.0
        if max_entropy > 0:
            entropy /= max_entropy

        return entropy

    def _suggest_clarification(self, top_nodes: List[Tuple[Gebit, float]],
                              entropy: float) -> str:
        """Suggests the type of clarification needed"""
        if entropy > 0.7:
            return "high_entropy_resolution"
        elif top_nodes[0][1] < 0.3:
            return "low_confidence_boost"
        else:
            return "subtle_tie_break"

    def _apply_multi_criteria_bias(self, nodes: List[Gebit],
                                  bias_factor: float) -> Dict:
        """
        Applies bias based on multiple criteria:
        1. Historical stability
        2. Field coherence
        3. Connection strength
        4. Geometric centrality
        """
        criteria_scores = []

        for node in nodes:
            scores = {
                'node': node.label,
                'historical_stability': self._calculate_stability_score(node),
                'field_coherence': self._calculate_field_coherence(node),
                'connection_strength': self._calculate_avg_connection_strength(node),
                'geometric_centrality': self._calculate_centrality_score(node),
                'total_score': 0.0
            }

            # Weighted total score
            scores['total_score'] = (
                scores['historical_stability'] * 0.3 +
                scores['field_coherence'] * 0.25 +
                scores['connection_strength'] * 0.25 +
                scores['geometric_centrality'] * 0.2
            )

            criteria_scores.append(scores)

        # Find winner based on multi-criteria
        winner = max(criteria_scores, key=lambda x: x['total_score'])
        loser = min(criteria_scores, key=lambda x: x['total_score'])

        # Apply bias
        for node in nodes:
            if node.label == winner['node']:
                node.state *= (1.0 + bias_factor)
            elif node.label == loser['node']:
                node.state *= (1.0 - bias_factor * 0.5)

        return {
            'winner': winner['node'],
            'winner_score': winner['total_score'],
            'loser': loser['node'],
            'loser_score': loser['total_score'],
            'criteria_breakdown': criteria_scores
        }

    def _calculate_stability_score(self, node: Gebit) -> float:
        """Calculates historical stability score"""
        if len(node.history) < 3:
            return 0.5

        recent = node.history[-5:] if len(node.history) >= 5 else node.history
        stability = 1.0 - np.std(recent)
        return max(0.0, min(1.0, stability))

    def _calculate_field_coherence(self, node: Gebit) -> float:
        """Calculates field coherence with neighbors"""
        if not node.connections:
            return 0.5

        coherences = []
        for neighbor in node.connections:
            interference = node.calculate_field_interaction(neighbor)
            weight = node.connections[neighbor]
            coherences.append(interference * weight)

        if not coherences:
            return 0.5

        return np.mean(coherences)

    def _calculate_avg_connection_strength(self, node: Gebit) -> float:
        """Average strength of connections"""
        if not node.connections:
            return 0.0
        return np.mean(list(node.connections.values()))

    def _calculate_centrality_score(self, node: Gebit) -> float:
        """Geometric/topological centrality"""
        # Simplified centrality: degree normalized by possible connections
        max_possible = node.constraint.max_connections
        if max_possible == float('inf'):
            return 0.5

        current = len(node.connections)
        return current / max_possible if max_possible > 0 else 0.0

    def _generate_clarification_question(self, bias_results: Dict) -> str:
        """Generates a human-readable clarification question"""
        winner = bias_results['winner']
        loser = bias_results['loser']

        # Find the strongest criterion for the decision
        criteria = bias_results['criteria_breakdown']
        winner_criteria = next(c for c in criteria if c['node'] == winner)

        # Identify the most decisive criterion
        decisive_criteria = max(
            ['historical_stability', 'field_coherence',
             'connection_strength', 'geometric_centrality'],
            key=lambda k: winner_criteria[k]
        )

        questions = {
            'historical_stability': f"Should temporal stability favor '{winner}' over '{loser}'?",
            'field_coherence': f"Should field coherence prioritize '{winner}' over '{loser}'?",
            'connection_strength': f"Should connection strength favor '{winner}' over '{loser}'?",
            'geometric_centrality': f"Should topological centrality prioritize '{winner}' over '{loser}'?"
        }

        return questions.get(decisive_criteria,
                            f"Clarify preference between '{winner}' and '{loser}'")


# =========================
# ENHANCED TBA WITH CLARIFICATION
# =========================

class EnhancedTemporalBindingArbiter(TemporalBindingArbiter):
    """
    TBA enhanced with clarification mechanism for tie-breaking
    """

    def __init__(self, engine: CTGVEngine, binding_threshold: float = 0.7,
                 clarification_sensitivity: float = 0.05):
        super().__init__(engine, binding_threshold)
        self.clarifier = ClarificationMechanism(engine, clarification_sensitivity)
        self.clarifications_applied = 0

    def resolve_ambiguity(self, competing_sources: List[Gebit],
                         max_cycles: int = 50,
                         enable_clarification: bool = True) -> Dict:
        """
        Enhanced resolution with clarification mechanism
        """
        print(f"\n[Enhanced TBA] Starting with {len(competing_sources)} sources")

        # Initialize
        for source in competing_sources:
            source.state = source.intensity

        for cycle in range(max_cycles):
            # Propagate
            result = self.engine.propagate(competing_sources, reset=False)

            # Calculate strengths
            for source in competing_sources:
                strength = self._calculate_narrative_strength(source, result)
                self.narrative_strengths[source.label] = (
                    self.narrative_strengths.get(source.label, 0.0) * 0.7 +
                    strength * 0.3
                )

            # Check for clarification opportunity
            if enable_clarification and cycle % 5 == 0:  # Check periodically
                clarification_needed = self.clarifier.check_ambiguous_state(
                    competing_sources
                )

                if clarification_needed:
                    print(f"  Cycle {cycle}: CLARIFICATION TRIGGERED")
                    print(f"    Top contenders: {clarification_needed['top_nodes']}")
                    print(f"    Strength diff: {clarification_needed['strength_difference']:.4f}")
                    print(f"    System entropy: {clarification_needed['system_entropy']:.4f}")
                    print(f"    Action: {clarification_needed['recommended_action']}")

                    # Apply clarification
                    clarification_result = self.clarifier.apply_clarification(
                        competing_sources,
                        clarification_needed,
                        bias_factor=0.15
                    )

                    if clarification_result['applied']:
                        self.clarifications_applied += 1
                        print(f"    Clarification applied: {clarification_result['clarification_question']}")
                        print(f"    Entropy reduction: {clarification_result['entropy_reduction']:.4f}")

            # Calculate ambiguity
            ambiguity = self._calculate_ambiguity()
            self.ambiguity_history.append(ambiguity)

            print(f"  Cycle {cycle}: Ambiguity = {ambiguity:.3f}")

            # Check convergence
            if ambiguity < self.binding_threshold:
                dominant_label = max(self.narrative_strengths.items(),
                                   key=lambda x: x[1])[0]
                print(f"  Dominant narrative: {dominant_label}")
                break

        # Final reinforcement
        self._reinforce_dominant(competing_sources)

        # Compile results
        final_results = {
            'dominant_narrative': max(self.narrative_strengths.items(),
                                    key=lambda x: x[1])[0] if self.narrative_strengths else None,
            'narrative_strengths': self.narrative_strengths,
            'final_ambiguity': self.ambiguity_history[-1] if self.ambiguity_history else 1.0,
            'cycles_completed': min(cycle + 1, max_cycles),
            'ambiguity_history': self.ambiguity_history,
            'clarifications_applied': self.clarifications_applied,
            'clarification_history': self.clarifier.clarification_history
        }

        return final_results