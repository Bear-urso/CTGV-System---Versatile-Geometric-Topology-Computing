"""
CTGV System - Clarification Engine
Módulo para clarificação de decisões e resolução de ambiguidades
"""
from typing import List, Dict, Optional, Tuple
import numpy as np
from .gebit import Gebit
from .engine import CTGVEngine
from .shapes import Shape

class ClarificationEngine:
    """
    Engine de clarificação para esclarecer decisões e reduzir ambiguidades
    no sistema CTGV através de análise de coerência e refinamento iterativo
    """

    def __init__(self, base_engine: CTGVEngine, clarification_threshold: float = 0.7):
        self.base_engine = base_engine
        self.clarification_threshold = clarification_threshold

    def clarify_decision(self, decision_gebits: List[Gebit],
                        context_gebits: List[Gebit] = None,
                        max_clarification_rounds: int = 5) -> Dict:
        """
        Processo de clarificação para tomada de decisão

        Args:
            decision_gebits: Gebits representando opções de decisão
            context_gebits: Gebits de contexto para auxiliar na clarificação
            max_clarification_rounds: Máximo de rodadas de clarificação

        Returns:
            Resultado da clarificação com decisão final
        """
        if not decision_gebits:
            return {
                'clarified_decision': None,
                'confidence': 0.0,
                'clarification_rounds': 0,
                'ambiguity_reduction': 0.0
            }

        context_gebits = context_gebits or []

        # Estado inicial
        initial_ambiguity = self._calculate_decision_ambiguity(decision_gebits)
        current_decisions = decision_gebits.copy()
        clarification_rounds = 0

        # Processo iterativo de clarificação
        while clarification_rounds < max_clarification_rounds:
            # Propaga sinal através do contexto
            if context_gebits:
                context_result = self.base_engine.propagate(context_gebits)
                context_coherence = context_result['global_coherence']

                # Aplica influência do contexto nas decisões
                self._apply_context_influence(current_decisions, context_gebits, context_coherence)

            # Propaga decisões
            decision_result = self.base_engine.propagate(current_decisions)
            current_coherence = decision_result['global_coherence']

            # Verifica se atingiu threshold de clarificação
            if current_coherence >= self.clarification_threshold:
                break

            # Refina decisões baseado na coerência atual
            current_decisions = self._refine_decisions(current_decisions, decision_result)

            clarification_rounds += 1

        # Análise final
        final_ambiguity = self._calculate_decision_ambiguity(current_decisions)
        ambiguity_reduction = initial_ambiguity - final_ambiguity

        # Determina decisão final
        final_decision = self._determine_final_decision(current_decisions, decision_result)

        return {
            'clarified_decision': final_decision,
            'confidence': decision_result['global_coherence'],
            'clarification_rounds': clarification_rounds,
            'ambiguity_reduction': ambiguity_reduction,
            'final_states': decision_result['final_states'],
            'decision_evolution': [g.state for g in current_decisions]
        }

    def clarify_pattern(self, pattern: np.ndarray,
                       clarification_focus: str = 'symmetry') -> Dict:
        """
        Clarifica padrões através de análise topológica

        Args:
            pattern: Padrão 2D a ser clarificado
            clarification_focus: Tipo de clarificação ('symmetry', 'structure', 'noise')

        Returns:
            Padrão clarificado e métricas
        """
        from .modeler import GeometricDataModeler

        modeler = GeometricDataModeler()
        network = modeler.encode_pattern(pattern)

        # Análise inicial
        initial_result = self.base_engine.propagate([network[0]])
        initial_coherence = initial_result['global_coherence']

        # Aplicação de foco de clarificação
        clarified_network = self._apply_clarification_focus(network, clarification_focus)

        # Análise após clarificação
        clarified_result = self.base_engine.propagate([clarified_network[0]])
        clarified_coherence = clarified_result['global_coherence']

        # Decodificação do padrão clarificado
        clarified_pattern = modeler.decode_pattern(clarified_network, pattern.shape)

        # Métricas de clarificação
        coherence_improvement = clarified_coherence - initial_coherence
        pattern_clarity = self._calculate_pattern_clarity(clarified_pattern)

        return {
            'original_pattern': pattern,
            'clarified_pattern': clarified_pattern,
            'coherence_improvement': coherence_improvement,
            'pattern_clarity': pattern_clarity,
            'clarification_focus': clarification_focus,
            'initial_coherence': initial_coherence,
            'final_coherence': clarified_coherence
        }

    def _calculate_decision_ambiguity(self, decision_gebits: List[Gebit]) -> float:
        """Calcula ambiguidade nas decisões"""
        if len(decision_gebits) < 2:
            return 0.0

        states = [g.state for g in decision_gebits]
        mean_state = np.mean(states)
        variance = np.var(states)

        # Ambiguidade = variância normalizada
        return variance / (mean_state**2 + 1e-10) if mean_state > 0 else 0.0

    def _apply_context_influence(self, decisions: List[Gebit],
                               context: List[Gebit], context_coherence: float):
        """Aplica influência do contexto nas decisões"""
        context_states = [g.state for g in context]
        avg_context = np.mean(context_states)

        # Modula decisões baseado no contexto
        for decision in decisions:
            # Influência proporcional à coerência do contexto
            influence = (avg_context - 0.5) * context_coherence * 0.1
            decision.state = np.clip(decision.state + influence, 0.0, 1.0)

    def _refine_decisions(self, decisions: List[Gebit], result: Dict) -> List[Gebit]:
        """Refina decisões baseado nos resultados atuais"""
        # Remove decisões com estados muito baixos (threshold adaptativo)
        active_threshold = np.mean([g.state for g in decisions]) * 0.5

        refined = []
        for decision in decisions:
            if decision.state >= active_threshold:
                # Amplifica decisões fortes, atenua fracas
                if decision.state > 0.7:
                    decision.state = min(1.0, decision.state * 1.1)
                elif decision.state < 0.3:
                    decision.state = max(0.0, decision.state * 0.9)
                refined.append(decision)

        return refined if refined else decisions

    def _determine_final_decision(self, decisions: List[Gebit], result: Dict) -> Optional[Gebit]:
        """Determina a decisão final baseada nos estados"""
        if not decisions:
            return None

        # Seleciona decisão com maior estado
        best_decision = max(decisions, key=lambda g: g.state)

        # Verifica se é significativamente melhor que as outras
        other_states = [g.state for g in decisions if g != best_decision]
        if other_states:
            avg_other = np.mean(other_states)
            if best_decision.state > avg_other * 1.5:  # Pelo menos 50% melhor
                return best_decision

        return None  # Ainda ambíguo

    def _apply_clarification_focus(self, network: List[Gebit], focus: str) -> List[Gebit]:
        """Aplica foco específico de clarificação ao network"""
        clarified = [g for g in network]  # Cópia

        if focus == 'symmetry':
            # Reforça simetrias no padrão
            self._enhance_symmetry(clarified)
        elif focus == 'structure':
            # Reforça estruturas coerentes
            self._enhance_structure(clarified)
        elif focus == 'noise':
            # Reduz ruído e inconsistências
            self._reduce_noise(clarified)

        return clarified

    def _enhance_symmetry(self, network: List[Gebit]):
        """Melhora simetria no network"""
        # Identifica e reforça padrões simétricos
        size = int(np.sqrt(len(network)))
        for i in range(size):
            for j in range(size // 2):
                left = network[i * size + j]
                right = network[i * size + (size - 1 - j)]

                # Média ponderada para simetria
                avg_state = (left.state + right.state) / 2
                left.state = right.state = avg_state

    def _enhance_structure(self, network: List[Gebit]):
        """Melhora estrutura coerente no network"""
        # Reforça conexões locais fortes
        for gebit in network:
            neighbors = [n for n in gebit.connections.keys()]
            if neighbors:
                avg_neighbor = np.mean([n.state for n in neighbors])
                # Puxa estado na direção dos vizinhos
                gebit.state = 0.7 * gebit.state + 0.3 * avg_neighbor

    def _reduce_noise(self, network: List[Gebit]):
        """Reduz ruído no network"""
        # Suavização baseada em vizinhos
        for gebit in network:
            neighbors = [n for n in gebit.connections.keys()]
            if len(neighbors) >= 2:
                neighbor_states = [n.state for n in neighbors]
                median_neighbor = np.median(neighbor_states)

                # Se estado diverge muito da mediana dos vizinhos, ajusta
                diff = abs(gebit.state - median_neighbor)
                if diff > 0.3:
                    gebit.state = 0.6 * gebit.state + 0.4 * median_neighbor

    def _calculate_pattern_clarity(self, pattern: np.ndarray) -> float:
        """Calcula clareza do padrão (inverso da entropia)"""
        # Simplificação: clareza baseada na variância
        # Padrões com alta variância (contraste) são considerados mais claros
        variance = np.var(pattern)
        mean = np.mean(pattern)

        # Clareza = variância normalizada pelo quadrado da média
        return variance / (mean**2 + 1e-10) if mean > 0 else 0.0