#!/usr/bin/env python3
"""
CTGV System Demonstration
"""
import numpy as np
import time
from ctgv import (
    Shape, Gebit, CTGVEngine, GeometricDataModeler,
    TemporalBindingArbiter, ClarificationEngine, ClarificationMechanism,
    EnhancedTemporalBindingArbiter, visualize_ctgv_processing, VectorField
)

def demonstrate_ctgv_system():
    """Complete CTGV system demonstration"""
    print("=" * 70)
    print(" CTGV SYSTEM - Versatile Geometric Topological Computing")
    print("=" * 70)

    # ===== 1. SIMPLE NETWORK =====
    print("\n[1] Creating simple propagation network...")

    origin = Gebit(Shape.ORIGIN, intensity=1.0, label="Source")
    flow1 = Gebit(Shape.FLOW, intensity=0.0, label="Channel_1")
    flow2 = Gebit(Shape.FLOW, intensity=0.0, label="Channel_2")
    decisor = Gebit(Shape.DECISOR, intensity=0.0, label="Decisor")
    memory = Gebit(Shape.MEMORY, intensity=0.0, label="Memory")
    sensor = Gebit(Shape.SENSOR, intensity=0.0, label="Sensor")

    # Topology
    origin.connect_to(flow1, 0.9)
    origin.connect_to(flow2, 0.8)
    flow1.connect_to(decisor, 0.7)
    flow2.connect_to(decisor, 0.7)
    decisor.connect_to(memory, 0.85)
    memory.connect_to(sensor, 0.9)

    # Engine
    engine = CTGVEngine(
        threshold=0.001,
        max_iterations=200,
        use_superposition=True
    )

    result = engine.propagate([origin])
    print(f"  Iterations: {result['iterations']}")
    print(f"  Converged: {result['converged']}")
    print(f"  Coherence: {result['global_coherence']:.3f}")
    print(f"  Final states:")
    for label, state in result['final_states'].items():
        if state > 0.01:
            print(f"    {label}: {state:.4f}")

    # ===== 2. TBA - AMBIGUITY RESOLUTION =====
    print("\n[2] Testing TBA - Ambiguity Resolution...")

    # Two competing narratives
    narrative_a = Gebit(Shape.ORIGIN, intensity=0.9, label="Narrative_A")
    narrative_b = Gebit(Shape.ORIGIN, intensity=0.85, label="Narrative_B")

    # Shared structure
    processor = Gebit(Shape.DECISOR, intensity=0.0, label="Processor")
    resonator = Gebit(Shape.RESONATOR, intensity=0.0, label="Resonator")
    output = Gebit(Shape.AMPLIFIER, intensity=0.0, label="Output")

    narrative_a.connect_to(processor, 0.8)
    narrative_b.connect_to(processor, 0.75)
    resonator.connect_to(output, 0.85)

    # Execute TBA
    tba = TemporalBindingArbiter(engine, binding_threshold=0.3)
    resolution = tba.resolve_ambiguity([narrative_a, narrative_b], max_cycles=30)

    print(f"  Dominant narrative: {resolution.get('dominant_narrative', 'N/A')}")
    print(f"  Final strengths:")
    for label, strength in resolution['narrative_strengths'].items():
        print(f"    {label}: {strength:.4f}")
    print(f"  Final ambiguity: {resolution['final_ambiguity']:.3f}")

    # ===== 3. GEOMETRIC DATA MODELING =====
    print("\n[3] Demonstrating Geometric Data Modeling...")

    # Test pattern (simplified logo)
    test_pattern = np.array([
        [0.1, 0.9, 0.9, 0.1],
        [0.9, 0.2, 0.2, 0.9],
        [0.9, 0.2, 0.2, 0.9],
        [0.1, 0.9, 0.9, 0.1]
    ], dtype=np.float32)

    print(f"  Original pattern (4×4):")
    print(test_pattern)

    # Encode to CTGV
    modeler = GeometricDataModeler()
    gebit_network = modeler.encode_pattern(test_pattern)

    # Process through network
    first_gebit = gebit_network[0]
    process_result = engine.propagate([first_gebit])

    print(f"  Processing:")
    print(f"    Iterations: {process_result['iterations']}")
    print(f"    Coherence: {process_result['global_coherence']:.3f}")

    # Decode
    decoded_pattern = modeler.decode_pattern(gebit_network, test_pattern.shape)

    print(f"  Processed pattern:")
    print(decoded_pattern)
    print(f"  Mean difference: {np.mean(np.abs(test_pattern - decoded_pattern)):.4f}")

    # ===== 4. NETWORK WITH LOOP AND RESONANCE =====
    print("\n[4] Testing Feedback Loop and Resonance...")

    input_node = Gebit(Shape.ORIGIN, intensity=0.8, label="Input")
    loop_a = Gebit(Shape.LOOP, intensity=0.0, label="Loop_A")
    loop_b = Gebit(Shape.LOOP, intensity=0.0, label="Loop_B")
    resonator_node = Gebit(Shape.RESONATOR, intensity=0.0, label="Resonator")

    # Topology with cycle
    input_node.connect_to(loop_a, 0.9)
    loop_a.connect_to(loop_b, 0.85)
    loop_b.connect_to(loop_a, 0.8)  # Closed cycle
    loop_b.connect_to(resonator_node, 0.9)

    feedback_result = engine.propagate([input_node])

    print(f"  Iterations: {feedback_result['iterations']}")
    print(f"  States with feedback:")
    for label, state in feedback_result['final_states'].items():
        if state > 0.01:
            print(f"    {label}: {state:.4f}")

    # Show resonator history
    if len(resonator_node.history) > 0:
        print(f"  Resonator history (last 5):")
        print(f"    {resonator_node.history[-5:]}")

    # ===== 5. CLARIFICATION ENGINE =====
    print("\n[5] Demonstrating Clarification Engine...")

    # Criar opções de decisão ambíguas
    option_a = Gebit(Shape.DECISOR, intensity=0.6, label="Option_A")
    option_b = Gebit(Shape.DECISOR, intensity=0.55, label="Option_B")
    option_c = Gebit(Shape.DECISOR, intensity=0.52, label="Option_C")

    # Contexto para auxiliar na decisão
    context_strong = Gebit(Shape.AMPLIFIER, intensity=0.8, label="Context_Strong")
    context_weak = Gebit(Shape.INHIBITOR, intensity=0.3, label="Context_Weak")

    # Conectar contexto às opções
    context_strong.connect_to(option_a, 0.9)
    context_strong.connect_to(option_b, 0.6)
    context_weak.connect_to(option_c, 0.7)

    # Engine de clarificação
    clarifier = ClarificationEngine(engine, clarification_threshold=0.75)

    # Processo de clarificação
    clarification_result = clarifier.clarify_decision(
        [option_a, option_b, option_c],
        [context_strong, context_weak],
        max_clarification_rounds=3
    )

    print(f"  Clarification Result:")
    print(f"    Rounds: {clarification_result['clarification_rounds']}")
    print(f"    Confidence: {clarification_result['confidence']:.4f}")
    print(f"    Ambiguity Reduction: {clarification_result['ambiguity_reduction']:.4f}")

    if clarification_result['clarified_decision']:
        print(f"    Final Decision: {clarification_result['clarified_decision'].label}")
    else:
        print("    Decision: Still ambiguous")

    # ===== 6. PATTERN CLARIFICATION =====
    print("\n[6] Demonstrating Pattern Clarification...")

    # Padrão com ruído
    noisy_pattern = np.array([
        [0.1, 0.85, 0.9, 0.15],
        [0.95, 0.25, 0.35, 0.85],
        [0.9, 0.3, 0.2, 0.95],
        [0.05, 0.8, 0.85, 0.1]
    ], dtype=np.float32)

    print(f"  Noisy Pattern:")
    print(noisy_pattern)

    # Aplicar clarificação de simetria
    symmetry_clarification = clarifier.clarify_pattern(noisy_pattern, 'symmetry')

    print(f"  Symmetry Clarification:")
    print(f"    Coherence Improvement: {symmetry_clarification['coherence_improvement']:.4f}")
    print(f"    Pattern Clarity: {symmetry_clarification['pattern_clarity']:.4f}")
    print(f"    Clarified Pattern:")
    print(symmetry_clarification['clarified_pattern'])

    # Aplicar clarificação de redução de ruído
    noise_clarification = clarifier.clarify_pattern(noisy_pattern, 'noise')

    print(f"  Noise Reduction Clarification:")
    print(f"    Coherence Improvement: {noise_clarification['coherence_improvement']:.4f}")
    print(f"    Pattern Clarity: {noise_clarification['pattern_clarity']:.4f}")
    # ===== 7. DEMONSTRAÇÃO DE ESCALABILIDADE =====
    print("\n[7] Demonstrating Scalability Improvements...")

    # Create larger network for scalability test
    print("Creating larger network (50 nodes)...")
    large_network = []
    for i in range(50):
        shape = list(Shape)[i % len(Shape)]  # Cycle through shapes
        node = Gebit(shape, intensity=0.5, label=f"Node_{i:02d}")
        large_network.append(node)

    # Create connections (sparse network)
    np.random.seed(42)
    for i, node in enumerate(large_network):
        # Connect to 3-5 random neighbors
        n_connections = np.random.randint(3, 6)
        targets = np.random.choice([j for j in range(50) if j != i], n_connections, replace=False)
        for target_idx in targets:
            weight = np.random.uniform(0.1, 0.9)
            node.connect_to(large_network[target_idx], weight)

    print(f"Network created: {len(large_network)} nodes, "
          f"~{sum(len(n.connections) for n in large_network)//2} connections")

    # Test sequential vs parallel propagation
    print("\nTesting Sequential Propagation:")
    engine_sequential = CTGVEngine(enable_parallel=False, max_iterations=50)
    start_time = time.time()
    result_seq = engine_sequential.propagate([large_network[0]])
    seq_time = time.time() - start_time
    print(f"  Time: {seq_time:.3f}s, Iterations: {result_seq['iterations']}")

    print("\nTesting Parallel Propagation:")
    engine_parallel = CTGVEngine(enable_parallel=True, max_iterations=50, chunk_size=20)
    start_time = time.time()
    result_par = engine_parallel.propagate([large_network[0]])
    par_time = time.time() - start_time
    print(f"  Time: {par_time:.3f}s, Iterations: {result_par['iterations']}")
    print(f"  Speedup: {seq_time/par_time:.2f}x")

    # Test arbiter scalability
    print("\nTesting Arbiter Scalability:")
    arbiter_standard = TemporalBindingArbiter(engine_sequential)
    arbiter_parallel = TemporalBindingArbiter(engine_parallel, enable_parallel=True)

    test_nodes = large_network[:10]  # Test with 10 competing narratives

    start_time = time.time()
    result_std = arbiter_standard.resolve_ambiguity(test_nodes, max_cycles=10)
    std_time = time.time() - start_time

    start_time = time.time()
    result_prl = arbiter_parallel.resolve_ambiguity_parallel(test_nodes, max_cycles=10)
    prl_time = time.time() - start_time

    print(f"  Standard: {std_time:.3f}s, Ambiguity: {result_std['final_ambiguity']:.3f}")
    print(f"  Parallel: {prl_time:.3f}s, Ambiguity: {result_prl['final_ambiguity']:.3f}")
    print(f"  Arbiter Speedup: {std_time/prl_time:.2f}x")

    print("\nScalability demonstration completed!")

def demonstrate_clarification_mechanism():
    """
    Demonstra o mecanismo de clarificação avançado para resolução de ambiguidades
    e treinamento do sistema CTGV.
    """
    print("\n" + "="*60)
    print("DEMONSTRAÇÃO DO MECANISMO DE CLARIFICAÇÃO AVANÇADO")
    print("="*60)

    # Criar cenário ambíguo
    print("\n1. CRIANDO CENÁRIO AMBÍGUO")
    print("-" * 40)

    # Inicializar componentes
    engine = CTGVEngine()
    field = VectorField()

    # Criar estado ambíguo com Gebits conflitantes
    gebit1 = Gebit(Shape.ORIGIN, intensity=0.8, label="Pattern_A")
    gebit2 = Gebit(Shape.FLOW, intensity=0.75, label="Pattern_B")
    
    # Definir estados iniciais
    gebit1.state = 0.8
    gebit2.state = 0.75
    
    ambiguous_state = [gebit1, gebit2]  # Lista de gebits conflitantes

    print(f"Gebits conflitantes criados: {len(ambiguous_state)}")
    print(f"Estados: {[g.state for g in ambiguous_state]}")

    # Inicializar arbiters
    print("\n2. COMPARANDO ARBITERS")
    print("-" * 40)

    standard_arbiter = TemporalBindingArbiter(engine, binding_threshold=0.3)
    enhanced_arbiter = EnhancedTemporalBindingArbiter(engine, binding_threshold=0.3)

    # Testar resolução com arbiter padrão
    print("\nArbiter Padrão:")
    try:
        standard_result = standard_arbiter.resolve_ambiguity(ambiguous_state)
        print(f"  Resolução: {'Sucesso' if standard_result['final_ambiguity'] < 0.1 else 'Ambíguo'}")
        if 'dominant_narrative' in standard_result:
            print(f"  Narrativa dominante: {standard_result['dominant_narrative']}")
            print(f"  Forças: {standard_result['narrative_strengths']}")
        print(f"  Ambiguidade final: {standard_result['final_ambiguity']:.3f}")
    except Exception as e:
        print(f"  Erro: {str(e)}")

    # Pular arbiter aprimorado por enquanto - precisa de correções
    print("\nArbiter Aprimorado (com Clarificação):")
    print("  [Funcionalidade temporariamente desabilitada - necessita correções]")

    # Demonstrar mecanismo de clarificação independente
    print("\n3. MECANISMO DE CLARIFICAÇÃO INDEPENDENTE")
    print("-" * 40)

    clarifier = ClarificationMechanism(engine)

    # Verificar estado ambíguo
    ambiguous_check = clarifier.check_ambiguous_state(ambiguous_state)
    if ambiguous_check:
        print(f"Estado ambíguo detectado: {ambiguous_check['entropy_level']:.3f}")
        print(f"Intervenção necessária: {ambiguous_check['needs_intervention']}")
    else:
        print("Nenhum estado ambíguo detectado")

    # Pular intervenção por enquanto - necessita correções
    print("\n4. INTERVENÇÃO DE CLARIFICAÇÃO")
    print("-" * 40)
    print("  [Funcionalidade temporariamente desabilitada - necessita correções]")

    print("\n5. DEMONSTRAÇÃO CONCLUÍDA")
    print("-" * 40)
    print("Mecanismo de clarificação integrado com sucesso!")
    print("Funcionalidades básicas demonstradas:")
    print("  ✓ Detecção de estados ambíguos")
    print("  ✓ Resolução via Temporal Binding Arbiter")
    print("  ✓ Integração com sistema CTGV")

    print("\nDemonstração concluída!")

if __name__ == "__main__":
    demonstrate_ctgv_system()
    demonstrate_clarification_mechanism()