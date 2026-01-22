"""
Basic tests for CTGV System
"""
import numpy as np
from ctgv import Shape, Gebit, CTGVEngine, GeometricDataModeler

def test_basic_propagation():
    """Test basic signal propagation"""
    origin = Gebit(Shape.ORIGIN, intensity=1.0, label="Source")
    flow = Gebit(Shape.FLOW, intensity=0.0, label="Channel")

    origin.connect_to(flow, 0.9)

    engine = CTGVEngine(threshold=0.001, max_iterations=100)
    result = engine.propagate([origin])

    assert result['converged'] == True
    assert result['iterations'] > 0
    assert 'Source' in result['final_states']
    print("✓ Basic propagation test passed")

def test_geometric_constraints():
    """Test that geometric constraints are enforced"""
    flow = Gebit(Shape.FLOW, intensity=1.0)

    # FLOW can only have 2 connections max
    assert flow.connect_to(Gebit(Shape.ORIGIN), 0.5) == True
    assert flow.connect_to(Gebit(Shape.ORIGIN), 0.5) == True
    assert flow.connect_to(Gebit(Shape.ORIGIN), 0.5) == False  # Should fail

    print("✓ Geometric constraints test passed")

def test_data_modeling():
    """Test pattern encoding/decoding"""
    pattern = np.array([[0.1, 0.9], [0.9, 0.1]], dtype=np.float32)

    modeler = GeometricDataModeler()
    network = modeler.encode_pattern(pattern)

    assert len(network) == 4  # 2x2 grid

    decoded = modeler.decode_pattern(network, pattern.shape)
    assert decoded.shape == pattern.shape

    print("✓ Data modeling test passed")

def test_cache_fix():
    """Test that interference cache is properly managed"""
    gebit1 = Gebit(Shape.ORIGIN, intensity=1.0)
    gebit2 = Gebit(Shape.FLOW, intensity=0.5)

    gebit1.connect_to(gebit2, 0.8)

    # First call should compute
    interference1 = gebit1.calculate_field_interaction(gebit2)

    # Second call should use cache
    interference2 = gebit1.calculate_field_interaction(gebit2)

    assert interference1 == interference2
    print("✓ Cache fix test passed")

if __name__ == "__main__":
    print("Running CTGV System tests...")
    test_basic_propagation()
    test_geometric_constraints()
    test_data_modeling()
    test_cache_fix()
    print("All tests passed! ✅")